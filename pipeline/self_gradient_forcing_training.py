from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from agf.utils.scheduler import SchedulerInterface
from agf.utils.wan_wrapper import WanDiffusionWrapper


@dataclass
class SelfGradientForcingState:
    """The bounded state recorded by the no-grad SGF rollout."""

    initial_noise: torch.Tensor
    noisy_at_t: torch.Tensor
    context_latents: torch.Tensor
    reconstruction_timestep: torch.Tensor
    denoised_timestep_from: int
    denoised_timestep_to: int
    exit_step_index: int


class SelfGradientForcingTrainingPipeline:
    """Two-pass Self Gradient Forcing training pipeline.

    Pass 1 follows the serial KV-cache inference path without retaining a graph.
    Pass 2 reconstructs the training window in parallel through the causal
    teacher-forcing mask, allowing future losses to train the context writer.
    """

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        grad_mode: str = "adversarial_self_gradient_forcing",
        per_rank_exit_step: bool = True,
        self_gradient_forcing_match_context: bool = False,
        self_gradient_forcing_cache_mode: str = "exit",
    ):
        super().__init__()
        if grad_mode not in (
            "self_gradient_forcing",
            "adversarial_self_gradient_forcing",
        ):
            raise ValueError(
                "grad_mode must be 'self_gradient_forcing' or "
                f"'adversarial_self_gradient_forcing', got {grad_mode!r}"
            )
        if independent_first_frame:
            raise NotImplementedError(
                "Adversarial history SGF currently supports text-to-video only"
            )
        if self_gradient_forcing_cache_mode not in ("exit", "final_clean"):
            raise ValueError(
                "self_gradient_forcing_cache_mode must be 'exit' or "
                f"'final_clean', got {self_gradient_forcing_cache_mode!r}"
            )

        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = int(num_frame_per_block)
        self.context_noise = int(context_noise)
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.num_max_frames = int(num_max_frames)
        if self.num_max_frames <= 0 or self.num_max_frames % self.num_frame_per_block:
            raise ValueError("num_max_frames must be a positive multiple of num_frame_per_block")
        self.kv_cache_size = self.num_max_frames * self.frame_seq_length
        self.per_rank_exit_step = bool(per_rank_exit_step)
        self.self_gradient_forcing_match_context = bool(self_gradient_forcing_match_context)
        self.self_gradient_forcing_cache_mode = self_gradient_forcing_cache_mode

        self.kv_cache = None
        self.crossattn_cache = None

    def _sample_exit_step(self, num_denoising_steps: int, device) -> int:
        if self.self_gradient_forcing_cache_mode == "exit" and self.per_rank_exit_step:
            index = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device,
            )
            if self.last_step_only:
                index.fill_(num_denoising_steps - 1)
            return int(index.item())

        if not dist.is_available() or not dist.is_initialized():
            index = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device,
            )
            if self.last_step_only:
                index.fill_(num_denoising_steps - 1)
            return int(index.item())

        if dist.get_rank() == 0:
            index = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device,
            )
            if self.last_step_only:
                index.fill_(num_denoising_steps - 1)
        else:
            index = torch.empty(1, dtype=torch.long, device=device)
        dist.broadcast(index, src=0)
        return int(index.item())

    def rollout(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        **conditional_dict,
    ) -> SelfGradientForcingState:
        """Run the serial on-policy pass and record a bounded SGF state."""
        if initial_latent is not None:
            raise NotImplementedError(
                "Adversarial history SGF currently supports text-to-video only"
            )

        batch_size, num_frames = noise.shape[:2]
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by "
                f"num_frame_per_block={self.num_frame_per_block}"
            )
        if num_frames != self.num_max_frames:
            raise ValueError("The rollout must match the configured training window")

        device, dtype = noise.device, noise.dtype
        num_blocks = num_frames // self.num_frame_per_block
        local_num_frames = num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_step_index = self._sample_exit_step(
            num_denoising_steps=num_denoising_steps,
            device=device,
        )
        train_timestep = self.denoising_step_list[exit_step_index]

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        local_shape = (batch_size, local_num_frames, *noise.shape[2:])
        recorded_initial_noise = noise.new_zeros(local_shape)
        recorded_context = noise.new_zeros(local_shape)
        recorded_cache_context = noise.new_zeros(local_shape)
        noisy_at_t = noise.new_zeros(local_shape)

        current_start_frame = 0
        with torch.no_grad():
            for _ in range(num_blocks):
                current_num_frames = self.num_frame_per_block
                rollout_slice = slice(
                    current_start_frame,
                    current_start_frame + current_num_frames,
                )
                local_slice = rollout_slice
                noisy_input = noise[:, rollout_slice]
                recorded_initial_noise[:, local_slice] = noisy_input
                x0_exit = None
                x0_final = None
                timestep = None

                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=device,
                            dtype=torch.int64,
                        )
                        * current_timestep
                    )

                    if index == exit_step_index:
                        noisy_at_t[:, local_slice] = noisy_input

                    _, x0 = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache,
                        crossattn_cache=self.crossattn_cache,
                        current_start=(current_start_frame * self.frame_seq_length),
                    )
                    x0_final = x0

                    if index == exit_step_index:
                        x0_exit = x0
                        # A synchronized exit can stop immediately. Per-rank
                        # exits must execute every FSDP forward on every rank.
                        if (
                            self.self_gradient_forcing_cache_mode == "exit"
                            and not self.per_rank_exit_step
                        ):
                            break

                    if index < num_denoising_steps - 1:
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            x0.flatten(0, 1),
                            torch.randn_like(x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, x0.shape[:2])

                if x0_exit is None or x0_final is None or timestep is None:
                    raise RuntimeError("Failed to record the SGF denoising exit state")

                if self.self_gradient_forcing_cache_mode == "final_clean":
                    context_latent = x0_final
                else:
                    context_latent = x0_exit
                recorded_context[:, local_slice] = context_latent

                context_timestep = torch.ones_like(timestep) * self.context_noise
                cache_context = self.scheduler.add_noise(
                    context_latent.flatten(0, 1),
                    torch.randn_like(context_latent.flatten(0, 1)),
                    context_timestep.flatten(0, 1).long(),
                ).unflatten(0, context_latent.shape[:2])
                recorded_cache_context[:, local_slice] = cache_context

                self.generator(
                    noisy_image_or_video=cache_context,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame * self.frame_seq_length),
                )
                current_start_frame += current_num_frames

        reconstruction_timestep = (
            torch.ones(
                [batch_size, local_num_frames],
                device=device,
                dtype=torch.int64,
            )
            * train_timestep
        )

        if self.self_gradient_forcing_match_context:
            pass2_context = recorded_cache_context.detach()
        else:
            pass2_context = recorded_context.detach()

        denoised_timestep_from, denoised_timestep_to = self._denoised_timestep_bounds(
            exit_step_index, device
        )

        # The serial cache is no longer needed by any Pass-2 branch. Dropping
        # the references lets the caching allocator reuse several GB.
        self.kv_cache = None
        self.crossattn_cache = None

        return SelfGradientForcingState(
            initial_noise=recorded_initial_noise.detach(),
            noisy_at_t=noisy_at_t,
            context_latents=pass2_context,
            reconstruction_timestep=reconstruction_timestep,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            exit_step_index=exit_step_index,
        )

    def reconstruct(
        self,
        state: SelfGradientForcingState,
        conditional_dict: dict,
        context_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the differentiable parallel SGF reconstruction pass."""
        if context_latents is None:
            context_latents = state.context_latents

        if self.self_gradient_forcing_match_context:
            aug_t = torch.ones_like(state.reconstruction_timestep) * self.context_noise
        else:
            aug_t = torch.zeros_like(state.reconstruction_timestep)

        _, output = self.generator(
            noisy_image_or_video=state.noisy_at_t,
            conditional_dict=conditional_dict,
            timestep=state.reconstruction_timestep,
            clean_x=context_latents,
            aug_t=aug_t,
        )
        return output

    def predict_first_step_flow_velocity(
        self,
        state: SelfGradientForcingState,
        conditional_dict: dict,
        context_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict native Wan flow at the shared highest-noise student step.

        ``state.initial_noise`` is identical for clean and adversarial calls;
        only ``context_latents`` changes.  Returning the generator's first
        output deliberately avoids converting velocity to a final latent.
        """
        if context_latents is None:
            context_latents = state.context_latents
        if context_latents.shape != state.initial_noise.shape:
            raise ValueError("first-step velocity history and initial noise must match")

        first_timestep = self.denoising_step_list[0].to(device=state.initial_noise.device)
        timestep = torch.ones_like(state.reconstruction_timestep) * first_timestep
        if self.self_gradient_forcing_match_context:
            aug_t = torch.ones_like(timestep) * self.context_noise
        else:
            aug_t = torch.zeros_like(timestep)

        flow_velocity, _ = self.generator(
            noisy_image_or_video=state.initial_noise,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=context_latents,
            aug_t=aug_t,
        )
        return flow_velocity

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ):
        """Compatibility entry point for clean SGF training."""
        state = self.rollout(
            noise=noise,
            initial_latent=initial_latent,
            **conditional_dict,
        )
        output = self.reconstruct(
            state=state,
            conditional_dict=conditional_dict,
        )
        if return_sim_step:
            return (
                output,
                state.denoised_timestep_from,
                state.denoised_timestep_to,
                state.exit_step_index + 1,
            )
        return (
            output,
            state.denoised_timestep_from,
            state.denoised_timestep_to,
        )

    def _denoised_timestep_bounds(self, exit_step_index: int, device):
        current = self.denoising_step_list[exit_step_index].to(device)
        denoised_timestep_from = (
            1000 - torch.argmin((self.scheduler.timesteps.to(device) - current).abs(), dim=0).item()
        )

        if exit_step_index == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
        else:
            following = self.denoising_step_list[exit_step_index + 1].to(device)
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (self.scheduler.timesteps.to(device) - following).abs(),
                    dim=0,
                ).item()
            )
        return denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache = []
        for _ in range(self.num_transformer_blocks):
            self.kv_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, self.kv_cache_size, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "use_relative_sink": False,
                }
            )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            self.crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "is_init": False,
                }
            )
