from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from agf.model.dmd import DMD
from agf.model.prediction_consistency import (
    target_ratio,
    dmd_output_gradient_rms,
    backward_prediction_consistency,
)
from agf.pipeline import SelfGradientForcingTrainingPipeline
from agf.utils.wan_wrapper import WanDiffusionWrapper


class AdversarialHistoryDMD(DMD):
    """DMD with SGF reconstruction and a future-guided history adversary."""

    handles_generator_backward = True
    uses_fake_replay = True

    def __init__(self, args, device):
        super().__init__(args, device)
        if self.independent_first_frame or getattr(args, "i2v", False):
            raise NotImplementedError(
                "Adversarial History SGF currently supports text-to-video only"
            )
        if self.num_training_frames % self.num_frame_per_block != 0:
            raise ValueError("num_training_frames must be divisible by num_frame_per_block")

        self.num_training_chunks = self.num_training_frames // self.num_frame_per_block
        self.attack_chunk_choices = [int(chunk) for chunk in args.attack_chunk_choices]
        for chunk in self.attack_chunk_choices:
            if chunk <= 0 or chunk >= self.num_training_chunks - 1:
                raise ValueError(
                    f"attack chunk {chunk} must leave both a prefix and future "
                    f"inside {self.num_training_chunks} chunks"
                )

        self.attack_epsilon_rms = float(args.attack_epsilon_rms)
        if self.attack_epsilon_rms < 0:
            raise ValueError("attack_epsilon_rms must be non-negative")
        self.clean_loss_weight = float(args.clean_loss_weight)
        self.adv_loss_weight = float(args.adv_loss_weight)
        if self.clean_loss_weight < 0 or self.adv_loss_weight < 0:
            raise ValueError("clean/adv loss weights must be non-negative")
        if abs(self.clean_loss_weight + self.adv_loss_weight - 1.0) > 1e-8:
            raise ValueError("clean_loss_weight and adv_loss_weight must sum to 1")

        if not getattr(args, "attack_future_only", True):
            raise NotImplementedError("AGF requires attack_future_only=true")
        if not getattr(args, "attack_temporal_shared", True):
            raise NotImplementedError("AGF requires attack_temporal_shared=true")
        if getattr(args, "attack_steps", 1) != 1:
            raise NotImplementedError("Only one-step DMD-FGSM is supported")
        if getattr(args, "attack_random_start", False):
            raise NotImplementedError("The designed DMD-FGSM attack has no random start")

        replay_dtype_name = str(getattr(args, "fake_replay_dtype", "bf16")).lower()
        replay_dtypes = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if replay_dtype_name not in replay_dtypes:
            raise ValueError(f"Unsupported fake_replay_dtype={replay_dtype_name!r}")
        replay_device_name = str(getattr(args, "fake_replay_device", "cpu")).lower()
        if replay_device_name == "cpu":
            self.fake_replay_device = torch.device("cpu")
        elif replay_device_name in {"cuda", "gpu"}:
            self.fake_replay_device = torch.device("cuda", int(device))
        else:
            raise ValueError(
                f"fake_replay_device must be 'cpu' or 'cuda', got {replay_device_name!r}"
            )
        self.fake_replay_dtype = replay_dtypes[replay_dtype_name]
        self._pending_replay_samples = []
        self.inference_pipeline: Optional[SelfGradientForcingTrainingPipeline] = None

    def _initialize_sgf_pipeline(self):
        self.inference_pipeline = SelfGradientForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            same_step_across_blocks=self.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            grad_mode=getattr(
                self.args,
                "grad_mode",
                "adversarial_self_gradient_forcing",
            ),
            per_rank_exit_step=getattr(self.args, "per_rank_exit_step", True),
            self_gradient_forcing_match_context=getattr(
                self.args,
                "self_gradient_forcing_match_context",
                False,
            ),
            self_gradient_forcing_cache_mode=getattr(
                self.args,
                "self_gradient_forcing_cache_mode",
                "exit",
            ),
        )

    @contextmanager
    def _input_gradient_only(self):
        """Exclude generator parameters from the inner attack backward.

        FSDP's parameter post-backward hooks are incompatible with an
        ``autograd.grad`` input-gradient pass between accumulated
        ``no_sync()`` microbatches. The FGSM inner problem only needs the
        gradient with respect to the history perturbation, so temporarily
        freezing every generator parameter preserves the requested input
        gradient while leaving FSDP's accumulated parameter gradients intact.
        """
        parameters = list(self.generator.parameters())
        requires_grad = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, enabled in zip(parameters, requires_grad):
                parameter.requires_grad_(enabled)

    def _rollout(
        self,
        image_or_video_shape,
        conditional_dict,
    ):
        if self.inference_pipeline is None:
            self._initialize_sgf_pipeline()
        noise = torch.randn(
            image_or_video_shape,
            device=self.device,
            dtype=self.dtype,
        )
        return self.inference_pipeline.rollout(noise=noise, **conditional_dict)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        return torch.sqrt(value.float().square().mean(dim=dimensions, keepdim=True))

    @classmethod
    def _normalize_rms(cls, value: torch.Tensor) -> torch.Tensor:
        value = torch.nan_to_num(value.float())
        return value / (cls._rms(value) + 1.0e-6)

    def _sample_attack_chunk(self) -> int:
        index = torch.randint(
            low=0,
            high=len(self.attack_chunk_choices),
            size=(1,),
            device=self.device,
        )
        return self.attack_chunk_choices[int(index.item())]

    def _future_mask(
        self,
        reference: torch.Tensor,
        attack_chunk: int,
    ) -> torch.Tensor:
        future_start = (attack_chunk + 1) * self.num_frame_per_block
        if future_start <= 0 or future_start >= reference.shape[1]:
            raise ValueError("The attacked chunk must leave a non-empty scored future")
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[:, future_start:] = True
        return mask

    def _perturb_context(
        self,
        context: torch.Tensor,
        attack_chunk: int,
        shared_direction: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        start = attack_chunk * self.num_frame_per_block
        end = start + self.num_frame_per_block
        delta = (epsilon * shared_direction.repeat(1, self.num_frame_per_block, 1, 1, 1)).to(
            dtype=context.dtype
        )
        return torch.cat(
            [context[:, :start], context[:, start:end] + delta, context[:, end:]],
            dim=1,
        )

    def _distribution_matching_gradient(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift
                    * (timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                    * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                )
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )
            gradient, log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=image_or_video,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )
        return gradient, log_dict

    @staticmethod
    def _dmd_pseudo_loss(
        image_or_video: torch.Tensor,
        gradient: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original = image_or_video.double()
        target = (original - gradient.double()).detach()
        if gradient_mask is not None:
            return 0.5 * F.mse_loss(
                original[gradient_mask],
                target[gradient_mask],
                reduction="mean",
            )
        return 0.5 * F.mse_loss(original, target, reduction="mean")

    def _record_replay(
        self,
        output: torch.Tensor,
        conditional_dict: dict,
        kind: str,
    ):
        self._pending_replay_samples.append(
            {
                "generated_image": output.detach().to(
                    device=self.fake_replay_device,
                    dtype=self.fake_replay_dtype,
                ),
                "conditional_dict": {
                    key: value.detach().to(
                        device=self.fake_replay_device,
                        dtype=self.fake_replay_dtype,
                    )
                    for key, value in conditional_dict.items()
                },
                "kind": kind,
            }
        )

    def pop_replay_samples(self):
        samples = self._pending_replay_samples
        self._pending_replay_samples = []
        return samples

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        training_step: int = 0,
        loss_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Run one adversarial generator microbatch and backprop in-place.

        Backward happens inside this method so the clean reconstruction graph can
        be released before the robust reconstruction is built.
        """
        if initial_latent is not None:
            raise NotImplementedError(
                "Adversarial History SGF currently supports text-to-video only"
            )
        self._pending_replay_samples = []
        state = self._rollout(
            image_or_video_shape,
            conditional_dict,
        )
        ratio = target_ratio(self.args, training_step)
        regularization_log = {}
        attack_chunk = self._sample_attack_chunk()
        future_mask = self._future_mask(
            state.context_latents,
            attack_chunk,
        )

        epsilon = self.attack_epsilon_rms
        robust_weight = self.adv_loss_weight
        clean_weight = self.clean_loss_weight
        shared_perturbation = torch.zeros(
            [
                image_or_video_shape[0],
                1,
                image_or_video_shape[2],
                image_or_video_shape[3],
                image_or_video_shape[4],
            ],
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        with self._input_gradient_only():
            attack_context = self._perturb_context(
                state.context_latents,
                attack_chunk,
                shared_perturbation,
                epsilon,
            )
            inner_output = self.inference_pipeline.reconstruct(
                state=state,
                conditional_dict=conditional_dict,
                context_latents=attack_context,
            )
            clean_gradient, clean_log = self._distribution_matching_gradient(
                image_or_video=inner_output,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                denoised_timestep_from=(state.denoised_timestep_from),
                denoised_timestep_to=state.denoised_timestep_to,
            )
            attack_loss = self._dmd_pseudo_loss(
                inner_output,
                clean_gradient,
                gradient_mask=future_mask,
            )
            # ``_dmd_pseudo_loss`` is a mean over roughly a million future
            # latent values. Its direction is correct, but the resulting
            # gradient is smaller than the 1e-6 stabilizer used by RMS
            # normalization. Undo that mean only for the inner
            # maximization: multiplying by a positive scalar leaves the
            # FGSM direction unchanged while making the requested epsilon
            # an actual RMS radius. The clean/robust outer losses remain
            # mean-reduced.
            attack_objective = attack_loss * future_mask.sum(dtype=torch.float32).detach()
            attack_gradient = torch.autograd.grad(
                attack_objective,
                shared_perturbation,
                retain_graph=False,
                create_graph=False,
            )[0]
            attack_grad_rms = self._rms(attack_gradient).mean().detach()
            worst_direction = self._normalize_rms(attack_gradient).detach()
            del (
                attack_context,
                attack_gradient,
                attack_loss,
                attack_objective,
                inner_output,
                shared_perturbation,
            )

        # FSDP parameters are re-sharded by autograd.grad, so the same
        # activation-checkpointed graph cannot safely be reused for the
        # outer clean backward. Reconstruct deterministically and apply the
        # exact DMD gradient field sampled above.
        clean_output = self.inference_pipeline.reconstruct(
            state=state,
            conditional_dict=conditional_dict,
        )
        clean_loss = self._dmd_pseudo_loss(
            clean_output,
            clean_gradient,
        )
        clean_rms = dmd_output_gradient_rms(clean_gradient, clean_output) if ratio > 0 else None
        (float(loss_scale) * clean_weight * clean_loss).backward()
        self._record_replay(
            clean_output,
            conditional_dict,
            kind="clean",
        )
        del clean_output, clean_gradient

        adversarial_context = self._perturb_context(
            state.context_latents,
            attack_chunk,
            worst_direction,
            epsilon,
        )
        robust_output = self.inference_pipeline.reconstruct(
            state=state,
            conditional_dict=conditional_dict,
            context_latents=adversarial_context,
        )
        robust_gradient, robust_log = self._distribution_matching_gradient(
            image_or_video=robust_output,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=(state.denoised_timestep_from),
            denoised_timestep_to=state.denoised_timestep_to,
        )
        robust_loss = self._dmd_pseudo_loss(
            robust_output,
            robust_gradient,
            gradient_mask=future_mask,
        )
        base_rms = (
            (
                clean_weight * clean_rms
                + robust_weight
                * dmd_output_gradient_rms(robust_gradient, robust_output, future_mask)
            ).detach()
            if ratio > 0
            else None
        )
        (float(loss_scale) * robust_weight * robust_loss).backward()
        self._record_replay(
            robust_output,
            conditional_dict,
            kind="robust",
        )
        robust_loss_value = robust_loss.detach()
        robust_gradient_norm = robust_log["dmdtrain_gradient_norm"]
        attack_delta_rms = self._rms(epsilon * worst_direction).mean().detach()
        if ratio > 0:
            del robust_output, robust_gradient, robust_loss
            regularization_log = backward_prediction_consistency(
                self,
                state=state,
                conditional_dict=conditional_dict,
                adversarial_context=adversarial_context,
                future_mask=future_mask,
                base_rms=base_rms,
                ratio=ratio,
                loss_scale=loss_scale,
            )
        total_loss_value = clean_weight * clean_loss.detach() + robust_weight * robust_loss_value
        if regularization_log:
            total_loss_value = total_loss_value + regularization_log["regularization_loss"]

        clean_gradient_norm = clean_log["dmdtrain_gradient_norm"]
        combined_gradient_norm = (
            1.0 - robust_weight
        ) * clean_gradient_norm + robust_weight * robust_gradient_norm
        log_dict = {
            "generator_loss": total_loss_value,
            "clean_dmd_loss": clean_loss.detach(),
            "robust_dmd_loss": robust_loss_value,
            "dmdtrain_gradient_norm": combined_gradient_norm.detach(),
            "clean_dmd_gradient_norm": clean_gradient_norm.detach(),
            "robust_dmd_gradient_norm": robust_gradient_norm.detach(),
            "attack_chunk": torch.tensor(float(attack_chunk), device=self.device),
            "attack_epsilon": torch.tensor(float(epsilon), device=self.device),
            "robust_weight": torch.tensor(float(robust_weight), device=self.device),
            "attack_grad_rms": attack_grad_rms,
            "attack_delta_rms": attack_delta_rms,
            "exit_step": torch.tensor(float(state.exit_step_index + 1), device=self.device),
        }
        log_dict.update(regularization_log)
        return total_loss_value, log_dict

    def critic_loss_from_generated(
        self,
        generated_image: torch.Tensor,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Train fake score directly from a clean/robust replay sample."""
        generated_image = generated_image.detach().to(device=self.device, dtype=self.dtype)
        conditional_dict = {
            key: value.to(device=self.device, dtype=self.dtype)
            for key, value in conditional_dict.items()
        }
        image_or_video_shape = list(generated_image.shape)
        critic_timestep = self._get_timestep(
            self.min_score_timestep,
            self.num_train_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, image_or_video_shape[:2])
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
        )

        if self.args.denoising_loss_type == "flow":
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )
        return denoising_loss, {
            "critic_timestep": critic_timestep.detach(),
        }
