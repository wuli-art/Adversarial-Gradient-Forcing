import gc
import json
import logging
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path

from agf.utils.checkpoint import load_optimizer_bundle
from agf.utils.dataset import cycle, training_dataloader, validate_dataloader_resume
from agf.utils.distributed import (
    EMA_FSDP,
    clean_fsdp_state_dict,
    fsdp_optim_state_dict,
    fsdp_optim_state_dict_to_load,
    fsdp_wrap,
    fsdp_state_dict,
    launch_distributed_job,
)
from agf.utils.misc import (
    set_seed,
    merge_dict_list,
    resolve_gradient_accumulation_steps
)
import torch.distributed as dist
from omegaconf import OmegaConf
from agf.model import AdversarialHistoryDMD
import torch
import time
import os
import numpy as np


def validate_resume_checkpoint(checkpoint: dict, config) -> int:
    """Check model sections against the checkpoint step and EMA schedule."""
    configured_step = config.resume_step
    checkpoint_step = checkpoint.get("training_step")
    if configured_step is None and checkpoint_step is None:
        raise ValueError("resume_step is required for legacy checkpoints")
    step = int(configured_step if configured_step is not None else checkpoint_step)
    if checkpoint_step is not None and int(checkpoint_step) != step:
        raise ValueError(
            f"resume_step={step} disagrees with checkpoint training_step={checkpoint_step}"
        )
    if step <= 0:
        raise ValueError("resume_step must be positive")
    required = {"generator", "critic"}
    ema_active = config.ema_weight > 0 and step >= config.ema_start_step
    if ema_active:
        required.add("generator_ema")
    elif "generator_ema" in checkpoint:
        raise ValueError("Checkpoint EMA disagrees with the configured EMA schedule")
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise KeyError(f"Resume checkpoint is missing sections: {missing}")
    return step


class FakeScoreReplayBuffer:
    """Per-rank replay for the recent clean/robust generator batches."""

    def __init__(
        self,
        capacity_per_kind,
        clean_ratio,
        robust_ratio,
        seed,
    ):
        capacity_per_kind = int(capacity_per_kind)
        if capacity_per_kind <= 0:
            raise ValueError("Replay capacity must be positive")
        clean_ratio = float(clean_ratio)
        robust_ratio = float(robust_ratio)
        if clean_ratio < 0 or robust_ratio < 0 or \
                clean_ratio + robust_ratio <= 0:
            raise ValueError("Replay ratios must be non-negative and non-zero")

        self.clean = deque(maxlen=capacity_per_kind)
        self.robust = deque(maxlen=capacity_per_kind)
        self.clean_ratio = clean_ratio
        self.robust_ratio = robust_ratio
        self.random = random.Random(int(seed))

    def push(self, sample):
        kind = sample.get("kind")
        if kind == "clean":
            self.clean.append(sample)
        elif kind == "robust":
            self.robust.append(sample)
        else:
            raise ValueError(f"Unknown replay kind {kind!r}")

    def sample(self):
        if not self.clean and not self.robust:
            raise RuntimeError("Fake-score replay is empty")
        if not self.clean:
            source = self.robust
        elif not self.robust:
            source = self.clean
        else:
            robust_probability = self.robust_ratio / (
                self.clean_ratio + self.robust_ratio
            )
            source = (
                self.robust
                if self.random.random() < robust_probability
                else self.clean
            )
        return source[self.random.randrange(len(source))]

    def __len__(self):
        return len(self.clean) + len(self.robust)

    @staticmethod
    def _move_sample(sample, device):
        return {
            **sample,
            "generated_image": sample["generated_image"].detach().to(device),
            "conditional_dict": {
                key: value.detach().to(device)
                for key, value in sample["conditional_dict"].items()
            },
        }

    def state_dict(self):
        """Serialize replay tensors on CPU without changing queue order."""
        return {
            "capacity_per_kind": int(self.clean.maxlen),
            "clean_ratio": float(self.clean_ratio),
            "robust_ratio": float(self.robust_ratio),
            "clean": [
                self._move_sample(sample, torch.device("cpu"))
                for sample in self.clean
            ],
            "robust": [
                self._move_sample(sample, torch.device("cpu"))
                for sample in self.robust
            ],
            "random_state": self.random.getstate(),
        }

    def load_state_dict(self, state_dict, device):
        expected_capacity = int(self.clean.maxlen)
        saved_capacity = int(state_dict["capacity_per_kind"])
        if saved_capacity != expected_capacity:
            raise ValueError(
                "Replay capacity mismatch: checkpoint has "
                f"{saved_capacity}, config expects {expected_capacity}"
            )
        if float(state_dict["clean_ratio"]) != self.clean_ratio or \
                float(state_dict["robust_ratio"]) != self.robust_ratio:
            raise ValueError("Replay sampling ratios differ from checkpoint")
        self.clean.clear()
        self.robust.clear()
        self.clean.extend(
            self._move_sample(sample, device)
            for sample in state_dict["clean"]
        )
        self.robust.extend(
            self._move_sample(sample, device)
            for sample in state_dict["robust"]
        )
        self.random.setstate(state_dict["random_state"])


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.global_rank = global_rank
        self.world_size = dist.get_world_size()
        self.gradient_accumulation_steps, self.effective_total_batch_size = \
            resolve_gradient_accumulation_steps(
                batch_size=config.batch_size,
                world_size=self.world_size,
                total_batch_size=getattr(config, "total_batch_size", None),
                gradient_accumulation_steps=getattr(
                    config, "gradient_accumulation_steps", None
                )
            )

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        if not self.disable_wandb:
            import wandb
            self.wandb = wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            self.wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = AdversarialHistoryDMD(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        self.fake_replay = None
        if getattr(config, "fake_replay_enabled", False):
            if not getattr(self.model, "uses_fake_replay", False):
                raise ValueError(
                    "fake_replay_enabled requires a replay-aware model"
                )
            replay_batches = int(config.fake_replay_batches)
            replay_capacity = (
                replay_batches * self.gradient_accumulation_steps
            )
            self.fake_replay = FakeScoreReplayBuffer(
                capacity_per_kind=replay_capacity,
                clean_ratio=float(config.fake_replay_clean_ratio),
                robust_ratio=float(config.fake_replay_adv_ratio),
                seed=int(config.seed) + global_rank + 7919,
            )

        # Step 3: Initialize the dataloader
        dataloader_num_workers = int(getattr(
            config, "dataloader_num_workers", 8
        ))
        dataloader = training_dataloader(
            config.data_path,
            batch_size=config.batch_size,
            world_size=self.world_size,
            rank=global_rank,
            num_workers=dataloader_num_workers,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataloader.dataset))
            print(
                "GRADIENT ACCUMULATION %d | EFFECTIVE GLOBAL BATCH SIZE %d"
                % (
                    self.gradient_accumulation_steps,
                    self.effective_total_batch_size
                )
            )
            print("MAX TRAINING STEPS %s" % getattr(config, "max_steps", None))
            print("DATALOADER WORKERS %d" % dataloader_num_workers)
            if self.fake_replay is not None:
                print(
                    "FAKE REPLAY %d RECENT GENERATOR BATCHES | "
                    "CAPACITY %d PER KIND / RANK"
                    % (int(config.fake_replay_batches), replay_capacity)
                )
        self.dataloader = cycle(dataloader)
        self.dataloader_batches_consumed = 0

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.ema_rank0_only = bool(getattr(
            config, "ema_rank0_only", False
        ))
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_main_process:
                print(
                    f"Setting up EMA with weight {ema_weight} | "
                    f"rank0_only={self.ema_rank0_only}"
                )
            self.generator_ema = EMA_FSDP(
                self.model.generator,
                decay=ema_weight,
                rank0_only=self.ema_rank0_only,
            )

        ##############################################################################################################
        # 7. Load either a complete training checkpoint or generator init.
        self.initialization_metadata = {
            "mode": "model_default",
            "fake_score_initialization": (
                f"pretrained:{getattr(config, 'fake_name', 'Wan2.1-T2V-1.3B')}"
            ),
        }
        resume_checkpoint = getattr(config, "resume_checkpoint", None)
        if resume_checkpoint:
            if self.is_main_process:
                print(f"Resuming training weights from {resume_checkpoint}")
            state_dict = torch.load(
                resume_checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            self.step = validate_resume_checkpoint(state_dict, config)
            if self.step < config.ema_start_step:
                self.generator_ema = None
            reset_optimizers = bool(getattr(
                config, "resume_reset_optimizers", False
            ))
            reset_replay = bool(getattr(
                config, "resume_reset_replay", False
            ))
            optimizer_state_saved = bool(
                state_dict.get("optimizer_state_saved", False)
            )
            replay_state_saved = bool(
                state_dict.get("replay_state_saved", False)
            )
            if not reset_optimizers and not optimizer_state_saved:
                raise ValueError(
                    "Checkpoint does not contain optimizer state; set "
                    "resume_reset_optimizers=true to acknowledge the reset"
                )
            if not reset_replay and not replay_state_saved:
                raise ValueError(
                    "Checkpoint does not contain replay/RNG state; set "
                    "resume_reset_replay=true to acknowledge the reset"
                )
            self.model.generator.load_state_dict(
                clean_fsdp_state_dict(state_dict["generator"]), strict=True
            )
            self.model.fake_score.load_state_dict(
                clean_fsdp_state_dict(state_dict["critic"]), strict=True
            )
            if self.generator_ema is None and "generator_ema" in state_dict:
                raise ValueError(
                    "Resume checkpoint has generator_ema but EMA is disabled"
                )
            if self.generator_ema is not None:
                self.generator_ema.load_state_dict(
                    clean_fsdp_state_dict(state_dict["generator_ema"])
                )

            rank_training_state = None
            if not reset_replay:
                rank_state_path = (
                    Path(resume_checkpoint).parent
                    / "trainer_state"
                    / f"rank_{self.global_rank:05d}.pt"
                )
                if not rank_state_path.is_file():
                    raise FileNotFoundError(
                        "Missing per-rank trainer state: "
                        f"{rank_state_path}"
                    )
                rank_training_state = torch.load(
                    rank_state_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if int(rank_training_state["world_size"]) != self.world_size:
                    raise ValueError("Exact resume requires the checkpoint world size")
                if int(rank_training_state["global_rank"]) != self.global_rank:
                    raise ValueError("Resume state belongs to a different rank")
                if int(rank_training_state["training_step"]) != self.step:
                    raise ValueError(
                        "Per-rank trainer state step disagrees with model "
                        f"checkpoint: {rank_training_state['training_step']} "
                        f"vs {self.step}"
                    )

            configured_dataloader_batches = getattr(
                config, "resume_dataloader_batches", None
            )
            if rank_training_state is not None:
                validate_dataloader_resume(rank_training_state, dataloader)
                dataloader_batches = int(
                    rank_training_state["dataloader_batches_consumed"]
                )
                if configured_dataloader_batches is not None and int(
                    configured_dataloader_batches
                ) != dataloader_batches:
                    raise ValueError(
                        "resume_dataloader_batches disagrees with saved "
                        "trainer state"
                    )
            else:
                dataloader_batches = int(
                    configured_dataloader_batches or 0
                )
            if dataloader_batches < 0:
                raise ValueError("resume_dataloader_batches must be non-negative")
            for _ in range(dataloader_batches):
                next(self.dataloader)
            self.dataloader_batches_consumed = dataloader_batches

            if not reset_optimizers:
                optimizer_bundle = load_optimizer_bundle(state_dict, resume_checkpoint) if self.is_main_process else None
                generator_optim_state = fsdp_optim_state_dict_to_load(
                    self.model.generator,
                    self.generator_optimizer,
                    optimizer_bundle["generator_optimizer"] if self.is_main_process else None,
                )
                self.generator_optimizer.load_state_dict(
                    generator_optim_state
                )
                critic_optim_state = fsdp_optim_state_dict_to_load(
                    self.model.fake_score,
                    self.critic_optimizer,
                    optimizer_bundle["critic_optimizer"] if self.is_main_process else None,
                )
                self.critic_optimizer.load_state_dict(critic_optim_state)
                del optimizer_bundle, generator_optim_state, critic_optim_state

            if rank_training_state is not None:
                if self.fake_replay is None:
                    raise ValueError(
                        "Saved replay state requires fake_replay_enabled=true"
                    )
                self.fake_replay.load_state_dict(
                    rank_training_state["fake_replay"],
                    device=self.model.fake_replay_device,
                )
                random.setstate(rank_training_state["python_rng_state"])
                np.random.set_state(rank_training_state["numpy_rng_state"])
                torch.set_rng_state(rank_training_state["torch_rng_state"])
                torch.cuda.set_rng_state(
                    rank_training_state["cuda_rng_state"],
                    device=self.device,
                )

            if self.is_main_process:
                print(
                    "RESUME TRAINING STEP %d | RESET OPTIMIZERS %s | "
                    "RESET REPLAY %s | SKIPPED DATALOADER BATCHES %d"
                    % (
                        self.step,
                        "yes" if reset_optimizers else "no",
                        "yes" if reset_replay else "no",
                        dataloader_batches,
                    )
                )
            self.initialization_metadata = {
                "mode": "resume_full_training_weights",
                "checkpoint": str(resume_checkpoint),
                "training_step": int(self.step),
                "generator_section": "generator",
                "fake_score_section": "critic",
                "ema_section": "generator_ema" if self.generator_ema is not None else None,
                "optimizers_reset": reset_optimizers,
                "replay_reset": reset_replay,
                "rng_restored": rank_training_state is not None,
                "dataloader_batches_consumed": dataloader_batches,
            }
            del state_dict, rank_training_state
            gc.collect()
        elif getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            checkpoint = torch.load(
                config.generator_ckpt,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            requested_key = getattr(config, "generator_ckpt_key", None)
            if requested_key is not None:
                requested_key = str(requested_key)
                if requested_key not in checkpoint:
                    raise KeyError(
                        f"generator_ckpt_key={requested_key!r} is absent "
                        f"from {config.generator_ckpt}; available top-level "
                        f"keys: {sorted(checkpoint.keys())}"
                    )
                selected_key = requested_key
                state_dict = checkpoint[selected_key]
            elif "generator" in checkpoint:
                selected_key = "generator"
                state_dict = checkpoint[selected_key]
            elif "model" in checkpoint:
                selected_key = "model"
                state_dict = checkpoint[selected_key]
            elif "generator_ema" in checkpoint:
                selected_key = "generator_ema"
                state_dict = checkpoint[selected_key]
            else:
                selected_key = "<root_state_dict>"
                state_dict = checkpoint
            state_dict = clean_fsdp_state_dict(state_dict)
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )
            initialize_ema = bool(getattr(
                config,
                "initialize_ema_from_generator_ckpt",
                False,
            ))
            if initialize_ema:
                if self.generator_ema is None:
                    raise ValueError(
                        "initialize_ema_from_generator_ckpt=true requires "
                        "ema_weight > 0"
                    )
                self.generator_ema.load_state_dict(state_dict)
            self.initialization_metadata = {
                "mode": "generator_checkpoint",
                "checkpoint": str(config.generator_ckpt),
                "generator_section": selected_key,
                "generator_strict_load": True,
                "ema_initialized_from_generator_checkpoint": initialize_ema,
                "fake_score_initialization": (
                    "pretrained:%s" % getattr(
                        config, "fake_name", "Wan2.1-T2V-1.3B"
                    )
                ),
            }
            print(
                "GENERATOR INIT SECTION %s | EMA FROM GENERATOR %s | "
                "FAKE SCORE INIT pretrained:%s"
                % (
                    selected_key,
                    "yes" if initialize_ema else "no",
                    getattr(config, "fake_name", "Wan2.1-T2V-1.3B"),
                )
            )
            del state_dict, checkpoint
            gc.collect()

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.empty_cache_interval = int(getattr(
            config, "empty_cache_interval", 20
        ))
        if self.empty_cache_interval < 0:
            raise ValueError("empty_cache_interval must be non-negative")
        self.infra_log_timing = bool(getattr(
            config, "infra_log_timing", False
        ))

    def save(self):
        if self.is_main_process:
            print("Start gathering distributed model states...", flush=True)
        training_state_steps = {
            int(step)
            for step in getattr(
                self.config, "save_training_state_steps", []
            )
        }
        save_training_state = self.step in training_state_steps
        if save_training_state and self.fake_replay is None:
            raise ValueError(
                "Saving exact trainer state requires fake replay to be enabled"
            )
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)
        if save_training_state:
            generator_optimizer_state = fsdp_optim_state_dict(
                self.model.generator, self.generator_optimizer
            )
            critic_optimizer_state = fsdp_optim_state_dict(
                self.model.fake_score, self.critic_optimizer
            )
        else:
            generator_optimizer_state = None
            critic_optimizer_state = None

        if self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "training_step": int(self.step),
                "optimizer_state_saved": save_training_state,
                "replay_state_saved": save_training_state,
                "checkpoint_format_version": 3,
                "initialization": self.initialization_metadata,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "training_step": int(self.step),
                "optimizer_state_saved": save_training_state,
                "replay_state_saved": save_training_state,
                "checkpoint_format_version": 3,
                "initialization": self.initialization_metadata,
            }
        if save_training_state:
            state_dict.update({
                "optimizer_state_path": "trainer_state/optimizers.pt",
                "rank_training_state_pattern": (
                    "trainer_state/rank_{rank:05d}.pt"
                ),
            })

        checkpoint_dir = Path(
            self.output_path, f"checkpoint_model_{self.step:06d}"
        )
        if self.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            if save_training_state:
                (checkpoint_dir / "trainer_state").mkdir(
                    parents=True, exist_ok=True
                )
        dist.barrier()

        if save_training_state:
            rank_state_path = (
                checkpoint_dir
                / "trainer_state"
                / f"rank_{self.global_rank:05d}.pt"
            )
            rank_state_temporary = rank_state_path.with_suffix(
                rank_state_path.suffix + ".partial"
            )
            rank_state_dict = {
                "checkpoint_format_version": 3,
                "training_step": int(self.step),
                "global_rank": int(self.global_rank),
                "world_size": int(self.world_size),
                "dataloader_batches_consumed": int(
                    self.dataloader_batches_consumed
                ),
                "dataloader_drop_last": True,
                "fake_replay": self.fake_replay.state_dict(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(self.device),
            }
            torch.save(rank_state_dict, rank_state_temporary)
            os.replace(rank_state_temporary, rank_state_path)
        dist.barrier()

        if self.is_main_process:
            if save_training_state:
                optimizer_path = checkpoint_dir / "trainer_state" / "optimizers.pt"
                optimizer_temporary = optimizer_path.with_suffix(".pt.partial")
                torch.save({
                    "checkpoint_format_version": 3,
                    "training_step": int(self.step),
                    "generator_optimizer": generator_optimizer_state,
                    "critic_optimizer": critic_optimizer_state,
                }, optimizer_temporary)
                os.replace(optimizer_temporary, optimizer_path)
            checkpoint_path = checkpoint_dir / "model.pt"
            temporary_path = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".partial"
            )
            torch.save(state_dict, temporary_path)
            os.replace(temporary_path, checkpoint_path)
            print(
                "Model saved to %s | exact_trainer_state=%s"
                % (checkpoint_path, save_training_state),
                flush=True,
            )
        dist.barrier()

    def _next_batch(self):
        batch = next(self.dataloader)
        self.dataloader_batches_consumed += 1
        return batch

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if not train_generator and self.fake_replay is not None:
            replay_sample = self.fake_replay.sample()
            critic_loss, critic_log_dict = (
                self.model.critic_loss_from_generated(
                    generated_image=replay_sample["generated_image"],
                    conditional_dict=replay_sample["conditional_dict"],
                )
            )
            (
                critic_loss / self.gradient_accumulation_steps
            ).backward()
            critic_log_dict.update({
                "critic_loss": critic_loss.detach(),
                "critic_replay_robust": torch.tensor(
                    float(replay_sample["kind"] == "robust"),
                    device=self.device,
                ),
            })
            return critic_log_dict

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if (
                not getattr(self, "unconditional_dict", None)
                or getattr(self, "unconditional_batch_size", None) != batch_size
            ):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                self.unconditional_batch_size = batch_size
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_kwargs = dict(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=(
                    image_latent if self.config.i2v else None
                ),
            )
            if getattr(self.model, "handles_generator_backward", False):
                generator_kwargs.update({
                    "training_step": self.step,
                    "loss_scale": 1.0 / self.gradient_accumulation_steps,
                })
            generator_loss, generator_log_dict = (
                self.model.generator_loss(**generator_kwargs)
            )

            if not getattr(
                self.model, "handles_generator_backward", False
            ):
                (
                    generator_loss / self.gradient_accumulation_steps
                ).backward()

            if self.fake_replay is not None:
                replay_samples = self.model.pop_replay_samples()
                if not replay_samples:
                    raise RuntimeError(
                        "Replay-aware generator produced no replay samples"
                    )
                for replay_sample in replay_samples:
                    self.fake_replay.push(replay_sample)

            generator_log_dict.update({
                "generator_loss": generator_loss.detach()
            })

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        (
            critic_loss / self.gradient_accumulation_steps
        ).backward()

        critic_log_dict.update({
            "critic_loss": critic_loss.detach()
        })

        return critic_log_dict

    def _gradient_sync_context(self, model, micro_step):
        """Skip FSDP gradient synchronization before the final microbatch."""
        is_last_microbatch = (
            micro_step == self.gradient_accumulation_steps - 1
        )
        return nullcontext() if is_last_microbatch else model.no_sync()


    def train(self):
        start_step = self.step
        max_steps = getattr(self.config, "max_steps", None)
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps <= self.step:
                raise ValueError(
                    f"max_steps={max_steps} must be greater than "
                    f"the current step {self.step}"
                )

        while max_steps is None or self.step < max_steps:
            iteration_started = time.perf_counter()
            if self.infra_log_timing:
                torch.cuda.reset_peak_memory_stats(self.device)
            if self.empty_cache_interval > 0 and \
                    self.step % self.empty_cache_interval == 0:
                torch.cuda.empty_cache()

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            generator_wall_time = 0.0

            # Train the generator
            if TRAIN_GENERATOR:
                generator_started = time.perf_counter()
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for micro_step in range(self.gradient_accumulation_steps):
                    batch = self._next_batch()
                    with self._gradient_sync_context(
                        self.model.generator, micro_step
                    ):
                        extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                generator_grad_norm = self.model.generator.clip_grad_norm_(
                    self.max_grad_norm_generator
                )
                generator_log_dict["generator_grad_norm"] = \
                    generator_grad_norm.detach()
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                generator_wall_time = (
                    time.perf_counter() - generator_started
                )

            # Train the critic
            critic_started = time.perf_counter()
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for micro_step in range(self.gradient_accumulation_steps):
                batch = (
                    None
                    if self.fake_replay is not None
                    else self._next_batch()
                )
                with self._gradient_sync_context(
                    self.model.fake_score, micro_step
                ):
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic
            )
            critic_log_dict["critic_grad_norm"] = critic_grad_norm.detach()
            self.critic_optimizer.step()
            critic_wall_time = time.perf_counter() - critic_started

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    rank0_only=self.ema_rank0_only,
                )

            # Save the model
            reached_max_steps = (
                max_steps is not None and self.step >= max_steps
            )
            checkpoint_steps = {
                int(step)
                for step in getattr(
                    self.config, "checkpoint_steps", []
                )
            }
            should_save = (
                (self.step - start_step) > 0
                and (
                    self.step in checkpoint_steps
                    or (
                        not checkpoint_steps
                        and self.step % self.config.log_iters == 0
                    )
                    or reached_max_steps
                )
            )
            checkpoint_wall_time = 0.0
            if (not self.config.no_save) and should_save:
                checkpoint_started = time.perf_counter()
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                checkpoint_wall_time = (
                    time.perf_counter() - checkpoint_started
                )

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                    for key in (
                        "clean_dmd_loss",
                        "robust_dmd_loss",
                        "clean_dmd_gradient_norm",
                        "robust_dmd_gradient_norm",
                        "attack_chunk",
                        "attack_epsilon",
                        "robust_weight",
                        "attack_grad_rms",
                        "attack_delta_rms",
                        "exit_step",
                        "regularization_loss",
                        "regularization_raw_loss",
                        "regularization_scale",
                        "regularization_base_rms",
                        "regularization_raw_rms",
                        "regularization_target_ratio",
                        "regularization_achieved_ratio",
                    ):
                        if key in generator_log_dict:
                            wandb_loss_dict[key] = (
                                generator_log_dict[key].float().mean().item()
                            )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )
                if "critic_replay_robust" in critic_log_dict:
                    wandb_loss_dict.update({
                        "critic_replay_robust_fraction": (
                            critic_log_dict[
                                "critic_replay_robust"
                            ].float().mean().item()
                        ),
                        "fake_replay_size": float(len(self.fake_replay)),
                    })

                if self.infra_log_timing:
                    wandb_loss_dict.update({
                        "infra_iteration_time_sec": (
                            time.perf_counter() - iteration_started
                        ),
                        "infra_generator_time_sec": generator_wall_time,
                        "infra_critic_time_sec": critic_wall_time,
                        "infra_checkpoint_time_sec": checkpoint_wall_time,
                        "infra_peak_memory_allocated_gib": (
                            torch.cuda.max_memory_allocated(self.device)
                            / (1024 ** 3)
                        ),
                        "infra_peak_memory_reserved_gib": (
                            torch.cuda.max_memory_reserved(self.device)
                            / (1024 ** 3)
                        ),
                        "infra_dataloader_batches_consumed": float(
                            self.dataloader_batches_consumed
                        ),
                    })

                with (Path(self.output_path) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": self.step, **wandb_loss_dict}) + "\n")

                if not self.disable_wandb:
                    self.wandb.log(wandb_loss_dict, step=self.step)
                else:
                    print(f"[step {self.step}] " + " ".join(
                        f"{k}={v:.4f}" for k, v in wandb_loss_dict.items()), flush=True)

            gc_interval = int(self.config.gc_interval)
            if gc_interval > 0 and self.step % gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

        dist.barrier()
        if self.is_main_process:
            print(f"Training completed at step {self.step}", flush=True)
        dist.destroy_process_group()
