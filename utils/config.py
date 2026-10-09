"""Load and validate the complete configuration before allocating model memory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = PACKAGE_ROOT / "configs/train.yaml"
INFERENCE_CONFIG = PACKAGE_ROOT / "configs/inference.yaml"
EXAMPLE_PROMPTS = PACKAGE_ROOT / "prompts/examples.txt"


def load_config(
    path: str | Path | None = None,
    overrides: Sequence[str] = (),
    *,
    mode: str = "train",
    world_size: int | None = None,
) -> DictConfig:
    """Merge defaults, YAML and dot-list overrides; reject unknown fields."""
    if mode not in {"train", "inference"}:
        raise ValueError(f"Unknown configuration mode: {mode}")
    try:
        defaults = OmegaConf.load(TRAIN_CONFIG if mode == "train" else INFERENCE_CONFIG)
        OmegaConf.set_struct(defaults, True)
        supplied = OmegaConf.load(path) if path else OmegaConf.create({})
        if not OmegaConf.is_dict(supplied):
            raise ValueError("The configuration must be a YAML mapping")
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Expected KEY=VALUE, got {override!r}")
        config = OmegaConf.merge(defaults, supplied, OmegaConf.from_dotlist(list(overrides)))
        OmegaConf.resolve(config)
        _validate_types(config, defaults)
        validate_config(config, mode=mode, world_size=world_size)
        return config
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc


def _validate_types(config: DictConfig, defaults: DictConfig, prefix: str = "") -> None:
    optional_types = {
        "resume_checkpoint": str,
        "resume_step": int,
        "resume_dataloader_batches": int,
        "wandb_entity": str,
    }
    for key, expected in defaults.items():
        value = config[key]
        name = prefix + key
        if OmegaConf.is_dict(expected):
            if not OmegaConf.is_dict(value):
                raise ValueError(f"{name} must be a mapping")
            _validate_types(value, expected, name + ".")
        elif OmegaConf.is_list(expected):
            if not OmegaConf.is_list(value):
                raise ValueError(f"{name} must be a list")
            if expected and any(type(item) is not type(expected[0]) for item in value):
                raise ValueError(f"{name} has an invalid list element type")
        elif expected is None and value is not None and name in optional_types:
            if type(value) is not optional_types[name]:
                raise ValueError(f"{name} must have type {optional_types[name].__name__}")
        elif expected is not None:
            if isinstance(expected, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value) is type(expected)
            if not valid:
                raise ValueError(f"{name} must have type {type(expected).__name__}")


def _integer(config: DictConfig, name: str, minimum: int = 1) -> int:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(config: DictConfig, name: str, minimum: float = 0.0) -> float:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def validate_config(
    config: DictConfig, *, mode: str = "train", world_size: int | None = None
) -> None:
    """Validate supported geometry and algorithm invariants, not one experiment's tuning."""
    fixed = {
        "num_frame_per_block": 3,
        "independent_first_frame": False,
        "context_noise": 0,
        "num_train_timestep": 1000,
        "warp_denoising_step": True,
        "same_step_across_blocks": True,
        "mixed_precision": True,
    }
    if mode == "train":
        fixed.update(
            {
                "distribution_loss": "adversarial_history_dmd",
                "trainer": "score_distillation",
                "grad_mode": "adversarial_self_gradient_forcing",
                "attack_future_only": True,
                "attack_temporal_shared": True,
                "attack_steps": 1,
                "attack_random_start": False,
                "fake_replay_enabled": True,
                "i2v": False,
                "load_raw_video": False,
                "causal": True,
                "num_training_frames": 21,
                "denoising_loss_type": "flow",
                "self_gradient_forcing_match_context": False,
                "self_gradient_forcing_cache_mode": "exit",
                "fake_name": "Wan2.1-T2V-1.3B",
                "real_name": "Wan2.1-T2V-14B",
            }
        )
    for key, expected in fixed.items():
        if config[key] != expected:
            raise ValueError(f"{key} must be {expected!r} for AGF")
    if config.model_kwargs.model_name != "Wan2.1-T2V-1.3B":
        raise ValueError("Only the Wan2.1-T2V-1.3B causal generator is supported")
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError("denoising_step_list must be [1000, 750, 500, 250]")
    _integer(config, "seed", 0)
    if mode == "inference":
        geometry = (config.model_kwargs.local_attn_size, config.model_kwargs.sink_size,
                    config.model_kwargs.relative_sink)
        if geometry not in {(12, 3, True), (21, 0, False)}:
            raise ValueError("Use paper local12/sink3/relative or the named local21/no-sink ablation")
        return
    for key in (
        "batch_size",
        "total_batch_size",
        "max_steps",
        "dfake_gen_update_ratio",
        "fake_replay_batches",
        "log_iters",
    ):
        _integer(config, key)
    for key in ("ema_start_step", "dataloader_num_workers", "gc_interval", "empty_cache_interval"):
        _integer(config, key, 0)
    for key in (
        "attack_epsilon_rms",
        "clean_loss_weight",
        "adv_loss_weight",
        "weight_decay",
        "lr",
        "lr_critic",
        "max_grad_norm_generator",
        "max_grad_norm_critic",
    ):
        _number(config, key)
    for keys in (
        ("clean_loss_weight", "adv_loss_weight"),
        ("fake_replay_clean_ratio", "fake_replay_adv_ratio"),
    ):
        if not math.isclose(sum(_number(config, k) for k in keys), 1.0):
            raise ValueError(f"{keys[0]} + {keys[1]} must equal 1")
    _integer(config, "regularization_end_step")
    _integer(config, "regularization_ramp_start", 0)
    _integer(config, "regularization_ramp_end")
    if not config.regularization_ramp_start < config.regularization_ramp_end <= config.regularization_end_step:
        raise ValueError("Regularization requires ramp_start < ramp_end <= end_step")
    _number(config, "regularization_target_ratio")
    for key in ("regularization_gradient_eps", "regularization_max_scale"):
        if _number(config, key) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("beta1", "beta2", "beta1_critic", "beta2_critic", "ema_weight"):
        if not 0 <= _number(config, key) < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if not config.attack_chunk_choices or any(
        type(c) is not int or c not in (1, 2, 3) for c in config.attack_chunk_choices
    ):
        raise ValueError("attack_chunk_choices must be a nonempty subset of [1, 2, 3]")
    if list(config.image_or_video_shape) != [config.batch_size, 21, 16, 60, 104]:
        raise ValueError("image_or_video_shape must be [batch_size, 21, 16, 60, 104] (480x832)")
    if config.fake_replay_device not in ("cpu", "cuda", "gpu"):
        raise ValueError("fake_replay_device must be cpu or cuda")
    if config.fake_replay_dtype not in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"):
        raise ValueError("Unsupported fake_replay_dtype")
    if config.sharding_strategy not in ("full", "hybrid_full", "hybrid_zero2", "no_shard"):
        raise ValueError("Unsupported sharding_strategy")
    for key in (
        "generator_fsdp_wrap_strategy",
        "real_score_fsdp_wrap_strategy",
        "fake_score_fsdp_wrap_strategy",
        "text_encoder_fsdp_wrap_strategy",
    ):
        if config[key] != "size":
            raise ValueError(f"{key} must be size; transformer wrapping is not configured")
    for key in ("checkpoint_steps", "save_training_state_steps"):
        if any(type(step) is not int or step <= 0 for step in config[key]):
            raise ValueError(f"{key} must contain positive integer steps")
    if config.resume_checkpoint:
        if not isinstance(config.resume_checkpoint, str):
            raise ValueError("resume_checkpoint must be a path string")
        if type(config.resume_step) is not int or not 0 < config.resume_step < config.max_steps:
            raise ValueError("Resume requires 0 < resume_step < max_steps")
    elif config.resume_step is not None:
        raise ValueError("resume_step requires resume_checkpoint")
    if not config.resume_checkpoint and not config.generator_ckpt:
        raise ValueError("generator_ckpt or resume_checkpoint is required")
    if config.resume_dataloader_batches is not None:
        _integer(config, "resume_dataloader_batches", 0)
    if world_size is not None:
        if world_size <= 0 or config.total_batch_size % (config.batch_size * world_size):
            raise ValueError("total_batch_size must be divisible by batch_size * world_size")


def resolve_input_paths(config: DictConfig, base: Path | None = None) -> DictConfig:
    """Input paths are relative to the invocation directory, including paths in YAML."""
    base = base or Path.cwd()
    for key in ("data_path", "generator_ckpt", "resume_checkpoint"):
        if key in config and config[key]:
            path = Path(config[key]).expanduser()
            config[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    return config
