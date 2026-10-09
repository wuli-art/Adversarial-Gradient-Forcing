"""Inspect external inputs without loading tensors or allocating GPU memory."""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig


def model_root(value: str | Path | None = None) -> Path:
    return Path(value or os.environ.get("AGF_WAN_MODELS_DIR", "wan_models")).expanduser().resolve()


def check_model(root: Path, name: str) -> list[str]:
    directory = root / name
    required = ["config.json"]
    if name == "Wan2.1-T2V-1.3B":
        required += [
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/tokenizer_config.json",
        ]
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                return [f"Empty weight index: {index}"]
            required += sorted(shards)
        except (ValueError, KeyError, TypeError):
            return [f"Invalid weight index: {index}"]
    else:
        required += ["diffusion_pytorch_model.safetensors"]
    errors = [
        f"Missing or empty model file: {directory / file}"
        for file in required
        if not (directory / file).is_file() or (directory / file).stat().st_size == 0
    ]
    return errors


def training_asset_errors(config: DictConfig, root: Path, world_size: int) -> list[str]:
    errors = check_model(root, config.real_name) + check_model(root, config.fake_name)
    prompts = Path(config.data_path)
    if not prompts.is_file():
        errors.append(f"Missing training prompts: {prompts}")
    else:
        lines = prompts.read_text(encoding="utf-8").splitlines()
        if len(lines) < world_size * config.batch_size or any(not line.strip() for line in lines):
            errors.append("Training prompts must be nonempty and cover one distributed batch")
    checkpoint = Path(config.resume_checkpoint or config.generator_ckpt)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        errors.append(f"Missing or empty checkpoint: {checkpoint}")
    if config.resume_checkpoint and not config.resume_reset_replay:
        for rank in range(world_size):
            state = checkpoint.parent / "trainer_state" / f"rank_{rank:05d}.pt"
            if not state.is_file():
                errors.append(f"Missing resume state: {state}")
    return errors
