"""Portable format-v2/v3 checkpoint inspection and optimizer loading."""

from pathlib import Path


def load_optimizer_bundle(model, model_path, *, map_location="cpu"):
    import torch

    version = int(model.get("checkpoint_format_version", 0))
    if version == 2:
        bundle = model
    elif version == 3:
        root = Path(model_path).resolve().parent
        path = (root / model.get("optimizer_state_path", "trainer_state/optimizers.pt")).resolve()
        if root not in path.parents:
            raise ValueError("Optimizer state must be inside the checkpoint directory")
        bundle = torch.load(path, map_location=map_location, weights_only=False, mmap=True)
        if int(bundle.get("training_step", -1)) != int(model["training_step"]):
            raise ValueError("Optimizer step disagrees with model")
    else:
        raise ValueError(f"Unsupported resumable checkpoint format: {version}")
    if not {"generator_optimizer", "critic_optimizer"} <= bundle.keys():
        raise ValueError("Checkpoint is missing optimizer state")
    return bundle


def verify_checkpoint(path, *, step, world_size, ema_required=True):
    """Check all rank files and tensor structure without materializing tensor data."""
    import torch

    path = Path(path)
    model = torch.load(path, map_location="meta", weights_only=False, mmap=True)
    required = {"generator", "critic"} | ({"generator_ema"} if ema_required else set())
    if not required <= model.keys() or any(not model[k] for k in required):
        raise ValueError("Checkpoint is missing model/EMA sections")
    if model.get("training_step") != step:
        raise ValueError("Model step disagrees with requested resume step")
    if not model.get("optimizer_state_saved") or not model.get("replay_state_saved"):
        raise ValueError("Full optimizer and replay state are required")
    load_optimizer_bundle(model, path, map_location="meta")
    for rank in range(world_size):
        rank_path = path.parent / "trainer_state" / f"rank_{rank:05d}.pt"
        state = torch.load(rank_path, map_location="meta", weights_only=False, mmap=True)
        if (state.get("training_step"), state.get("world_size"), state.get("global_rank")) != (
            step,
            world_size,
            rank,
        ):
            raise ValueError(f"Rank {rank} checkpoint step, world size or rank mismatch")
        required_state = {
            "fake_replay",
            "python_rng_state",
            "numpy_rng_state",
            "torch_rng_state",
            "cuda_rng_state",
            "dataloader_batches_consumed",
        }
        if not required_state <= state.keys() or state["dataloader_batches_consumed"] < 0:
            raise ValueError(f"Rank {rank} is missing replay/RNG/dataloader state")
        replay = state["fake_replay"]
        if (
            not {
                "clean",
                "robust",
                "random_state",
                "capacity_per_kind",
                "clean_ratio",
                "robust_ratio",
            }
            <= replay.keys()
        ):
            raise ValueError(f"Rank {rank} replay is incomplete")
    return {"format": model["checkpoint_format_version"], "step": step, "world_size": world_size}
