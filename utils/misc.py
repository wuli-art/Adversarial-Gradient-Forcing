import numpy as np
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def resolve_gradient_accumulation_steps(
    batch_size,
    world_size,
    total_batch_size=None,
    gradient_accumulation_steps=None
):
    """
    Resolve the number of microbatches per optimizer update.

    ``total_batch_size`` is the target effective global batch size. An explicit
    ``gradient_accumulation_steps`` can be supplied instead; when both are
    provided, they must describe the same effective batch size.
    """
    batch_size = int(batch_size)
    world_size = int(world_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    micro_global_batch_size = batch_size * world_size

    if gradient_accumulation_steps is not None:
        gradient_accumulation_steps = int(gradient_accumulation_steps)
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, "
                f"got {gradient_accumulation_steps}"
            )

        effective_batch_size = (
            micro_global_batch_size * gradient_accumulation_steps
        )
        if total_batch_size is not None and int(total_batch_size) != effective_batch_size:
            raise ValueError(
                f"total_batch_size={int(total_batch_size)} conflicts with "
                f"batch_size={batch_size}, world_size={world_size}, and "
                "gradient_accumulation_steps="
                f"{gradient_accumulation_steps} (effective batch size "
                f"{effective_batch_size})"
            )
        return gradient_accumulation_steps, effective_batch_size

    if total_batch_size is None:
        return 1, micro_global_batch_size

    total_batch_size = int(total_batch_size)
    if total_batch_size <= 0:
        raise ValueError(
            f"total_batch_size must be positive, got {total_batch_size}"
        )
    if total_batch_size % micro_global_batch_size != 0:
        raise ValueError(
            f"total_batch_size={total_batch_size} must be divisible by the "
            f"micro global batch size ({batch_size} per GPU x {world_size} "
            f"GPUs = {micro_global_batch_size})"
        )

    gradient_accumulation_steps = (
        total_batch_size // micro_global_batch_size
    )
    return gradient_accumulation_steps, total_batch_size
