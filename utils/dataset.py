"""Prompt-only training data; one UTF-8 prompt per line."""

from pathlib import Path

from torch.utils.data import DataLoader, Dataset, DistributedSampler


class TextDataset(Dataset):
    def __init__(self, prompt_path: str | Path):
        self.prompt_list = Path(prompt_path).read_text(encoding="utf-8").splitlines()
        if not self.prompt_list or any(not line.strip() for line in self.prompt_list):
            raise ValueError("The prompt file must contain nonempty lines (one prompt per line)")

    def __len__(self) -> int:
        return len(self.prompt_list)

    def __getitem__(self, idx: int) -> dict:
        return {"prompts": self.prompt_list[idx], "idx": idx}


def training_dataloader(prompt_path, *, batch_size, world_size, rank, num_workers):
    """Keep each rank's microbatch full so accumulation preserves the global batch."""
    dataset = TextDataset(prompt_path)
    if batch_size <= 0 or len(dataset) < world_size * batch_size:
        raise ValueError("Training prompts must cover at least one distributed batch")
    if num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, drop_last=True,
    )


def validate_dataloader_resume(state, dataloader):
    """Legacy partial batches cannot be counted as full batches after this fix."""
    if (not state.get("dataloader_drop_last", False)
            and len(dataloader.sampler) % dataloader.batch_size):
        raise ValueError(
            "This legacy checkpoint used partial training batches; its consumed-batch "
            "position cannot be resumed with full batches. Continue it with the original "
            "code, or start a new training run."
        )


def cycle(dataloader):
    # Preserve the published sampler order and resume semantics.
    while True:
        yield from dataloader
