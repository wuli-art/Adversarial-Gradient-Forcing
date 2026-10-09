from itertools import islice
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from agf import train as worker
from agf.trainer.distillation import Trainer
from agf.utils import cli
from agf.utils.config import PACKAGE_ROOT
from agf.utils.dataset import cycle, training_dataloader, validate_dataloader_resume


@pytest.mark.parametrize("absolute", [False, True])
def test_training_rejects_existing_output_without_touching_any_files(tmp_path, monkeypatch, absolute):
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("AGF_RUN_ID", "existing")
    destination = tmp_path / "external" if absolute else tmp_path / "runs/AGF/existing/logs"
    destination.mkdir(parents=True)
    config = destination / "resolved_config.yaml"
    config.write_text("original configuration")
    old_checkpoint = destination / "checkpoint_model_000200/model.pt"
    old_checkpoint.parent.mkdir()
    old_checkpoint.write_bytes(b"original weights")
    monkeypatch.setattr(cli, "training_asset_errors", lambda *a: pytest.fail("output preflight is late"))
    flags = ["--logdir", str(destination)] if absolute else []
    assert cli.main(["train", *flags]) == 2
    assert config.read_text() == "original configuration"
    assert old_checkpoint.read_bytes() == b"original weights"
    assert not (tmp_path / "runs/AGF/existing/config").exists()


def test_direct_worker_refuses_existing_output_even_for_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "training_asset_errors", lambda *a: [])
    monkeypatch.setenv("RANK", "0")
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "resolved_config.yaml"
    sentinel.write_text("keep")
    with pytest.raises(SystemExit) as error:
        worker.main(["--logdir", str(destination), "--set", "resume_checkpoint=/old/model.pt",
                     "--set", "resume_step=200"])
    assert error.value.code == 2
    assert sentinel.read_text() == "keep"


def test_two_stages_share_run_id_without_overwriting_configs(tmp_path, monkeypatch):
    import agf.trainer.distillation as module

    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("AGF_RUN_ID", "paper")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(cli, "training_asset_errors", lambda *a: [])
    monkeypatch.setattr(worker, "training_asset_errors", lambda *a: [])
    seen = []

    class FakeTrainer:
        def __init__(self, config):
            seen.append(config)

        def train(self):
            pass

    monkeypatch.setattr(module, "Trainer", FakeTrainer)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: SimpleNamespace(
        returncode=worker.main(command[command.index("agf.train") + 1:])
    ))
    for stage in (1, 2):
        args = ["train", "--gpus", "0", "--config", str(PACKAGE_ROOT / f"configs/train_stage{stage}.yaml"),
                "--logdir", f"stage{stage}/logs"]
        if stage == 2:
            args += ["--resume", str(tmp_path / "AGF/paper/stage1/logs/checkpoint_model_000200/model.pt"),
                     "--resume-step", "200"]
        assert cli.main(args) == 0
    first = OmegaConf.load(tmp_path / "AGF/paper/stage1/logs/resolved_config.yaml")
    second = OmegaConf.load(tmp_path / "AGF/paper/stage2/logs/resolved_config.yaml")
    assert first.max_steps == 200 and first.regularization_enabled
    assert second.max_steps == 400 and not second.regularization_enabled
    assert second.resume_step == 200 and not second.resume_reset_optimizers
    assert len(seen) == 2


@pytest.mark.parametrize("batch_size,count", [(1, 56), (4, 56), (4, 32), (2, 83)])
def test_training_batches_are_full_on_all_ranks_and_resume_order_is_preserved(tmp_path, batch_size, count):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(count)))
    for rank in range(8):
        def loader():
            return training_dataloader(prompts, batch_size=batch_size, world_size=8, rank=rank, num_workers=0)

        data = loader()
        assert all(len(batch["prompts"]) == batch_size for batch in data)
        assert len(data) == (count // 8) // batch_size
        uninterrupted = list(islice(cycle(data), 7))
        restored = cycle(loader())
        for _ in range(5):
            next(restored)
        assert next(restored)["prompts"] == uninterrupted[5]["prompts"]
        assert next(restored)["prompts"] == uninterrupted[6]["prompts"]


def test_training_rejects_dataset_that_cannot_supply_one_full_batch(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(31)))
    with pytest.raises(ValueError, match="distributed batch"):
        training_dataloader(prompts, batch_size=4, world_size=8, rank=0, num_workers=0)


@pytest.mark.parametrize("batch_size,count", [(1, 56), (4, 56), (4, 32)])
def test_legacy_resume_rejects_only_changed_partial_batch_policy(tmp_path, batch_size, count):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(count)))
    loader = training_dataloader(prompts, batch_size=batch_size, world_size=8, rank=0, num_workers=0)
    validate_dataloader_resume({"dataloader_drop_last": True}, loader)
    if batch_size == 4 and count == 56:
        with pytest.raises(ValueError, match="legacy checkpoint used partial"):
            validate_dataloader_resume({}, loader)
    else:
        validate_dataloader_resume({}, loader)


def test_unconditional_cache_tracks_actual_batch_size():
    weight = torch.nn.Parameter(torch.tensor(1.0))
    encoder_batches, observed = [], []

    def encoder(*, text_prompts):
        encoder_batches.append((text_prompts[0], len(text_prompts)))
        return {"prompt_embeds": torch.ones(len(text_prompts), 2)}

    def loss(**kwargs):
        size = kwargs["image_or_video_shape"][0]
        assert kwargs["conditional_dict"]["prompt_embeds"].shape[0] == size
        assert kwargs["unconditional_dict"]["prompt_embeds"].shape[0] == size
        observed.append(size)
        return weight.square(), {}

    trainer = SimpleNamespace(
        model=SimpleNamespace(eval=lambda: None, text_encoder=encoder, generator_loss=loss),
        config=SimpleNamespace(i2v=False, image_or_video_shape=[4, 21, 16, 60, 104], negative_prompt="negative"),
        fake_replay=None, gradient_accumulation_steps=1,
    )
    for count in (4, 4, 3, 4):
        Trainer.fwdbwd_one_step(trainer, {"prompts": ["positive"] * count}, True)
    assert observed == [4, 4, 3, 4]
    assert [n for text, n in encoder_batches if text == "negative"] == [4, 3, 4]
