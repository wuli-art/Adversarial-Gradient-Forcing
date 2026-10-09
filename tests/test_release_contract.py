import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agf.utils.config import PACKAGE_ROOT, load_config
from agf.utils.checkpoint import load_optimizer_bundle, verify_checkpoint
from agf.utils.recipe import build_commands
from agf.trainer.distillation import Trainer, FakeScoreReplayBuffer


def test_paper_prompt_counts_and_integrity():
    root = PACKAGE_ROOT / "prompts/benchmark"
    manifest = json.loads((root / "manifest.json").read_text())
    for phase, count in [("5s", 944), ("60s", 240), ("240s", 120)]:
        meta = manifest["cohorts"][phase]
        p = root / meta["file"]
        lines = p.read_text().splitlines()
        assert len(lines) == len(set(lines)) == count
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]
    assert set((root / manifest["cohorts"]["60s"]["file"]).read_text().splitlines()) <= set(
        (root / manifest["cohorts"]["5s"]["file"]).read_text().splitlines()
    )


def test_stage_launch_preserves_identity_and_explicit_resume(monkeypatch, tmp_path):
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("AGF_RUN_ID", "paper")
    args = SimpleNamespace(
        gpus=["0", "1", "2", "3", "4", "5", "6", "7"],
        wan_models=None,
        ode_checkpoint=None,
        training_prompts=None,
        wandb=False,
        resume_stage2=None,
    )
    commands, env, root, source = build_commands(args)
    assert len(commands) == 2 and env["AGF_RUN_ID"] == "paper"
    assert env["AGF_RUNS_ROOT"] == str(tmp_path)
    assert source == root / "stage1/logs/checkpoint_model_000200/model.pt"
    assert commands[1][-4:] == ["--resume", str(source), "--resume-step", "200"]
    assert not root.exists()
    first = load_config(PACKAGE_ROOT / "configs/train_stage1.yaml")
    second = load_config(PACKAGE_ROOT / "configs/train_stage2.yaml")
    assert first.max_steps == 200 and first.regularization_enabled
    assert second.max_steps == 400 and not second.regularization_enabled
    assert not second.resume_reset_optimizers and not second.resume_reset_replay


def test_real_trainer_save_v3_and_v2_compatibility(monkeypatch, tmp_path):
    import agf.trainer.distillation as module

    generator = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 2)
    go = torch.optim.AdamW(generator.parameters())
    co = torch.optim.AdamW(critic.parameters())
    for model, opt in [(generator, go), (critic, co)]:
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
    replay = FakeScoreReplayBuffer(8, 0.5, 0.5, 1)
    for kind in ("clean", "robust"):
        replay.push(
            {"kind": kind, "generated_image": torch.ones(1, 3, 1, 1, 1), "conditional_dict": {}}
        )
    trainer = SimpleNamespace(
        is_main_process=True,
        config=SimpleNamespace(save_training_state_steps=[200]),
        step=200,
        model=SimpleNamespace(generator=generator, fake_score=critic),
        generator_optimizer=go,
        critic_optimizer=co,
        generator_ema=SimpleNamespace(state_dict=generator.state_dict),
        initialization_metadata={},
        output_path=str(tmp_path),
        global_rank=0,
        world_size=1,
        dataloader_batches_consumed=32,
        fake_replay=replay,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(module, "fsdp_state_dict", lambda model: model.state_dict())
    monkeypatch.setattr(module, "fsdp_optim_state_dict", lambda model, opt: opt.state_dict())
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: torch.get_rng_state())
    Trainer.save(trainer)
    path = tmp_path / "checkpoint_model_000200/model.pt"
    assert verify_checkpoint(path, step=200, world_size=1)["format"] == 3
    original_files = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        Trainer.save(trainer)
    assert all(p.read_bytes() == content for p, content in original_files.items())
    model = torch.load(path, weights_only=False)
    assert "generator_optimizer" not in model
    bundle = load_optimizer_bundle(model, path)
    assert bundle["generator_optimizer"]["state"]
    old = {
        **model,
        **{k: bundle[k] for k in ("generator_optimizer", "critic_optimizer")},
        "checkpoint_format_version": 2,
    }
    torch.save(old, path)
    assert verify_checkpoint(path, step=200, world_size=1)["format"] == 2
    with pytest.raises(ValueError, match="world size"):
        verify_checkpoint(path, step=200, world_size=2)
    with pytest.raises(ValueError, match="step"):
        verify_checkpoint(path, step=201, world_size=1)
    # Missing state must fail, rather than silently becoming a reset.
    del old["critic_optimizer"]
    torch.save(old, path)
    with pytest.raises(ValueError, match="optimizer"):
        verify_checkpoint(path, step=200, world_size=1)


def test_evaluator_rejects_partial_or_nonfinite_results(tmp_path):
    from agf.evaluation.vbench import read_result

    p = tmp_path / "result.json"
    p.write_text(
        json.dumps({"subject_consistency": [0.9, [{"video_path": "x", "video_results": 0.9}]]})
    )
    assert read_result(p, "subject_consistency", expected_records=1)["detail_records"] == 1
    with pytest.raises(RuntimeError, match="count"):
        read_result(p, "subject_consistency", expected_records=2)
    p.write_text('{"subject_consistency": [NaN, []]}')
    with pytest.raises(RuntimeError, match="non-finite"):
        read_result(p, "subject_consistency", expected_records=0)
