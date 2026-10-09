import os
import subprocess
import sys
from pathlib import Path

import pytest

from agf.utils.config import load_config, resolve_input_paths
from agf.utils.cli import launch_environment, parser
from agf.utils.run_paths import resolve_run_output


@pytest.mark.parametrize(
    "override",
    [
        "attak_epsilon_rms=0.1",
        "model_kwargs.typo=4",
        "attack_create_graph=true",
        "pass1_serial_no_grad=false",
        "frozen_reference_velocity_anchor_enabled=true",
        "attack_steps=2",
        "clean_loss_weight=0.8",
        "total_batch_size=7",
        "num_training_frames=24",
        "model_kwargs.model_name=Wan2.1-T2V-14B",
        "fake_replay_enabled=false",
        "gradient_checkpointing='false'",
        "max_steps=2.5",
        "attack_epsilon_rms=.nan",
        "lr=-1",
        "sharding_strategy=oops",
        "random_noise_epsilon_end=0.3",
        "clean_warmup_end=1",
        "long_rollout_enabled=true",
        "rollout_stage1_end=20",
        "generator_fsdp_wrap_strategy=transformer",
    ],
)
def test_invalid_overrides_fail_before_launch(override):
    with pytest.raises(ValueError):
        load_config(overrides=[override], world_size=8)


def test_yaml_and_cli_precedence(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("attack_epsilon_rms: 0.1\nmax_steps: 400\n")
    config = load_config(config_file, ["max_steps=600"])
    assert config.max_steps == 600
    assert config.attack_epsilon_rms == 0.1
    assert config.clean_loss_weight == 0.5


def test_valid_resume_and_invalid_step():
    overrides = ["resume_checkpoint=/models/model.pt", "resume_step=200"]
    assert load_config(overrides=overrides).resume_step == 200
    with pytest.raises(ValueError, match="resume_step"):
        load_config(overrides=overrides + ["max_steps=200"])


def test_paths_use_invocation_directory(tmp_path):
    config = resolve_input_paths(load_config(), tmp_path)
    assert config.generator_ckpt == str(tmp_path / "checkpoints/ode_init.pt")


def test_launch_propagates_run_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("AGF_RUN_ID", "test")
    args = parser().parse_args(["train", "--gpus", "2,3", "--dry-run"])
    env, directory = launch_environment(args)
    assert directory == tmp_path / "AGF/test"
    assert env["AGF_RUNS_ROOT"] == str(tmp_path)
    assert env["AGF_RUN_ID"] == "test"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert not directory.exists()


def test_outputs_allow_absolute_paths_and_keep_relative_paths_in_run(monkeypatch, tmp_path):
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("AGF_RUN_ID", "paper")
    assert resolve_run_output("AGF", tmp_path / "outside", "logs") == tmp_path / "outside"
    assert (
        resolve_run_output("AGF", "stage1/logs", "logs") == tmp_path / "runs/AGF/paper/stage1/logs"
    )


def test_unrelated_workspace_environment_cannot_override_agf(monkeypatch, tmp_path):
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path / "agf"))
    monkeypatch.setenv("AGF_RUN_ID", "requested")
    monkeypatch.setenv("KV_RUNS_ROOT", str(tmp_path / "other"))
    monkeypatch.setenv("KV_RUN_DIR", str(tmp_path / "other/forced"))
    monkeypatch.setenv("KV_RUN_ID", "inherited")
    monkeypatch.setenv("KV_ENFORCE_DURABLE_OUTPUTS", "1")
    assert resolve_run_output("AGF", None, ".") == tmp_path / "agf/AGF/requested"
    assert resolve_run_output("AGF", tmp_path / "outside", "logs") == tmp_path / "outside"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "a/b", "a\\b"])
def test_run_id_is_one_directory_name(monkeypatch, run_id):
    monkeypatch.setenv("AGF_RUN_ID", run_id)
    with pytest.raises(ValueError, match="AGF_RUN_ID"):
        resolve_run_output("AGF", None, ".")


def test_cli_dry_run_has_no_side_effects(tmp_path):
    env = dict(os.environ, AGF_RUNS_ROOT=str(tmp_path), AGF_RUN_ID="dry")
    result = subprocess.run(
        [sys.executable, "-m", "agf", "train", "--dry-run", "--set", "max_steps=1"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"gradient_accumulation_steps": 8' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_worker_checks_invalid_config_before_torch_import():
    result = subprocess.run(
        [sys.executable, "-m", "agf.train", "--set", "attack_steps=2"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "attack_steps must be 1" in result.stderr


def test_inference_config_seed_is_used_by_launcher(tmp_path, monkeypatch, capsys):
    from agf.utils.cli import main

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded during dry run")
    monkeypatch.setenv("AGF_RUNS_ROOT", str(tmp_path / "runs"))
    assert main(["infer", "--checkpoint", str(checkpoint), "--set", "seed=19", "--dry-run"]) == 0
    assert "--seed 19" in capsys.readouterr().out
