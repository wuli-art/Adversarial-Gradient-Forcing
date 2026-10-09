"""Run the paper's regularized-200 -> full-state resume -> pure-AGF-400 recipe."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

from agf.utils.config import PACKAGE_ROOT, load_config
from agf.utils.cli import gpu_ids, launch_environment
from agf.utils.checkpoint import verify_checkpoint


def build_commands(args):
    env, run_dir = launch_environment(args)
    common = ["--gpus", ",".join(args.gpus)]
    for flag, value in (
        ("--wan-models", args.wan_models),
        ("--ode-checkpoint", args.ode_checkpoint),
        ("--training-prompts", args.training_prompts),
    ):
        if value:
            common += [flag, value]
    if args.wandb:
        common += ["--wandb"]
    stage1 = [
        sys.executable,
        "-m",
        "agf",
        "train",
        "--config",
        str(PACKAGE_ROOT / "configs/train_stage1.yaml"),
        "--logdir",
        "stage1/logs",
        *common,
    ]
    source = (
        Path(args.resume_stage2).expanduser().resolve()
        if args.resume_stage2
        else (run_dir / "stage1/logs/checkpoint_model_000200/model.pt")
    )
    stage2 = [
        sys.executable,
        "-m",
        "agf",
        "train",
        "--config",
        str(PACKAGE_ROOT / "configs/train_stage2.yaml"),
        "--logdir",
        "stage2/logs",
        *common,
        "--resume",
        str(source),
        "--resume-step",
        "200",
    ]
    # Validate batch geometry before any files are written or workers are started.
    for config in ("train_stage1.yaml", "train_stage2.yaml"):
        load_config(PACKAGE_ROOT / "configs" / config, world_size=len(args.gpus))
    return ([] if args.resume_stage2 else [stage1]) + [stage2], env, run_dir, source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=gpu_ids, default=gpu_ids("0,1,2,3,4,5,6,7"))
    parser.add_argument("--wan-models")
    parser.add_argument("--ode-checkpoint")
    parser.add_argument("--training-prompts")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-stage2", help="Existing complete step-200 model.pt; skip stage 1")
    args = parser.parse_args(argv)
    try:
        commands, env, run_dir, source = build_commands(args)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "commands": commands,
                    "transition": "step 200: retain all state; disable prediction consistency",
                },
                indent=2,
            )
        )
        if args.dry_run:
            for command in commands:
                subprocess.run([*command, "--dry-run"], env=env, check=True)
            return 0
        # A fresh stage must never overwrite an earlier training run.
        for stage in ["stage2"] if args.resume_stage2 else ["stage1", "stage2"]:
            if (run_dir / stage).exists():
                raise ValueError(f"Output already exists: {run_dir / stage}; use a fresh run ID")
        for command in commands:
            if "--resume" in command:
                verification = verify_checkpoint(source, step=200, world_size=len(args.gpus))
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "stage_transition.json").write_text(
                    json.dumps(
                        {
                            **verification,
                            "source": str(source),
                            "regularization_enabled": False,
                            "reset_optimizers": False,
                            "reset_replay": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            subprocess.run(command, env=env, check=True)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"AGF recipe: {exc}", file=sys.stderr)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 2


if __name__ == "__main__":
    raise SystemExit(main())
