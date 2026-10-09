"""Portable AGF commands, usable from an editable checkout or installed wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from agf import __version__
from agf.utils.assets import check_model, model_root, training_asset_errors
from agf.utils.config import EXAMPLE_PROMPTS, load_config, resolve_input_paths
from agf.utils.run_paths import resolve_run_output
from agf.utils.inference_outputs import inference_request, prepare_run


def gpu_ids(value: str) -> list[str]:
    ids = [part.strip() for part in value.split(",")]
    if not ids or any(not part.isdigit() for part in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("--gpus must contain unique indices, e.g. 0,1,2,3")
    return ids


def environment_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "agf": __version__}
    for name in ("torch", "diffusers", "transformers", "huggingface-hub", "flash-attn", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def launch_environment(args: argparse.Namespace) -> tuple[dict[str, str], Path]:
    env = os.environ.copy()
    env["AGF_WAN_MODELS_DIR"] = str(model_root(args.wan_models))
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
    run_dir = resolve_run_output("AGF", None, ".")
    # Fix the run identity once, then propagate it to every child and rank.
    env["AGF_RUNS_ROOT"] = str(run_dir.parent.parent)
    env["AGF_RUN_ID"] = run_dir.name
    return env, run_dir


def _overrides(args: argparse.Namespace) -> list[str]:
    values = list(args.set_values)
    if getattr(args, "ode_checkpoint", None):
        values.append(f"generator_ckpt={args.ode_checkpoint}")
    if getattr(args, "training_prompts", None):
        values.append(f"data_path={args.training_prompts}")
    if getattr(args, "resume", None):
        values.append(f"resume_checkpoint={args.resume}")
    if getattr(args, "resume_step", None) is not None:
        values.append(f"resume_step={args.resume_step}")
    return values


def train(args: argparse.Namespace) -> int:
    world_size = len(args.gpus)
    config = resolve_input_paths(load_config(args.config, _overrides(args), world_size=world_size))
    env, run_dir = launch_environment(args)
    logdir = resolve_run_output("AGF", args.logdir, "logs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        "--module",
        "agf.train",
        "--logdir",
        str(logdir),
    ]
    if args.config:
        command += ["--config_path", str(Path(args.config).resolve())]
    for value in _overrides(args):
        command += ["--set", value]
    if args.no_save:
        command.append("--no_save")
    if args.wandb:
        command.append("--wandb")
    payload = {
        "command": shlex.join(command),
        "run_dir": str(run_dir),
        "logdir": str(logdir),
        "world_size": world_size,
        "gradient_accumulation_steps": config.total_batch_size // (config.batch_size * world_size),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return 0
    if logdir.exists():
        raise FileExistsError(
            f"Training output already exists: {logdir}; choose a new run ID or --logdir, "
            "including when resuming from a checkpoint"
        )
    errors = training_asset_errors(config, Path(env["AGF_WAN_MODELS_DIR"]), world_size)
    if errors:
        raise ValueError("\n".join(errors))
    if args.wandb and importlib.util.find_spec("wandb") is None:
        raise ValueError("Install the logging extra: pip install '.[logging]'")
    return subprocess.run(command, env=env, check=False).returncode


def infer(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set_values, mode="inference")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.latent_frames <= 0 or args.latent_frames % config.num_frame_per_block:
        raise ValueError("--latent-frames must be a positive multiple of 3")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    prompts = Path(args.prompts).expanduser().resolve()
    for label, path in (("checkpoint", checkpoint), ("prompts", prompts)):
        if not path.is_file():
            raise ValueError(f"Missing {label}: {path}")
    env, run_dir = launch_environment(args)
    output = resolve_run_output("AGF", args.output, "videos")
    command = [sys.executable]
    if len(args.gpus) > 1:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(len(args.gpus)),
            "--module",
            "agf.inference",
        ]
    else:
        command += ["-m", "agf.inference"]
    command += [
        "--checkpoint_path",
        str(checkpoint),
        "--data_path",
        str(prompts),
        "--output_folder",
        str(output),
        "--num_output_frames",
        str(args.latent_frames),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(config.seed if args.seed is None else args.seed),
    ]
    if args.config:
        command += ["--config_path", str(Path(args.config).resolve())]
    for value in args.set_values:
        command += ["--set", value]
    if not args.no_ema:
        command.append("--use_ema")
    if args.resume:
        command.append("--resume")
    print(
        json.dumps(
            {"command": shlex.join(command), "output": str(output), "run_dir": str(run_dir)},
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    errors = check_model(Path(env["AGF_WAN_MODELS_DIR"]), "Wan2.1-T2V-1.3B")
    if errors:
        raise ValueError("\n".join(errors))
    prompt_lines = prompts.read_text(encoding="utf-8").splitlines()
    if not prompt_lines or any(not p.strip() for p in prompt_lines):
        raise ValueError("Provide one nonempty prompt per line")
    request = inference_request(
        config, prompt_lines, seed=config.seed if args.seed is None else args.seed,
        latent_frames=args.latent_frames, num_samples=args.num_samples,
        use_ema=not args.no_ema, model_root=env["AGF_WAN_MODELS_DIR"],
    )
    signature = prepare_run(
        output, checkpoint, request, resume=args.resume, check_existing=True,
    )
    command += ["--prepared-signature", signature]
    return subprocess.run(command, env=env, check=False).returncode


def verify(args: argparse.Namespace) -> int:
    config = resolve_input_paths(
        load_config(args.config, _overrides(args), world_size=len(args.gpus))
    )
    errors = []
    runtime = {}
    if args.require_assets:
        errors += training_asset_errors(config, model_root(args.wan_models), len(args.gpus))
    if args.runtime:
        try:
            import torch
            from agf.model.adversarial_history_dmd import AdversarialHistoryDMD
            from agf.pipeline.causal_inference import CausalInferencePipeline
            from agf.wan.modules.attention import (
                FLASH_ATTN_2_AVAILABLE,
                FLASH_ATTN_3_AVAILABLE,
            )

            runtime = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "flash_attention": FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE,
                "core_imports": bool(AdversarialHistoryDMD and CausalInferencePipeline),
            }
        except Exception as exc:
            errors.append(f"Runtime import failed: {type(exc).__name__}: {exc}")
    print(
        json.dumps(
            {
                "ok": not errors,
                "errors": errors,
                "versions": environment_versions(),
                "runtime": runtime,
                "assets_checked": args.require_assets,
            },
            indent=2,
        )
    )
    return int(bool(errors))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Adversarial Gradient Forcing")
    result.add_argument("--version", action="version", version=__version__)
    subs = result.add_subparsers(dest="command", required=True)
    for name in ("train", "verify", "infer"):
        p = subs.add_parser(name)
        p.add_argument("--config", help="YAML overrides; relative inputs use the current directory")
        p.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--wan-models", help="Directory containing the Wan model directories")
        p.add_argument(
            "--gpus", type=gpu_ids, default=gpu_ids("0" if name == "infer" else "0,1,2,3,4,5,6,7")
        )
        if name in ("train", "verify"):
            p.add_argument("--ode-checkpoint")
            p.add_argument("--training-prompts")
            p.add_argument(
                "--resume", help="Full AGF model.pt; keep adjacent trainer_state directory"
            )
            p.add_argument("--resume-step", type=int)
        if name in ("train", "infer"):
            p.add_argument(
                "--dry-run",
                action="store_true",
                help="Validate and print the launch; write no files",
            )
        if name == "train":
            p.add_argument("--logdir", default="logs")
            p.add_argument("--no-save", action="store_true")
            p.add_argument("--wandb", action="store_true", help="Enable optional W&B logging")
        elif name == "verify":
            p.add_argument("--require-assets", action="store_true")
            p.add_argument(
                "--runtime", action="store_true", help="Import the training and inference stack"
            )
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--prompts", default=str(EXAMPLE_PROMPTS))
            p.add_argument("--output", default="videos")
            p.add_argument("--latent-frames", type=int, default=21)
            p.add_argument("--num-samples", type=int, default=1)
            p.add_argument("--seed", type=int, help="Override the inference config seed")
            p.add_argument("--no-ema", action="store_true")
            p.add_argument("--resume", action="store_true", help="Verify and skip completed videos")
    p = subs.add_parser("config", help="Print the packaged default configuration")
    p.add_argument("--mode", choices=("train", "inference"), default="train")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "config":
            print(OmegaConf.to_yaml(load_config(mode=args.mode)), end="")
            return 0
        return {"train": train, "infer": infer, "verify": verify}[args.command](args)
    except (ValueError, OSError) as exc:
        print(f"agf: {exc}", file=sys.stderr)
        return 2
