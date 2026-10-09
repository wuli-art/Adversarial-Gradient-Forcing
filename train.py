"""Distributed training worker. Usually launched through ``agf train``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf, open_dict

from agf.utils.assets import model_root, training_asset_errors
from agf.utils.config import load_config, resolve_input_paths
from agf.utils.cli import environment_versions
from agf.utils.run_paths import resolve_run_output
import json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path")
    parser.add_argument("--config_override", nargs="*", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        config = resolve_input_paths(
            load_config(args.config_path, args.config_override + args.set, world_size=world_size)
        )
        errors = training_asset_errors(config, model_root(), world_size)
        if errors:
            raise ValueError("\n".join(errors))
        logdir = resolve_run_output("AGF", args.logdir, "logs")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with open_dict(config):
        config.no_save = args.no_save
        config.no_visualize = True
        config.logdir = str(logdir)
        config.wandb_save_dir = str(logdir / "wandb")
        config.disable_wandb = not args.wandb
        config.config_name = Path(args.config_path).stem if args.config_path else "agf_direct"
    if int(os.environ.get("RANK", "0")) == 0:
        # Reserve the destination atomically, including direct torchrun launches.
        # Resume reads its source checkpoint but always writes to a fresh directory.
        try:
            logdir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f"Training output already exists: {logdir}; choose a new --logdir")
        OmegaConf.save(config, logdir / "resolved_config.yaml")
        (logdir / "environment.json").write_text(
            json.dumps(environment_versions(), indent=2) + "\n"
        )
    from agf.trainer.distillation import Trainer

    try:
        trainer = Trainer(config)
        trainer.train()
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        if args.wandb:
            import wandb

            wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
