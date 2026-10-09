"""T2V inference worker with deterministic, complete prompt sharding."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agf.utils.assets import model_root
from agf.utils.config import EXAMPLE_PROMPTS, load_config
from agf.utils.run_paths import resolve_run_output
from agf.utils.inference_outputs import (
    atomic_json, file_sha256, inference_request, output_lock, pending_samples,
    prepare_run, remove_incomplete_sample, sample_metadata, sample_path,
)


def prompt_indices(count: int, rank: int, world_size: int) -> range:
    """Visit every prompt exactly once, including an uneven final batch."""
    if count < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid prompt count or distributed rank")
    return range(rank, count, world_size)


def generator_weights(checkpoint: dict, use_ema: bool) -> dict:
    key = "generator_ema" if use_ema else "generator"
    if key not in checkpoint or checkpoint[key] is None:
        raise ValueError(f"Checkpoint has no {key}; choose a matching checkpoint or --no-ema")
    from agf.utils.distributed import clean_fsdp_state_dict

    return clean_fsdp_state_dict(checkpoint[key])


def write_video(destination: Path, frames, fps: int = 16) -> None:
    """Write a playable MP4 on local disks and sequential-write object mounts."""
    import imageio.v2 as imageio

    temporary = destination.with_suffix(".partial.mp4")
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Output exists; choose a new output directory: {destination}")
    with imageio.get_writer(
        temporary,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        # A fragmented MP4 does not seek back to rewrite the file header.
        output_params=["-movflags", "+frag_keyframe+empty_moov"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    # Validate all frames before publishing completion metadata used by --resume.
    with imageio.get_reader(temporary) as reader:
        actual = reader.get_meta_data()
        if (reader.count_frames() != len(frames)
                or abs(actual["fps"] - fps) > 0.01
                or reader.get_data(0).shape != frames[0].shape
                or reader.get_data(len(frames) - 1).shape != frames[-1].shape):
            raise ValueError(f"Encoded video has unexpected geometry, frame count or FPS: {temporary}")
    temporary.replace(destination)


def load_pipeline(config, checkpoint_path, use_ema):
    import torch
    from agf.pipeline.causal_inference import CausalInferencePipeline
    from agf.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    pipeline = CausalInferencePipeline(config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    pipeline.generator.load_state_dict(generator_weights(checkpoint, use_ema), strict=True)
    del checkpoint
    pipeline.to(dtype=torch.bfloat16)
    low_memory = get_cuda_free_memory_gb(device) < 40
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline, device, low_memory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", default=str(EXAMPLE_PROMPTS))
    parser.add_argument("--output_folder", default="videos")
    parser.add_argument(
        "--num_output_frames", type=int, default=21, help="Output latent frames, a multiple of 3"
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared-signature", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config_path, args.set, mode="inference")
        args.seed = config.seed if args.seed is None else args.seed
        if args.num_output_frames <= 0 or args.num_output_frames % 3:
            raise ValueError("num_output_frames must be a positive multiple of 3")
        if args.num_samples <= 0 or args.seed < 0:
            raise ValueError("num_samples must be positive and seed must be nonnegative")
        prompts = Path(args.data_path).read_text(encoding="utf-8").splitlines()
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("Provide one nonempty prompt per line")
        output = resolve_run_output("AGF", args.output_folder, "videos")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        indices = prompt_indices(len(prompts), rank, world_size)
        request = inference_request(
            config, prompts, seed=args.seed, latent_frames=args.num_output_frames,
            num_samples=args.num_samples, use_ema=args.use_ema, model_root=model_root(),
        )
        signature = prepare_run(
            output, args.checkpoint_path, request, resume=args.resume,
            prepared_signature=args.prepared_signature,
        )
        expected = {
            index: [sample_metadata(request, signature, index, prompts[index], sample)
                    for sample in range(args.num_samples)]
            for index in indices
        }
        # Fail on conflicts before allocating GPU memory, including direct workers.
        for metadata in expected.values():
            pending_samples(output, metadata, resume=args.resume)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    import torch
    from agf.utils.misc import set_seed

    pipeline = None
    for index, metadata in expected.items():
        with output_lock(output, f"prompt-{index:05d}.lock"):
            pending = pending_samples(output, metadata, resume=args.resume)
            if not pending:
                print(f"Prompt {index}: verified, skipping", flush=True)
                continue
            with torch.no_grad():
                if pipeline is None:
                    pipeline, device, low_memory = load_pipeline(
                        config, args.checkpoint_path, args.use_ema
                    )
                # Regenerate the whole sample batch to preserve its original RNG path.
                # Completed samples are verified above and never rewritten.
                set_seed(args.seed + index)
                noise = torch.randn(
                    [args.num_samples, args.num_output_frames, 16, 60, 104],
                    device=device, dtype=torch.bfloat16,
                )
                video = pipeline.inference(
                    noise=noise, text_prompts=[prompts[index]] * args.num_samples,
                    low_memory=low_memory,
                )
                frames = (video.permute(0, 1, 3, 4, 2).cpu().float() * 255).clamp(0, 255).to(torch.uint8)
                pipeline.vae.model.clear_cache()
            for sample in pending:
                destination = sample_path(output, metadata[sample])
                if tuple(frames[sample].shape) != (
                    metadata[sample]["decoded_frames"], request["height"], request["width"], 3
                ):
                    raise ValueError("Generated video geometry differs from the inference request")
                remove_incomplete_sample(destination)
                write_video(destination, frames[sample].numpy())
                atomic_json(destination.with_suffix(".json"), {
                    **metadata[sample], "checkpoint": str(Path(args.checkpoint_path).resolve()),
                    "video_bytes": destination.stat().st_size,
                    "video_sha256": file_sha256(destination),
                })
            del video, frames, noise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
