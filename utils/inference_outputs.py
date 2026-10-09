"""Bind resumable inference outputs to their inputs and validate completed videos."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def checkpoint_stat(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev, "inode": stat.st_ino,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def output_lock(output: Path, name: str, *, blocking: bool = False):
    directory = output / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another inference worker is using {output / name}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def inference_request(config, prompts, *, seed, latent_frames, num_samples, use_ema, model_root):
    resolved = OmegaConf.to_container(config, resolve=True)
    resolved["seed"] = seed
    return {
        "config": resolved,
        "prompts_sha256": _digest(prompts), "prompt_count": len(prompts),
        "seed": seed, "latent_frames": latent_frames, "num_samples": num_samples,
        "ema": use_ema, "width": 832, "height": 480, "fps": 16,
        "wan_models": str(Path(model_root).resolve()),
    }


def prepare_run(output, checkpoint, request, *, resume, prepared_signature=None,
                check_existing=False):
    """Hash the checkpoint once in the launcher; workers validate its unchanged stat.

    Direct workers also work, but each hashes the checkpoint independently. Ranks
    and GPU count are deliberately excluded: prompt seeding is rank-independent.
    """
    output, checkpoint = Path(output), Path(checkpoint)
    with output_lock(output, "manifest.lock", blocking=True):
        manifest_path = output / "inference_run.json"
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if old is not None and (old.get("version") != 1 or old.get("request") != request):
            raise ValueError("Inference inputs differ from this output directory; use a new output")
        if prepared_signature is not None:
            if (old is None or old.get("signature") != prepared_signature
                    or old.get("checkpoint_stat") != checkpoint_stat(checkpoint)):
                raise ValueError("Checkpoint or inference manifest changed after launch preparation")
            return prepared_signature
        if old is None and any(p.name != ".locks" for p in output.iterdir()):
            raise ValueError("Existing output has no inference manifest; use a new output directory")
        if check_existing and not resume and any(
            p.name not in {".locks", "inference_run.json", "inference_config.yaml"}
            for p in output.iterdir()
        ):
            raise FileExistsError("Inference outputs already exist; use --resume or a new output")
        before = checkpoint_stat(checkpoint)
        identity = {"request": request, "checkpoint_sha256": file_sha256(checkpoint)}
        if checkpoint_stat(checkpoint) != before:
            raise ValueError("Checkpoint changed while hashing; use a completed checkpoint")
        signature = _digest(identity)
        if old is not None and old.get("signature") != signature:
            raise ValueError("Checkpoint contents differ from this output directory; use a new output")
        atomic_json(manifest_path, {
            "version": 1, **identity, "signature": signature, "checkpoint_stat": before,
        })
        temporary = output / "inference_config.yaml.partial"
        OmegaConf.save(OmegaConf.create(request["config"]), temporary)
        temporary.replace(output / "inference_config.yaml")
        return signature


def sample_metadata(request, signature, index, prompt, sample):
    return {
        "run_signature": signature, "prompt_index": index, "prompt": prompt,
        "seed": request["seed"] + index, "sample_index": sample,
        "ema": request["ema"], "latent_frames": request["latent_frames"],
        "decoded_frames": 4 * request["latent_frames"] - 3,
        "fps": request["fps"], "width": request["width"], "height": request["height"],
    }


def sample_path(output, metadata):
    suffix = "ema" if metadata["ema"] else "generator"
    return Path(output) / f'{metadata["prompt_index"]:05d}-{metadata["sample_index"]:02d}_{suffix}.mp4'


def pending_samples(output, expected, *, resume):
    """Return incomplete sample indices; refuse valid metadata for different inputs."""
    pending = []
    for sample, metadata in enumerate(expected):
        path = sample_path(output, metadata)
        sidecar = path.with_suffix(".json")
        partial = path.with_suffix(".partial.mp4")
        if not resume and any(p.exists() for p in (path, sidecar, partial)):
            raise FileExistsError(f"Output exists: {path}; use --resume or a new output directory")
        saved = None
        if sidecar.exists():
            try:
                saved = json.loads(sidecar.read_text())
            except (ValueError, UnicodeDecodeError):
                pass  # A damaged sidecar is incomplete, not a completed sample.
        if saved is not None:
            if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in metadata.items()):
                raise ValueError(f"Video metadata differs from requested inputs: {sidecar}")
            if (path.is_file() and saved.get("video_bytes") == path.stat().st_size
                    and saved.get("video_sha256") == file_sha256(path)):
                continue
        pending.append(sample)
    return pending


def remove_incomplete_sample(path):
    """Called only under the prompt lock after manifest and metadata validation."""
    for candidate in (path, path.with_suffix(".partial.mp4"), path.with_suffix(".json"),
                      path.with_suffix(".json.partial")):
        candidate.unlink(missing_ok=True)
