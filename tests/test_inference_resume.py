import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agf import inference
from agf.utils import cli
from agf.utils.config import load_config
from agf.utils.inference_outputs import (
    atomic_json, file_sha256, inference_request, output_lock, pending_samples,
    prepare_run, sample_metadata, sample_path,
)


def run_inputs(tmp_path, num_samples=1):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint contents")
    prompts = ["first prompt", "second prompt", "third prompt"]
    request = inference_request(
        load_config(mode="inference"), prompts, seed=19, latent_frames=3,
        num_samples=num_samples, use_ema=True, model_root=tmp_path / "models",
    )
    # Small real encoded videos exercise integrity without model or GPU dependencies.
    request.update(width=48, height=32)
    output = tmp_path / "videos"
    signature = prepare_run(output, checkpoint, request, resume=False)
    return checkpoint, prompts, request, output, signature


def complete_sample(output, metadata):
    path = sample_path(output, metadata)
    frames = np.full((metadata["decoded_frames"], metadata["height"], metadata["width"], 3), 100, dtype=np.uint8)
    inference.write_video(path, frames)
    atomic_json(path.with_suffix(".json"), {
        **metadata, "video_sha256": file_sha256(path), "video_bytes": path.stat().st_size,
    })
    return path


@pytest.mark.parametrize("change", ["seed", "ema", "prompts", "config", "samples", "frames", "checkpoint"])
def test_resume_rejects_changed_inputs_and_leaves_old_outputs_intact(tmp_path, change):
    checkpoint, prompts, request, output, signature = run_inputs(tmp_path)
    meta = sample_metadata(request, signature, 0, prompts[0], 0)
    video = complete_sample(output, meta)
    before = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
    if change == "seed":
        request["seed"] += 1
    elif change == "ema":
        request["ema"] = False
    elif change == "prompts":
        request["prompts_sha256"] = "different order"
    elif change == "config":
        request["config"]["model_kwargs"]["local_attn_size"] = 9
    elif change == "samples":
        request["num_samples"] = 2
    elif change == "frames":
        request["latent_frames"] = 21
    else:
        checkpoint.write_bytes(b"different contents!")
    with pytest.raises(ValueError, match="differ"):
        prepare_run(output, checkpoint, request, resume=True)
    assert video.exists()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_partial_sample_batch_skips_only_verified_outputs(tmp_path):
    checkpoint, prompts, request, output, signature = run_inputs(tmp_path, num_samples=3)
    expected = [sample_metadata(request, signature, 0, prompts[0], s) for s in range(3)]
    good = complete_sample(output, expected[0])
    damaged = complete_sample(output, expected[1])
    # Same-size corruption must not be accepted by only checking file size.
    contents = bytearray(damaged.read_bytes())
    contents[-1] ^= 1
    damaged.write_bytes(contents)
    sample_path(output, expected[2]).with_suffix(".partial.mp4").write_bytes(b"interrupted")
    assert prepare_run(output, checkpoint, request, resume=True) == signature
    assert pending_samples(output, expected, resume=True) == [1, 2]
    with pytest.raises(FileExistsError):
        pending_samples(output, expected, resume=False)
    assert good.exists()
    wrong = json.loads(good.with_suffix(".json").read_text())
    wrong["seed"] += 1
    atomic_json(good.with_suffix(".json"), wrong)
    with pytest.raises(ValueError, match="metadata differs"):
        pending_samples(output, expected, resume=True)


def test_launch_preparation_hashes_once_and_workers_reject_checkpoint_changes(tmp_path, monkeypatch):
    checkpoint, _, request, output, signature = run_inputs(tmp_path)
    import agf.utils.inference_outputs as module

    monkeypatch.setattr(module, "file_sha256", lambda p: pytest.fail("worker rehashed checkpoint"))
    assert prepare_run(output, checkpoint, request, resume=False, prepared_signature=signature) == signature
    checkpoint.write_bytes(b"changed after launch")
    with pytest.raises(ValueError, match="changed after"):
        prepare_run(output, checkpoint, request, resume=True, prepared_signature=signature)


def test_legacy_outputs_are_not_silently_adopted(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "old"
    output.mkdir()
    old = output / "00000-00_ema.mp4"
    old.write_bytes(b"unidentified old video")
    with pytest.raises(ValueError, match="no inference manifest"):
        prepare_run(output, checkpoint, {}, resume=True)
    assert old.read_bytes() == b"unidentified old video"


def test_prompt_lock_prevents_concurrent_writers(tmp_path):
    with output_lock(tmp_path, "prompt-00000.lock"):
        with pytest.raises(RuntimeError, match="Another inference worker"):
            with output_lock(tmp_path, "prompt-00000.lock"):
                pytest.fail("concurrent write allowed")


def test_cli_forwards_resume_and_prepared_signature(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first prompt\n")
    monkeypatch.setattr(cli, "check_model", lambda *a: [])
    launches = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: (
        launches.append(command) or SimpleNamespace(returncode=0)
    ))
    assert cli.main(["infer", "--checkpoint", str(checkpoint), "--prompts", str(prompts),
                     "--output", str(tmp_path / "videos"), "--resume", "--gpus", "0,1"]) == 0
    assert "--resume" in launches[0]
    assert "--prepared-signature" in launches[0]
    assert launches[0][launches[0].index("--nproc_per_node") + 1] == "2"


def test_worker_resume_preserves_samples_and_seeds_across_world_sizes(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\nthird\n")
    output = tmp_path / "videos"
    calls = []
    loads = []

    class TinyPipeline:
        vae = SimpleNamespace(model=SimpleNamespace(clear_cache=lambda: None))

        def inference(self, noise, text_prompts, **kw):
            calls.append((text_prompts[0], noise.clone()))
            # Distinct samples retain a deterministic dependence on the original RNG.
            levels = torch.sigmoid(noise[:, 0, 0, 0, 0].float())
            return levels[:, None, None, None, None].expand(-1, 9, 3, 480, 832)

    def load(*args):
        loads.append(1)
        return TinyPipeline(), torch.device("cpu"), False

    monkeypatch.setattr(inference, "load_pipeline", load)
    args = ["--checkpoint_path", str(checkpoint), "--data_path", str(prompts),
            "--output_folder", str(output), "--num_output_frames", "3", "--num_samples", "2",
            "--use_ema", "--seed", "23"]
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert inference.main(args) == 0  # first launch completed prompts 0 and 2 only
    good = output / "00000-00_ema.mp4"
    before = good.read_bytes(), good.stat().st_mtime_ns
    incomplete = output / "00000-01_ema.mp4"
    incomplete.unlink()  # interruption partway through a sample batch
    incomplete.with_suffix(".partial.mp4").write_bytes(b"partial")
    original_noise = calls[0][1]
    calls.clear()
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert inference.main([*args, "--resume"]) == 0
    assert [p for p, _ in calls] == ["first", "second"]
    assert torch.equal(calls[0][1], original_noise)
    assert (good.read_bytes(), good.stat().st_mtime_ns) == before
    assert len(list(output.glob("*.mp4"))) == 6
    assert not list(output.glob("*.partial*"))
    for index in range(3):
        meta = json.loads((output / f"{index:05d}-00_ema.json").read_text())
        assert meta["seed"] == 23 + index
    calls.clear()
    loads.clear()
    assert inference.main([*args, "--resume"]) == 0
    assert not calls and not loads  # completed jobs never allocate the model
    with pytest.raises(SystemExit):
        inference.main(args)  # no --resume must fail before loading the model
    assert not loads
