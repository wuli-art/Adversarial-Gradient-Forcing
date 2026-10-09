#!/usr/bin/env python3
"""Run resumable 5s/60s official-full-16 and 240s custom-six VBench."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any
import uuid


FULL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)
CUSTOM_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
LIGHT_240S_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
)
HEAVY_240S_DIMENSIONS = (
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
)
EXPECTED_DETAIL_RECORDS = {
    "5s": {
        "aesthetic_quality": 93,
        "appearance_style": 90,
        "background_consistency": 86,
        # The pinned official cohort contains 85 color prompts. GRiT emits a
        # detail row only when the prompted object is detected, so the valid
        # range is content-dependent and bounded by that cohort cardinality.
        "color": (1, 85),
        "dynamic_degree": 72,
        "human_action": 100,
        "imaging_quality": 93,
        "motion_smoothness": 72,
        "multiple_objects": 82,
        "object_class": 79,
        "overall_consistency": 93,
        "scene": 86,
        "spatial_relationship": 84,
        "subject_consistency": 72,
        "temporal_flickering": 75,
        "temporal_style": 100,
    },
    "60s": {
        "aesthetic_quality": 120,
        "appearance_style": 552,
        "background_consistency": 1260,
        # As above, up to 20 color prompts x 30 long-video clips.
        "color": (1, 600),
        "dynamic_degree": 600,
        "human_action": 120,
        "imaging_quality": 120,
        "motion_smoothness": 600,
        "multiple_objects": 600,
        "object_class": 600,
        "overall_consistency": 120,
        "scene": 1260,
        "spatial_relationship": 600,
        "subject_consistency": 600,
        # VBench-Long first filters clips classified as static, so the result
        # inventory is content-dependent (56..91 in the reference models).
        "temporal_flickering": (1, 600),
        "temporal_style": 120,
    },
    "240s": {dimension: 14400 for dimension in CUSTOM_DIMENSIONS},
}
DIMENSION_RANGES = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.364),
    "overall_consistency": (0.0, 0.364),
}
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
SEMANTIC_DIMENSIONS = (
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_prompts(path: Path) -> list[str]:
    prompts = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError(f"prompt cohort must be unique: {path}")
    return prompts


def read_result(
    path: Path,
    dimension: str,
    *,
    expected_records: int | tuple[int, int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(dimension)
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise RuntimeError(f"unexpected {dimension} payload in {path}")
    score = float(value[0])
    if not math.isfinite(score):
        raise RuntimeError(f"non-finite {dimension} score in {path}: {score}")
    details = value[1]
    if isinstance(expected_records, int):
        record_count_valid = isinstance(details, list) and len(details) == expected_records
        expected_description = str(expected_records)
    else:
        minimum, maximum = expected_records
        record_count_valid = isinstance(details, list) and minimum <= len(details) <= maximum
        expected_description = f"[{minimum}, {maximum}]"
    if not record_count_valid:
        raise RuntimeError(
            f"{dimension} detail count mismatch in {path}: "
            f"{len(details) if isinstance(details, list) else 'invalid'} "
            f"not in expected {expected_description}"
        )
    paths = [
        row.get("video_path") for row in details if isinstance(row, dict) and row.get("video_path")
    ]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate detail video paths in {path}")
    for row in details:
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid detail row in {path}")
        if isinstance(row.get("video_results"), (int, float)) and not math.isfinite(
            row["video_results"]
        ):
            raise RuntimeError(f"non-finite detail result in {path}")
    averages = value[2] if len(value) == 3 else None
    if averages is not None and not isinstance(averages, list):
        raise RuntimeError(f"invalid long-video averages in {path}")
    return {
        "score": score,
        "detail_records": len(details),
        "long_video_records": len(averages) if averages is not None else None,
        "payload_length": len(value),
        "result_path": str(path.resolve()),
        "result_sha256": sha256_file(path),
    }


def latest_valid_result(
    output_dir: Path,
    dimension: str,
    expected_records: int | tuple[int, int],
) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        output_dir.glob("*_eval_results.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        try:
            return candidate, read_result(
                candidate,
                dimension,
                expected_records=expected_records,
            )
        except Exception as exc:
            print(f"WARNING: invalid reusable result {candidate}: {exc}", flush=True)
    return None


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"mismatched symlink: {link}")
    elif link.exists():
        raise RuntimeError(f"refusing non-symlink path: {link}")
    else:
        link.symlink_to(target)


def prepare_official_input(
    *,
    prompts_path: Path,
    videos_path: Path,
    video_tag: str,
    input_dir: Path,
    full_info_source: Path,
    full_info_output: Path,
) -> dict[str, Any]:
    prompts = read_prompts(prompts_path)
    if any("/" in prompt or "\x00" in prompt for prompt in prompts):
        raise RuntimeError("official prompt cannot be represented as an exact filename")
    for index, prompt in enumerate(prompts):
        source = videos_path / f"{index:05d}-00_{video_tag}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        ensure_symlink(input_dir / f"{prompt}-0.mp4", source)
    prompt_set = set(prompts)
    records = json.loads(full_info_source.read_text(encoding="utf-8"))
    selected = [row for row in records if row.get("prompt_en") in prompt_set]
    selected_unique = {row.get("prompt_en") for row in selected}
    if selected_unique != prompt_set:
        missing = sorted(prompt_set - selected_unique)
        raise RuntimeError(f"official prompts absent from VBench metadata: {missing[:5]}")
    atomic_json(full_info_output, selected)
    return {
        "prompt_count": len(prompts),
        "metadata_record_count": len(selected),
        "prompt_sha256": sha256_file(prompts_path),
        "full_info_sha256": sha256_file(full_info_output),
    }


def prepare_long_clips(
    *,
    input_dir: Path,
    full_info: Path,
    mode: str,
    vbench_root: Path,
) -> None:
    import torch
    from vbench2_beta_long import VBenchLong

    output = input_dir / ".preprocess_metadata"
    output.mkdir(parents=True, exist_ok=True)
    runner = VBenchLong(torch.device("cuda"), str(full_info), str(output))
    runner.preprocess(
        str(input_dir),
        mode,
        use_semantic_splitting=False,
        clip_length_config="clip_length_mix.yaml",
    )


@dataclass
class Task:
    task_id: str
    dimension: str
    output_dir: Path
    log_path: Path
    command: list[str]
    expected_records: int | tuple[int, int]
    extra_environment: dict[str, str]


def run_tasks(
    tasks: list[Task],
    *,
    base_environment: dict[str, str],
    vbench_root: Path,
    gpus: list[int],
    slots_per_gpu: int,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if slots_per_gpu <= 0:
        raise ValueError("slots_per_gpu must be positive")
    reusable: dict[str, dict[str, Any]] = {}
    pending: deque[Task] = deque()
    for task in tasks:
        valid = latest_valid_result(task.output_dir, task.dimension, task.expected_records)
        if valid is None:
            pending.append(task)
            continue
        result_path, result = valid
        reusable[task.task_id] = result
        state["tasks"][task.task_id] = {
            "status": "complete_reused",
            "dimension": task.dimension,
            "result": result,
            "result_path": str(result_path.resolve()),
        }

    allowed_cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    slots = [(gpu, slot) for gpu in gpus for slot in range(slots_per_gpu)]
    cpu_groups = [
        set(allowed_cpus[index :: len(slots)]) or {allowed_cpus[index % len(allowed_cpus)]}
        for index in range(len(slots))
    ]
    free_slots = deque(range(len(slots)))
    running: dict[str, tuple[Task, int, subprocess.Popen[str], Any]] = {}
    failures: list[str] = []
    state["status"] = "running"
    atomic_json(state_path, state)

    while pending or running:
        while pending and free_slots:
            task = pending.popleft()
            slot_index = free_slots.popleft()
            gpu, gpu_slot = slots[slot_index]
            task.output_dir.mkdir(parents=True, exist_ok=True)
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = base_environment.copy()
            environment.update(task.extra_environment)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["MASTER_PORT"] = str(31000 + slot_index)
            handle = task.log_path.open("a", encoding="utf-8")
            handle.write(f"[{now()}] START {json.dumps(task.command)}\n")
            handle.flush()
            affinity = cpu_groups[slot_index]
            process = subprocess.Popen(
                task.command,
                cwd=vbench_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                preexec_fn=(lambda cpus=affinity: os.sched_setaffinity(0, cpus))
                if hasattr(os, "sched_setaffinity")
                else None,
            )
            running[task.task_id] = (task, slot_index, process, handle)
            state["tasks"][task.task_id] = {
                "status": "running",
                "dimension": task.dimension,
                "gpu": gpu,
                "gpu_slot": gpu_slot,
                "cpu_affinity": sorted(affinity),
                "pid": process.pid,
                "started_at_utc": now(),
                "command": task.command,
                "log": str(task.log_path.resolve()),
            }
        atomic_json(state_path, state)
        if not running:
            continue
        time.sleep(5)
        for task_id in list(running):
            task, slot_index, process, handle = running[task_id]
            return_code = process.poll()
            if return_code is None:
                continue
            handle.write(f"[{now()}] EXIT return_code={return_code}\n")
            handle.close()
            del running[task_id]
            free_slots.append(slot_index)
            item = state["tasks"][task_id]
            item.update(
                {
                    "return_code": return_code,
                    "finished_at_utc": now(),
                    "status": "process_complete" if return_code == 0 else "failed",
                }
            )
            if return_code != 0:
                failures.append(task_id)
                continue
            valid = latest_valid_result(task.output_dir, task.dimension, task.expected_records)
            if valid is None:
                item["status"] = "failed_invalid_or_missing_result"
                failures.append(task_id)
                continue
            result_path, result = valid
            reusable[task_id] = result
            item.update(
                {
                    "status": "complete",
                    "result": result,
                    "result_path": str(result_path.resolve()),
                }
            )
        state["updated_at_utc"] = now()
        atomic_json(state_path, state)

    if failures:
        state["status"] = "failed"
        state["failures"] = failures
        atomic_json(state_path, state)
        raise RuntimeError(f"VBench tasks failed: {failures}")
    return reusable


def official_aggregate(scores: dict[str, float]) -> dict[str, Any]:
    normalized = {
        dimension: (scores[dimension] - minimum) / (maximum - minimum)
        for dimension, (minimum, maximum) in DIMENSION_RANGES.items()
    }
    quality_weights = {
        dimension: (0.5 if dimension == "dynamic_degree" else 1.0)
        for dimension in QUALITY_DIMENSIONS
    }
    quality_score = sum(
        normalized[dimension] * quality_weights[dimension] for dimension in QUALITY_DIMENSIONS
    ) / sum(quality_weights.values())
    semantic_score = statistics.fmean(normalized[dimension] for dimension in SEMANTIC_DIMENSIONS)
    return {
        "raw_dimensions": scores,
        "normalized_dimensions": normalized,
        "quality_score": quality_score,
        "semantic_score": semantic_score,
        "total_score": 0.8 * quality_score + 0.2 * semantic_score,
        "formula": "total=0.8*quality+0.2*semantic; dynamic_degree weight=0.5",
    }


def merge_shards(
    *,
    dimension: str,
    shard_results: list[Path],
    output: Path,
) -> dict[str, Any]:
    details: list[Any] = []
    averages: list[Any] = []
    weighted_parts: list[float] = []
    total_weight = 0
    payload_lengths: set[int] = set()
    sources = []
    for result_path in shard_results:
        payload = json.loads(result_path.read_text(encoding="utf-8"))[dimension]
        score = float(payload[0])
        shard_details = payload[1]
        weight = len(shard_details)
        if weight != 600:
            raise RuntimeError(f"invalid 240s shard detail count: {result_path}")
        details.extend(shard_details)
        if len(payload) == 3:
            averages.extend(payload[2])
        payload_lengths.add(len(payload))
        weighted_parts.append(score * weight)
        total_weight += weight
        sources.append(
            {
                "path": str(result_path.resolve()),
                "sha256": sha256_file(result_path),
                "score": score,
                "detail_records": weight,
            }
        )
    if len(payload_lengths) != 1 or total_weight != 14400:
        raise RuntimeError(
            f"incompatible 240s shard payloads for {dimension}: "
            f"lengths={payload_lengths}, records={total_weight}"
        )
    merged_score = math.fsum(weighted_parts) / total_weight
    merged_value: list[Any] = [merged_score, details]
    if next(iter(payload_lengths)) == 3:
        merged_value.append(averages)
    atomic_json(output, {dimension: merged_value})
    validated = read_result(output, dimension, expected_records=14400)
    validated.update(
        {
            "merge_formula": "detail-count-weighted mean of 24 disjoint five-prompt shards",
            "shard_count": len(shard_results),
            "sources": sources,
        }
    )
    return validated


def validate_inputs(args):
    import av

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "prompts/benchmark/manifest.json").read_text()
    )
    cohort = manifest["cohorts"][args.phase]
    if sha256_file(args.prompts) != cohort["sha256"]:
        raise ValueError("Prompt content/order differs from the fixed paper cohort")
    commit = subprocess.check_output(
        ["git", "-C", str(args.vbench_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "45e79ec14e69a2187202c675d2dbce1a71843d53":
        raise ValueError("Use the VBench commit pinned in docs/evaluation.md")
    inventory = []
    for index in range(cohort["count"]):
        video = args.videos / f"{index:05d}-00_{args.video_tag}.mp4"
        if not video.is_file() or video.stat().st_size <= 10000:
            raise ValueError(f"Missing or undersized video: {video}")
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            frames = stream.frames or sum(1 for _ in container.decode(stream))
            if (stream.width, stream.height, frames) != (832, 480, cohort["pixel_frames"]):
                raise ValueError(f"Video geometry/frame count mismatch: {video}")
            if abs(float(stream.average_rate) - 16) > 0.01:
                raise ValueError(f"Video fps mismatch: {video}")
        inventory.append({"file": video.name, "sha256": sha256_file(video)})
    code = {
        str(p.relative_to(args.vbench_root)): sha256_file(p)
        for p in sorted(args.vbench_root.rglob("*.py"))
        if ".git" not in p.parts
    }
    return {
        "phase": args.phase,
        "prompt_sha256": cohort["sha256"],
        "videos": inventory,
        "vbench_commit": commit,
        "vbench_python_sha256": code,
        "driver_sha256": sha256_file(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("5s", "60s", "240s"), required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--video-tag", choices=("ema", "generator"), default="ema")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    for key in (
        "python",
        "vbench_root",
        "model_root",
        "videos",
        "prompts",
        "output_root",
        "work_root",
    ):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    import sys

    sys.path.insert(0, str(args.vbench_root))

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 for g in gpus):
        raise RuntimeError("evaluation requires distinct nonnegative GPU IDs")
    phase_counts = {"5s": 944, "60s": 240, "240s": 120}
    prompts = read_prompts(args.prompts)
    if len(prompts) != phase_counts[args.phase]:
        raise RuntimeError(
            f"{args.phase} prompt count mismatch: {len(prompts)} != {phase_counts[args.phase]}"
        )
    for required in (
        args.python,
        args.vbench_root / "evaluate.py",
        args.vbench_root / "vbench2_beta_long/eval_long.py",
        args.vbench_root / "vbench/VBench_full_info.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    binding = validate_inputs(args)
    for directory in (args.output_root, args.work_root):
        binding_path = directory / "input_binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise ValueError("Evaluation inputs changed; use new output and work directories")
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Unbound evaluation directory exists: {directory}")
    for directory in (args.output_root, args.work_root):
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "input_binding.json", binding)
    state_path = args.output_root / "live_status.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "phase": args.phase,
        "started_at_utc": now(),
        "tasks": {},
        "protocol": {
            "prompt_count": len(prompts),
            "prompt_sha256": sha256_file(args.prompts),
            "samples_per_prompt": 1,
            "vbench_commit": "45e79ec14e69a2187202c675d2dbce1a71843d53",
            "expected_detail_records": EXPECTED_DETAIL_RECORDS[args.phase],
        },
    }
    atomic_json(state_path, state)

    torch_home = args.work_root / "torch_home"
    torch_home.mkdir(parents=True, exist_ok=True)
    ensure_symlink(torch_home / "hub", args.model_root / "torch_hub")
    base_environment = os.environ.copy()
    base_environment.update(
        {
            "PYTHONPATH": str(args.vbench_root),
            "VBENCH_CACHE_DIR": str(args.model_root),
            "TORCH_HOME": str(torch_home),
            "HF_HOME": str(args.model_root / ".cache/huggingface"),
            "TRANSFORMERS_CACHE": str(args.model_root / ".cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )

    full_info_source = args.vbench_root / "vbench/VBench_full_info.json"
    filtered_full_info = args.output_root / "cohort_full_info.json"
    tasks: list[Task] = []
    results: dict[str, dict[str, Any]] = {}
    if args.phase in ("5s", "60s"):
        input_dir = args.work_root / "official_input"
        cohort = prepare_official_input(
            prompts_path=args.prompts,
            videos_path=args.videos,
            video_tag=args.video_tag,
            input_dir=input_dir,
            full_info_source=full_info_source,
            full_info_output=filtered_full_info,
        )
        state["protocol"].update(cohort)
        if args.phase == "60s":
            state["status"] = "splitting_long_videos"
            atomic_json(state_path, state)
            prepare_long_clips(
                input_dir=input_dir,
                full_info=filtered_full_info,
                mode="long_vbench_standard",
                vbench_root=args.vbench_root,
            )
        for index, dimension in enumerate(FULL_DIMENSIONS):
            output_dir = args.output_root / "raw" / dimension
            if args.phase == "5s":
                command = [
                    str(args.python),
                    str(args.vbench_root / "evaluate.py"),
                    "--videos_path",
                    str(input_dir),
                    "--full_json_dir",
                    str(filtered_full_info),
                    "--dimension",
                    dimension,
                    "--mode",
                    "vbench_standard",
                    "--load_ckpt_from_local",
                    "True",
                    "--output_path",
                    str(output_dir),
                ]
            else:
                command = [
                    str(args.python),
                    str(args.vbench_root / "vbench2_beta_long/eval_long.py"),
                    "--videos_path",
                    str(input_dir),
                    "--full_json_dir",
                    str(filtered_full_info),
                    "--dimension",
                    dimension,
                    "--mode",
                    "long_vbench_standard",
                    "--num_of_samples_per_prompt",
                    "1",
                    "--load_ckpt_from_local",
                    "True",
                    "--output_path",
                    str(output_dir),
                ]
            tasks.append(
                Task(
                    task_id=dimension,
                    dimension=dimension,
                    output_dir=output_dir,
                    log_path=args.output_root / "logs" / f"{dimension}.log",
                    command=command,
                    expected_records=EXPECTED_DETAIL_RECORDS[args.phase][dimension],
                    extra_environment={"MASTER_PORT": str(32000 + index)},
                )
            )
        results = run_tasks(
            tasks,
            base_environment=base_environment,
            vbench_root=args.vbench_root,
            gpus=gpus,
            slots_per_gpu=2,
            state_path=state_path,
            state=state,
        )
        scores = {dimension: results[dimension]["score"] for dimension in FULL_DIMENSIONS}
        aggregate = official_aggregate(scores)
    else:
        filtered_full_info.write_bytes(full_info_source.read_bytes())
        shard_count = 24
        prompts_per_shard = 5
        shard_inputs: list[Path] = []
        for shard in range(shard_count):
            input_dir = args.work_root / "shards" / f"shard_{shard:02d}" / "input"
            for local_index in range(prompts_per_shard):
                global_index = shard * prompts_per_shard + local_index
                source = args.videos / f"{global_index:05d}-00_{args.video_tag}.mp4"
                if not source.is_file():
                    raise FileNotFoundError(source)
                ensure_symlink(input_dir / f"case{global_index:04d}-0.mp4", source)
            shard_inputs.append(input_dir)
        state["status"] = "splitting_240s_shards"
        atomic_json(state_path, state)
        for shard, input_dir in enumerate(shard_inputs):
            print(f"Preparing 240s clips: shard {shard + 1}/{shard_count}", flush=True)
            prepare_long_clips(
                input_dir=input_dir,
                full_info=filtered_full_info,
                mode="long_custom_input",
                vbench_root=args.vbench_root,
            )

        def build_240s_tasks(dimensions: tuple[str, ...]) -> list[Task]:
            built = []
            for dimension in dimensions:
                for shard, input_dir in enumerate(shard_inputs):
                    output_dir = args.output_root / "raw_shards" / dimension / f"shard_{shard:02d}"
                    task_id = f"{dimension}.shard_{shard:02d}"
                    command = [
                        str(args.python),
                        str(args.vbench_root / "vbench2_beta_long/eval_long.py"),
                        "--videos_path",
                        str(input_dir),
                        "--full_json_dir",
                        str(filtered_full_info),
                        "--dimension",
                        dimension,
                        "--mode",
                        "long_custom_input",
                        "--load_ckpt_from_local",
                        "True",
                        "--output_path",
                        str(output_dir),
                    ]
                    built.append(
                        Task(
                            task_id=task_id,
                            dimension=dimension,
                            output_dir=output_dir,
                            log_path=args.output_root / "logs" / f"{task_id}.log",
                            command=command,
                            expected_records=600,
                            extra_environment={},
                        )
                    )
            return built

        shard_task_results = {}
        shard_task_results.update(
            run_tasks(
                build_240s_tasks(LIGHT_240S_DIMENSIONS),
                base_environment=base_environment,
                vbench_root=args.vbench_root,
                gpus=gpus,
                slots_per_gpu=2,
                state_path=state_path,
                state=state,
            )
        )
        shard_task_results.update(
            run_tasks(
                build_240s_tasks(HEAVY_240S_DIMENSIONS),
                base_environment=base_environment,
                vbench_root=args.vbench_root,
                gpus=gpus,
                slots_per_gpu=1,
                state_path=state_path,
                state=state,
            )
        )
        for dimension in CUSTOM_DIMENSIONS:
            source_paths = [
                Path(shard_task_results[f"{dimension}.shard_{shard:02d}"]["result_path"])
                for shard in range(shard_count)
            ]
            result = merge_shards(
                dimension=dimension,
                shard_results=source_paths,
                output=args.output_root / "raw" / dimension / "merged_eval_results.json",
            )
            results[dimension] = result
        scores = {dimension: results[dimension]["score"] for dimension in CUSTOM_DIMENSIONS}
        quality_dimensions = tuple(
            dimension for dimension in CUSTOM_DIMENSIONS if dimension != "dynamic_degree"
        )
        aggregate = {
            "raw_dimensions": scores,
            "five_dimension_quality_mean": statistics.fmean(
                scores[dimension] for dimension in quality_dimensions
            ),
            "five_dimension_quality_dimensions": list(quality_dimensions),
            "dynamic_degree": scores["dynamic_degree"],
            "unweighted_six_dimension_mean": statistics.fmean(scores.values()),
            "not_official_vbench_leaderboard_metric": True,
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "phase": args.phase,
        "completed_at_utc": now(),
        "protocol": state["protocol"],
        "dimensions": results,
        "aggregate": aggregate,
        "note": (
            "5s/60s use one sample per prompt and the official 16-dimension formulas; "
            "240s is the project-local six-dimension protocol."
        ),
    }
    atomic_json(args.output_root / "summary.json", summary)
    state.update({"status": "complete", "finished_at_utc": now()})
    atomic_json(state_path, state)
    (args.output_root / "_SUCCESS_PHASE").write_text(
        f"completed_at_utc={now()}\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
