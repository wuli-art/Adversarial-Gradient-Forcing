#!/usr/bin/env python3
"""Apply compatibility fixes to the pinned VBench checkout.

Patches are idempotent. The manifest records each changed file and its
before/after SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid


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


def replace_once(text: str, old: str, new: str, *, label: str) -> tuple[str, str]:
    if new and new in text:
        return text, "already_applied"
    count = text.count(old)
    if not new and count == 0:
        return text, "already_applied"
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1), "applied"


def patch_file(path: Path, transforms: list[tuple[str, str, str]]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before_sha256 = sha256_file(path)
    text = path.read_text(encoding="utf-8")
    statuses: list[dict[str, str]] = []
    for label, old, new in transforms:
        text, status = replace_once(text, old, new, label=f"{path}: {label}")
        statuses.append({"name": label, "status": status})
    path.write_text(text, encoding="utf-8")
    after_sha256 = sha256_file(path)
    return {
        "path": str(path),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "operations": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.vbench_root.resolve()
    files: list[dict[str, Any]] = []
    for relative in (
        "vbench/third_party/umt/datasets/kinetics.py",
        "vbench/third_party/umt/datasets/kinetics_sparse.py",
    ):
        files.append(
            patch_file(
                root / relative,
                [
                    (
                        "remove_numpy2_incompatible_disp_import",
                        "from numpy.lib.function_base import disp\n",
                        "",
                    )
                ],
            )
        )

    files.append(
        patch_file(
            root / "vbench/third_party/umt/models/modeling_finetune.py",
            [
                (
                    "discard_unsupported_cache_dir",
                    "    kwargs.pop('pretrained_cfg_overlay', None) # added by Ziqi to accommodate timm=0.9.12\n",
                    "    kwargs.pop('pretrained_cfg_overlay', None) # added by Ziqi to accommodate timm=0.9.12\n"
                    "    kwargs.pop('cache_dir', None)  # compatibility with current timm factories\n",
                )
            ],
        )
    )

    files.append(
        patch_file(
            root / "vbench/third_party/tag2Text/tag2text.py",
            [
                (
                    "use_pinned_local_bert_for_tag2text_model",
                    "self.vision_multi = BertModel.from_pretrained('bert-base-uncased',config=q2l_config, add_pooling_layer=False)",
                    "self.vision_multi = BertModel.from_pretrained(os.environ['VBENCH_BERT_BASE_UNCASED_DIR'],config=q2l_config, add_pooling_layer=False)",
                ),
                (
                    "use_pinned_local_bert_for_tag2text_tokenizer",
                    "    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')\n",
                    "    tokenizer = BertTokenizer.from_pretrained(os.environ['VBENCH_BERT_BASE_UNCASED_DIR'])\n",
                ),
                (
                    "resolve_enc_token_by_name",
                    "    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  \n",
                    "    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids('[ENC]')\n",
                )
            ],
        )
    )

    files.append(
        patch_file(
            root / "vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py",
            [
                (
                    "import_os_for_local_bert_path",
                    "import math\nimport torch\n",
                    "import math\nimport os\nimport torch\n",
                ),
                (
                    "use_pinned_local_bert_for_grit_tokenizer",
                    "        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)\n",
                    "        tokenizer = BertTokenizer.from_pretrained(os.environ['VBENCH_BERT_BASE_UNCASED_DIR'], do_lower_case=True)\n",
                ),
            ],
        )
    )

    files.append(
        patch_file(
            root / "vbench2_beta_long/subject_consistency.py",
            [
                (
                    "preserve_inclip_detail_when_clip2clip_disabled",
                    "    inclip_all_results = all_results\n\n\n    # compute clip2clip scores\n",
                    "    inclip_all_results = all_results\n"
                    "    if float(kwargs.get('w_clip2clip', 0.0)) == 0.0:\n"
                    "        return inclip_all_results, inclip_detailed_results\n\n"
                    "    # compute clip2clip scores\n",
                )
            ],
        )
    )
    files.append(
        patch_file(
            root / "vbench2_beta_long/background_consistency.py",
            [
                (
                    "preserve_inclip_detail_when_clip2clip_disabled",
                    "    inclip_all_results, inclip_detailed_results, inclip_average_scores = reorganize_clips_results(detailed_results)\n\n    # compute clip2clip scores\n",
                    "    inclip_all_results, inclip_detailed_results, inclip_average_scores = reorganize_clips_results(detailed_results)\n"
                    "    if float(kwargs.get('w_clip2clip', 0.0)) == 0.0:\n"
                    "        return inclip_all_results, inclip_detailed_results\n\n"
                    "    # compute clip2clip scores\n",
                )
            ],
        )
    )

    files.append(
        patch_file(
            root / "vbench2_beta_long/__init__.py",
            [
                (
                    "infer_video_suffix_from_videos_only",
                    "            video_names = os.listdir(videos_path)\n"
                    "            postfix = Path(video_names[0]).suffix\n"
                    "            video_clip_folder_names = [name.replace(postfix, '') for name in video_names]\n",
                    "            video_names = [name for name in os.listdir(videos_path)\n"
                    "                           if Path(name).suffix.lower() in ('.mp4', '.avi', '.mov')]\n"
                    "            postfix = Path(video_names[0]).suffix\n"
                    "            video_clip_folder_names = [Path(name).stem for name in video_names]\n",
                )
            ],
        )
    )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "vbench_root": str(root),
        "upstream_commit": "45e79ec14e69a2187202c675d2dbce1a71843d53",
        "files": files,
        "note": (
            "Tag2Text and GRiT resolve BERT from the explicit local BERT asset. "
            "Tag2Text uses Transformers 4.33.2 for compatibility. "
            "Long-video subject/"
            "background evaluation retains per-clip details when the official non-dev "
            "policy fixes w_clip2clip=0."
        ),
    }
    atomic_json(args.manifest, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
