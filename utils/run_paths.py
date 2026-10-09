"""Resolve AGF run directories and user-selected output paths."""

from __future__ import annotations

import os
import time
from pathlib import Path

_DEFAULT_RUN_ID = time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def resolve_run_output(
    project: str,
    value: str | os.PathLike[str] | None,
    default_relative: str | os.PathLike[str],
) -> Path:
    """Use a run-relative path by default; honor explicit absolute paths."""
    runs_root = Path(os.environ.get("AGF_RUNS_ROOT", Path.cwd() / "runs")).expanduser().resolve()
    run_id = os.environ.get("AGF_RUN_ID", _DEFAULT_RUN_ID)
    if not run_id or run_id in {".", ".."} or any(c in run_id for c in ("/", "\\", "\0")):
        raise ValueError("AGF_RUN_ID must be a nonempty directory name without path separators")
    run_dir = runs_root / project / run_id
    candidate = Path(value).expanduser() if value is not None else Path(default_relative)
    return (candidate if candidate.is_absolute() else run_dir / candidate).resolve()
