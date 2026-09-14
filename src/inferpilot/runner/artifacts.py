"""Artifact storage: unique run directory + immutable result JSON.

Storage is deliberately dumb and file-based (no database). Each run gets a fresh
unique directory. The result JSON is written once and then made read-only so a
stored result is immutable; server logs live alongside it in the same directory.
"""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

from ..results import ExperimentResult

RESULT_FILENAME = "result.json"
SERVER_STDOUT_FILENAME = "server.stdout.log"
SERVER_STDERR_FILENAME = "server.stderr.log"


def create_run_dir(base_dir: str | os.PathLike[str], experiment_id: str) -> Path:
    """Create and return a unique run directory under ``base_dir``.

    The uniqueness suffix is a random uuid (not a wall-clock timestamp), so this
    never participates in any duration calculation.
    """
    base = Path(base_dir)
    suffix = uuid.uuid4().hex[:8]
    run_dir = base / f"{experiment_id}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def result_path(run_dir: Path) -> Path:
    return run_dir / RESULT_FILENAME


def server_log_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / SERVER_STDOUT_FILENAME, run_dir / SERVER_STDERR_FILENAME


def write_result(run_dir: Path, result: ExperimentResult) -> Path:
    """Write the result JSON exactly once and make it read-only (immutable)."""
    path = result_path(run_dir)
    if path.exists():
        raise FileExistsError(f"result already written at {path}; results are immutable")
    path.write_text(result.model_dump_json(indent=2))
    # Make the file read-only for all — a stored result must not be mutated.
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path
