"""Artifact storage: unique run directory + immutable result JSON.

Storage is deliberately dumb and file-based (no database). Each run gets a fresh
unique directory. The result JSON is written once and then made read-only so a
stored result is immutable; server logs live alongside it in the same directory.
"""

from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path
from typing import Sequence

from ..measurements import RequestMeasurement
from ..results import ExperimentResult
from ..telemetry import ResourceSample
from .metrics_capabilities import MetricsCapabilityReport

RESULT_FILENAME = "result.json"
WARMUP_FILENAME = "warmup.json"
TELEMETRY_FILENAME = "telemetry.json"
LIFECYCLE_FILENAME = "lifecycle.json"
ARRIVALS_FILENAME = "arrivals.json"
PHASES_FILENAME = "phases.json"
METRICS_CAPABILITIES_FILENAME = "metrics-capabilities.json"
SERVER_STDOUT_FILENAME = "server.stdout.log"
SERVER_STDERR_FILENAME = "server.stderr.log"


def _write_immutable_json(path: Path, payload) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} already exists; artifacts are immutable")
    path.write_text(json.dumps(payload, indent=2))
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path


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


def write_warmup(run_dir: Path, measurements: Sequence[RequestMeasurement]) -> Path:
    """Persist warm-up measurements as a separate diagnostics artifact.

    Kept out of the measured result on purpose: warm-ups must never enter the
    measured aggregates, but their timings/errors are useful for debugging.
    """
    path = run_dir / WARMUP_FILENAME
    payload = [m.model_dump(mode="json") for m in measurements]
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_telemetry(run_dir: Path, samples: Sequence[ResourceSample]) -> Path:
    """Persist raw telemetry samples as an immutable artifact."""
    payload = [s.model_dump(mode="json") for s in samples]
    return _write_immutable_json(run_dir / TELEMETRY_FILENAME, payload)


def write_lifecycle(run_dir: Path, lifecycle: dict) -> Path:
    """Persist the teardown-vs-startup/measurement error classification."""
    return _write_immutable_json(run_dir / LIFECYCLE_FILENAME, lifecycle)


def write_arrivals(run_dir: Path, arrivals: dict) -> Path:
    """Persist the open-loop generated + actual dispatch offsets (immutable)."""
    return _write_immutable_json(run_dir / ARRIVALS_FILENAME, arrivals)


def write_phases(run_dir: Path, timing: "RunnerPhaseTiming") -> Path:
    """Persist the monotonic runner phase-timing artifact (immutable)."""
    return _write_immutable_json(run_dir / PHASES_FILENAME, timing.model_dump(mode="json"))


def write_metrics_capabilities(
    run_dir: Path, report: MetricsCapabilityReport
) -> Path:
    """Persist the metric-surface preflight captured from the running server."""
    return _write_immutable_json(
        run_dir / METRICS_CAPABILITIES_FILENAME,
        report.model_dump(mode="json"),
    )


def write_result(run_dir: Path, result: ExperimentResult) -> Path:
    """Write the result JSON exactly once and make it read-only (immutable)."""
    path = result_path(run_dir)
    if path.exists():
        raise FileExistsError(f"result already written at {path}; results are immutable")
    path.write_text(result.model_dump_json(indent=2))
    # Make the file read-only for all — a stored result must not be mutated.
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path
