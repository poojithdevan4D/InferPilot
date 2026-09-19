"""Slow zero-GPU proof that the advertised instrumentation dry run really executes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from inferpilot import ExperimentResult
from inferpilot.runner.artifacts import METRICS_CAPABILITIES_FILENAME, RESULT_FILENAME
from inferpilot.runner.metrics_capabilities import MetricsCapabilityReport


def test_full_instrumentation_dry_run(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "instrumentation_dry_run.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads(completed.stdout)
    assert report["pipeline"] == "PASS"
    assert report["real_pilot_gate"] == "NO_GO_EXPECTED"
    assert report["load_state"] == "overloaded"
    assert report["diagnosis_regime"] == "kv_pressure"
    assert report["metrics_ready"] is False

    run_dir = Path(report["run_dir"])
    result = ExperimentResult.model_validate_json((run_dir / RESULT_FILENAME).read_text())
    capability = MetricsCapabilityReport.model_validate_json(
        (run_dir / METRICS_CAPABILITIES_FILENAME).read_text()
    )
    assert result.load_evidence is not None
    assert capability.ready is False
