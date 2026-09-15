"""Pure contract tests for the sealed M2D held-out execution driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "experiments/m2d-heldout/run_study.py"
SPEC = importlib.util.spec_from_file_location("m2d_heldout_driver", PATH)
assert SPEC and SPEC.loader
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def test_execution_order_exactly_matches_preregistration() -> None:
    order = DRIVER.execution_order()
    assert len(order) == 36
    assert len(set(order)) == 36
    assert order[:6] == (
        ("prefill", 50, 1, 512),
        ("decode", 50, 1, 512),
        ("burst", 50, 1, 512),
        ("prefill", 50, 1, 2048),
        ("decode", 50, 1, 2048),
        ("burst", 50, 1, 2048),
    )
    assert order[-1] == ("burst", 52, 4, 2048)


def test_retry_is_only_one_identical_drift_retry() -> None:
    assert DRIVER.retry_allowed(["dispatch_drift_exceeded"], 1)
    assert not DRIVER.retry_allowed(["dispatch_drift_exceeded"], 2)
    assert not DRIVER.retry_allowed(["telemetry_incomplete"], 1)
    assert not DRIVER.retry_allowed(["dispatch_drift_exceeded", "phases_incomplete"], 1)


def test_resume_state_marks_terminal_shape_without_affecting_others() -> None:
    records = [{
        "experiment_id": "m2d-held-prefill-a50-seq1-tok512",
        "shape": "prefill",
        "attempt": 1,
        "accepted": False,
        "terminal": True,
    }]
    accepted, invalid, attempts = DRIVER._resume_state(records)
    assert accepted == set()
    assert invalid == {"prefill"}
    assert attempts == {"m2d-held-prefill-a50-seq1-tok512": 1}


def _record(experiment_id: str, shape: str, attempt: int, *, accepted: bool, problems: list[str]):
    retry = DRIVER.retry_allowed(problems, attempt)
    return {
        "experiment_id": experiment_id,
        "shape": shape,
        "attempt": attempt,
        "accepted": accepted,
        "terminal": not accepted and not retry,
        "retry_allowed": retry,
        "problems": problems,
    }


def test_manifest_order_allows_only_immediate_retry() -> None:
    first = "m2d-held-prefill-a50-seq1-tok512"
    second = "m2d-held-decode-a50-seq1-tok512"
    DRIVER._validate_manifest_order([
        _record(first, "prefill", 1, accepted=False, problems=["dispatch_drift_exceeded"]),
        _record(first, "prefill", 2, accepted=True, problems=[]),
        _record(second, "decode", 1, accepted=True, problems=[]),
    ])


def test_manifest_order_rejects_skipped_cell() -> None:
    second = "m2d-held-decode-a50-seq1-tok512"
    try:
        DRIVER._validate_manifest_order([
            _record(second, "decode", 1, accepted=True, problems=[]),
        ])
    except ValueError as exc:
        assert "violates sealed order" in str(exc)
    else:
        raise AssertionError("out-of-order manifest accepted")


def test_terminal_shape_is_skipped_but_other_shapes_keep_order() -> None:
    first = "m2d-held-prefill-a50-seq1-tok512"
    second = "m2d-held-decode-a50-seq1-tok512"
    DRIVER._validate_manifest_order([
        _record(first, "prefill", 1, accepted=False, problems=["telemetry_incomplete"]),
        _record(second, "decode", 1, accepted=True, problems=[]),
    ])
