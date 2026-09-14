"""Content-addressed result-store ingestion, deduplication, and integrity."""

from __future__ import annotations

import json

import pytest

from inferpilot.comparison import ResultStore
from test_comparison import _result


def _write_run(path, result) -> None:
    path.mkdir()
    (path / "result.json").write_text(result.model_dump_json(indent=2))
    (path / "server.stdout.log").write_text("server evidence")


def test_ingest_copies_bundle_and_deduplicates(tmp_path) -> None:
    source = tmp_path / "source"
    _write_run(source, _result("base", 0))
    store = ResultStore(tmp_path / "catalog")

    first = store.ingest(source)
    second = store.ingest(source)

    assert first.created is True
    assert second.created is False
    assert first.run_id == second.run_id
    assert (first.path / "server.stdout.log").read_text() == "server evidence"
    assert store.load(first.run_id).is_baseline_eligible is True


def test_ingest_refuses_ineligible_result(tmp_path) -> None:
    result = _result("bad", 0)
    result.telemetry = None
    source = tmp_path / "source"
    _write_run(source, result)
    store = ResultStore(tmp_path / "catalog")

    with pytest.raises(ValueError, match="not baseline-eligible"):
        store.ingest(source)


def test_load_detects_result_tampering(tmp_path) -> None:
    source = tmp_path / "source"
    _write_run(source, _result("base", 0))
    store = ResultStore(tmp_path / "catalog")
    ingested = store.ingest(source)

    stored_result = ingested.path / "result.json"
    stored_result.chmod(0o644)
    payload = json.loads(stored_result.read_text())
    payload["config"]["name"] = "tampered"
    stored_result.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="bundle integrity"):
        store.load(ingested.run_id)


def test_load_detects_log_tampering(tmp_path) -> None:
    source = tmp_path / "source"
    _write_run(source, _result("base", 0))
    store = ResultStore(tmp_path / "catalog")
    ingested = store.ingest(source)

    (ingested.path / "server.stdout.log").write_text("changed evidence")
    with pytest.raises(ValueError, match="bundle integrity"):
        store.load(ingested.run_id)
