"""Read-only screening from vLLM /metrics: separate the regimes an engineer cares
about from one or two Prometheus snapshots, and abstain when a signal is missing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot.cli import main
from inferpilot.metrics_snapshot import LiveReading, VLLMMetricsSnapshot, inspect_metrics


def _metrics(*, kv=None, waiting=None, running=None, preemptions=None, recompute=None) -> str:
    lines = []
    if kv is not None:
        lines.append(f'vllm:kv_cache_usage_perc{{model_name="q"}} {kv}')
    if running is not None:
        lines.append(f'vllm:num_requests_running{{model_name="q"}} {running}')
    if waiting is not None:
        lines.append(f'vllm:num_requests_waiting{{model_name="q"}} {waiting}')
    if preemptions is not None:
        lines.append(f'vllm:num_preemptions_total{{model_name="q"}} {preemptions}')
    if recompute is not None:
        lines.append(f'vllm:recomputed_token_executions_total{{model_name="q"}} {recompute}')
    return "# a vLLM metrics scrape\n" + "\n".join(lines) + "\n"


def test_parse_reads_named_series() -> None:
    snap = VLLMMetricsSnapshot.parse(_metrics(kv=0.97, waiting=5, running=8, preemptions=12))
    assert snap.kv_cache_usage_perc == 0.97
    assert snap.num_requests_waiting == 5.0
    assert snap.num_preemptions_total == 12.0


def test_kv_full_and_preempting_recommends_fp8() -> None:
    before = _metrics(kv=0.98, waiting=5, preemptions=10)
    after = _metrics(kv=0.98, waiting=6, preemptions=15)   # preemptions rose
    r = inspect_metrics(before, after)
    assert r.verdict == "kv_capacity_bound_preempting"
    assert r.recommended_lever == "kv_cache_dtype=fp8"


def test_kv_full_not_preempting_is_near_capacity() -> None:
    before = _metrics(kv=0.98, waiting=2, preemptions=10)
    after = _metrics(kv=0.98, waiting=2, preemptions=10)   # flat
    r = inspect_metrics(before, after)
    assert r.verdict == "near_capacity"
    assert r.recommended_lever == "none"


def test_queue_with_kv_headroom_is_not_kv_bound() -> None:
    r = inspect_metrics(_metrics(kv=0.55, waiting=7, preemptions=0))
    assert r.verdict == "not_kv_bound"
    assert "fp8 KV will not help" in r.next_step


def test_headroom_and_no_queue_is_healthy() -> None:
    r = inspect_metrics(_metrics(kv=0.3, waiting=0))
    assert r.verdict == "healthy_or_underutilized"


def test_single_snapshot_at_full_kv_needs_second() -> None:
    r = inspect_metrics(_metrics(kv=0.97, waiting=4, preemptions=10))
    assert r.verdict == "need_second_snapshot"


def test_missing_kv_is_insufficient() -> None:
    r = inspect_metrics(_metrics(waiting=3))
    assert r.verdict == "insufficient_metrics"
    assert "kv_cache_usage_perc" in r.next_step


def test_reading_roundtrips_and_rejects_tampering() -> None:
    r = inspect_metrics(_metrics(kv=0.98, waiting=5, preemptions=10),
                        _metrics(kv=0.98, waiting=6, preemptions=15))
    assert LiveReading.model_validate_json(r.model_dump_json()) == r
    raw = r.model_dump(mode="json")
    raw["recommended_lever"] = "none"
    with pytest.raises(ValidationError, match="inconsistent with its metrics snapshots"):
        LiveReading.model_validate(raw)


def test_cli_inspect_prints_screening(tmp_path, capsys) -> None:
    b = tmp_path / "before.txt"; b.write_text(_metrics(kv=0.98, waiting=5, preemptions=10))
    a = tmp_path / "after.txt"; a.write_text(_metrics(kv=0.98, waiting=6, preemptions=15))
    rc = main(["inspect", str(b), str(a)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "KV-BOUND & PREEMPTING" in out
    assert "kv_cache_dtype=fp8" in out


def test_cli_inspect_with_no_source_is_friendly(capsys) -> None:
    rc = main(["inspect"])
    assert rc == 2
    assert "--url" in capsys.readouterr().err


def test_render_live_line_is_compact() -> None:
    from inferpilot.report import render_live_line
    r = inspect_metrics(_metrics(kv=0.98, waiting=6, preemptions=10),
                        _metrics(kv=0.98, waiting=6, preemptions=19))
    line = render_live_line(r, stamp="14:32:07")
    assert "14:32:07" in line and "kv-bound" in line and "KV" in line and "q6" in line
    assert "preempt +9" in line
    assert "\n" not in line  # one line


def test_cli_doctor_alias_works(tmp_path, capsys) -> None:
    b = tmp_path / "b.txt"; b.write_text(_metrics(kv=0.98, waiting=5, preemptions=10))
    a = tmp_path / "a.txt"; a.write_text(_metrics(kv=0.98, waiting=6, preemptions=15))
    rc = main(["doctor", str(b), str(a)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "KV-BOUND & PREEMPTING" in out


def test_cli_inspect_json(tmp_path, capsys) -> None:
    import json
    b = tmp_path / "b.txt"; b.write_text(_metrics(kv=0.3, waiting=0))
    rc = main(["inspect", str(b), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "healthy_or_underutilized"
