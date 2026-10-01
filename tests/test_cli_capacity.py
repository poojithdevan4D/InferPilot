"""The `inferpilot capacity` command: load a rate sweep, print the ceiling (and a
lever's measured impact), end to end."""

from __future__ import annotations

from pathlib import Path

from inferpilot.cli import main

from test_load_state import make_result


def _write(path: Path, **kw) -> Path:
    path.write_text(make_result(**kw).model_dump_json())
    return path


def test_capacity_prints_bracketed_ceiling(tmp_path, capsys) -> None:
    f1 = _write(tmp_path / "r4.json", ttft=200.0, tpot=40.0, rate=4.0)
    f2 = _write(tmp_path / "r6.json", ttft=800.0, tpot=40.0, rate=6.0)
    rc = main(["capacity", str(f1), str(f2), "--ttft-p95-ms", "500"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SLO-capacity frontier" in out
    assert "Capacity ceiling" in out


def test_capacity_requires_an_slo(tmp_path, capsys) -> None:
    f1 = _write(tmp_path / "a.json", ttft=200.0, tpot=40.0, rate=4.0)
    f2 = _write(tmp_path / "b.json", ttft=800.0, tpot=40.0, rate=6.0)
    rc = main(["capacity", str(f1), str(f2)])
    assert rc == 2
    assert "SLO threshold" in capsys.readouterr().err


def test_capacity_with_candidate_reports_lever_impact(tmp_path, capsys) -> None:
    base1 = _write(tmp_path / "b4.json", ttft=200.0, tpot=40.0, rate=4.0)
    base2 = _write(tmp_path / "b6.json", ttft=800.0, tpot=40.0, rate=6.0)
    # candidate holds latency under the SLO to a higher rate -> higher ceiling.
    cand1 = _write(tmp_path / "c6.json", ttft=300.0, tpot=38.0, rate=6.0)
    cand2 = _write(tmp_path / "c8.json", ttft=900.0, tpot=38.0, rate=8.0)
    rc = main([
        "capacity", str(base1), str(base2),
        "--ttft-p95-ms", "500",
        "--candidate", str(cand1), str(cand2),
        "--lever", "kv_cache_dtype=fp8",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Measured impact" in out
    assert "kv_cache_dtype=fp8" in out
