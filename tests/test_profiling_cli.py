"""Phase 3: offline profiling CLI tests."""

from __future__ import annotations

import json

from inferpilot import WorkloadProfile
from inferpilot.profiling import main


def _obs(t, p=128, o=32):
    return {"arrival_offset_s": t, "prompt_tokens": p, "output_tokens": o, "success": True}


def _write_json(path, rows):
    path.write_text(json.dumps(rows))


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_json_input_produces_valid_profile(tmp_path) -> None:
    src = tmp_path / "obs.json"
    _write_json(src, [_obs(i * 0.5) for i in range(5)])
    out = tmp_path / "profile.json"
    assert main([str(src), str(out)]) == 0
    prof = WorkloadProfile.model_validate_json(out.read_text())
    assert prof.observation_count == 5 and prof.realized_request_rate_qps == 2.0


def test_jsonl_input_and_byte_stability(tmp_path) -> None:
    src = tmp_path / "obs.jsonl"
    _write_jsonl(src, [_obs(i * 0.5, p=100 + i) for i in range(6)])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert main([str(src), str(a)]) == 0
    assert main([str(src), str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()  # identical ordered observations -> identical bytes


def test_refuses_to_overwrite(tmp_path) -> None:
    src = tmp_path / "obs.json"
    _write_json(src, [_obs(0.0), _obs(0.5)])
    out = tmp_path / "profile.json"
    out.write_text("SENTINEL")
    assert main([str(src), str(out)]) == 2
    assert out.read_text() == "SENTINEL"  # untouched


def test_empty_and_insufficient_fail(tmp_path) -> None:
    empty = tmp_path / "empty.json"
    _write_json(empty, [])
    assert main([str(empty), str(tmp_path / "e.json")]) == 2

    one = tmp_path / "one.json"
    _write_json(one, [_obs(0.0)])
    assert main([str(one), str(tmp_path / "o.json")]) == 2


def test_malformed_and_unordered_fail(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json]")
    assert main([str(bad), str(tmp_path / "b1.json")]) == 2

    negative = tmp_path / "neg.json"
    _write_json(negative, [{"arrival_offset_s": 0.0, "prompt_tokens": -1, "output_tokens": 1, "success": True},
                           _obs(0.5)])
    assert main([str(negative), str(tmp_path / "b2.json")]) == 2

    unordered = tmp_path / "unordered.json"
    _write_json(unordered, [_obs(1.0), _obs(0.5)])
    assert main([str(unordered), str(tmp_path / "b3.json")]) == 2
