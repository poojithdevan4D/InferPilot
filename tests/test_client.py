"""Integration tests for the streaming client against a fake HTTP server.

No GPU, model, or vLLM required — a stdlib fake server streams SSE.
"""

from __future__ import annotations

import asyncio
from time import monotonic

from fake_vllm_server import serve_in_thread

from inferpilot.runner.client import GenerationParams, run_requests


def _run(base_url: str, prompts, params, concurrency=1):
    return asyncio.run(
        run_requests(base_url, prompts, params, t0=monotonic(), max_concurrency=concurrency)
    )


def _params(**kw) -> GenerationParams:
    base = dict(model="fake", max_tokens=8, temperature=0.0, seed=0, ignore_eos=True)
    base.update(kw)
    return GenerationParams(**base)


def test_normal_stream_produces_valid_measurements() -> None:
    with serve_in_thread(mode="normal", output_tokens=8, prompt_tokens=128) as url:
        results = _run(url, ["p1", "p2"], _params())
    assert len(results) == 2
    for m in results:
        assert m.success is True
        assert m.output_tokens == 8
        assert m.prompt_tokens == 128
        assert m.ttft_ms is not None and m.ttft_ms >= 0
        assert m.e2e_latency_ms is not None
        assert m.tpot_ms is not None  # >= 2 tokens
        # TPOT must equal the agreed formula on the recorded values.
        expected = (m.e2e_latency_ms - m.ttft_ms) / (m.output_tokens - 1)
        assert abs(m.tpot_ms - expected) < 1e-9


def test_one_token_output_has_no_tpot() -> None:
    with serve_in_thread(mode="normal", output_tokens=1) as url:
        (m,) = _run(url, ["p"], _params())
    assert m.success is True
    assert m.output_tokens == 1
    assert m.tpot_ms is None


def test_empty_output_is_failed() -> None:
    with serve_in_thread(mode="empty") as url:
        (m,) = _run(url, ["p"], _params())
    assert m.success is False
    assert "empty_output" in (m.error or "")


def test_malformed_stream_is_failed() -> None:
    with serve_in_thread(mode="malformed") as url:
        (m,) = _run(url, ["p"], _params())
    assert m.success is False
    assert "malformed_stream" in (m.error or "")


def test_http_error_is_failed() -> None:
    with serve_in_thread(mode="error500") as url:
        (m,) = _run(url, ["p"], _params())
    assert m.success is False
    assert "HTTP 500" in (m.error or "")


def test_timeout_is_failed() -> None:
    with serve_in_thread(mode="hang") as url:
        (m,) = _run(url, ["p"], _params(request_timeout_s=0.3))
    assert m.success is False
    assert "timeout" in (m.error or "")


def test_measurements_share_monotonic_timeline() -> None:
    with serve_in_thread(mode="normal", output_tokens=4) as url:
        results = _run(url, ["a", "b", "c"], _params(), concurrency=1)
    # closed-loop c=1: requests do not overlap; each start >= previous end.
    for prev, cur in zip(results, results[1:]):
        assert cur.start_time_s >= prev.end_time_s - 1e-6
