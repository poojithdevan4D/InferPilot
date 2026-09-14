"""Open-loop poisson-v1 arrival schedule + dispatch semantics."""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest
from fake_vllm_server import serve_in_thread

from inferpilot.runner.client import GenerationParams, run_requests_open_loop
from inferpilot.runner.schedule import POISSON_VERSION, generate_poisson_offsets


# --- schedule generation (deterministic) ------------------------------------ #


def test_identical_seeds_produce_identical_schedules() -> None:
    a = generate_poisson_offsets(25, 5.0, seed=123)
    b = generate_poisson_offsets(25, 5.0, seed=123)
    assert a == b
    assert a[0] == 0.0
    assert all(y >= x for x, y in zip(a, a[1:]))  # cumulative => non-decreasing


def test_different_seeds_produce_different_schedules() -> None:
    a = generate_poisson_offsets(25, 5.0, seed=1)
    b = generate_poisson_offsets(25, 5.0, seed=2)
    assert a != b
    assert a[0] == b[0] == 0.0  # first arrival always at offset 0


def test_version_and_edge_cases() -> None:
    assert POISSON_VERSION == "poisson-v1"
    assert generate_poisson_offsets(0, 5.0, 1) == []
    assert generate_poisson_offsets(1, 5.0, 1) == [0.0]
    with pytest.raises(ValueError):
        generate_poisson_offsets(3, 0.0, 1)


# --- dispatch semantics ----------------------------------------------------- #


def _params(**kw) -> GenerationParams:
    base = dict(model="fake", max_tokens=4, temperature=0.0, seed=0, ignore_eos=True)
    base.update(kw)
    return GenerationParams(**base)


def _run(url, prompts, offsets, **pkw):
    return asyncio.run(
        run_requests_open_loop(
            url, prompts, _params(**pkw), t0=monotonic(), offsets=offsets
        )
    )


def test_actual_dispatch_matches_generated_schedule() -> None:
    offsets = [0.0, 0.1, 0.2]
    with serve_in_thread(mode="normal", output_tokens=4) as url:
        results, dispatch = _run(url, ["a", "b", "c"], offsets)
    assert all(m.success for m in results)
    for scheduled, actual in zip(offsets, dispatch):
        assert abs(actual - scheduled) < 0.08  # tolerant timing bound


def test_arrivals_do_not_wait_for_earlier_completions() -> None:
    # Each response takes ~0.3s, but inter-arrivals are 0.05s. Closed-loop c=1
    # would push the 3rd dispatch to ~0.6s; open-loop must keep it near 0.1s.
    offsets = [0.0, 0.05, 0.1]
    with serve_in_thread(mode="normal", output_tokens=4, response_delay=0.3) as url:
        results, dispatch = _run(url, ["a", "b", "c"], offsets, request_timeout_s=5.0)
    assert all(m.success for m in results)
    assert dispatch[2] < 0.2  # dispatched before the first request completed


@pytest.mark.parametrize(
    "offsets, message",
    [
        ([0.0], "one entry per prompt"),
        ([0.0, -0.1], "non-negative"),
        ([0.1, 0.0], "monotonically non-decreasing"),
    ],
)
def test_open_loop_refuses_invalid_schedules(offsets, message) -> None:
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            run_requests_open_loop(
                "http://127.0.0.1:1",
                ["a", "b"],
                _params(),
                t0=monotonic(),
                offsets=offsets,
            )
        )
