"""Zero-GPU proofs for the frozen paid fp8 pilot protocol."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/fp8-instrumentation-pilot/pilot_protocol.py"
MODAL_PATH = ROOT / "experiments/fp8-instrumentation-pilot/modal_pilot.py"


@pytest.fixture(scope="module")
def protocol():
    spec = importlib.util.spec_from_file_location("fp8_pilot_protocol_test", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_execution_order_and_configs_match_frozen_protocol(protocol) -> None:
    order = protocol.execution_order()
    assert order == (
        (1, 9201, 121, "bf16_kv"),
        (1, 9201, 121, "fp8_kv"),
        (2, 9202, 122, "fp8_kv"),
        (2, 9202, 122, "bf16_kv"),
        (3, 9203, 123, "bf16_kv"),
        (3, 9203, 123, "fp8_kv"),
    )
    configs = [protocol.config_for(*cell) for cell in order]
    assert len({config.experiment_id for config in configs}) == 6
    for cell, config in zip(order, configs):
        block, prompt_seed, arrival_seed, arm = cell
        assert config.workload.prompt_seed == prompt_seed
        assert config.workload.arrival_seed == arrival_seed
        assert config.workload.num_requests == 100
        assert config.workload.warmup_requests == 4
        assert config.workload.request_rate_qps == 0.70
        assert config.engine.dtype == "bfloat16"
        assert config.engine.max_num_seqs == 128
        assert config.engine.max_num_batched_tokens == 2048
        assert config.engine.enable_chunked_prefill is True
        assert config.engine.enable_prefix_caching is False
        assert config.engine.extra_args == {
            "async-scheduling": False,
            "enable-logging-iteration-details": True,
        }
        assert config.engine.kv_cache_dtype == ("fp8" if arm == "fp8_kv" else "auto")
        assert f"block-{block}" in config.tags


def test_unregistered_cell_and_retry_rules_fail_closed(protocol) -> None:
    with pytest.raises(ValueError, match="frozen execution order"):
        protocol.config_for(1, 9201, 999, "bf16_kv")
    assert protocol.retry_allowed(["dispatch_drift_exceeded"], 1)
    assert not protocol.retry_allowed(["dispatch_drift_exceeded"], 2)
    assert not protocol.retry_allowed(["status=failed"], 1)
    assert not protocol.retry_allowed(["dispatch_drift_exceeded", "other"], 1)


def _outcome(block: int, arm: str, *, target: bool = True, throughput: float = 100.0) -> dict:
    base = arm == "bf16_kv"
    return {
        "experiment_id": f"b{block}-{arm}",
        "arm": arm,
        "throughput_tokens_per_s": throughput,
        "ttft_p95_ms": 100.0,
        "tpot_p95_ms": 10.0,
        "e2e_p95_ms": 1000.0,
        "successes": 100,
        "load_state": "overloaded" if target else "healthy",
        "kv_cache_usage_peak_perc": 0.99 if target else 0.50,
        "preemptions": 4 if base else 1,
        "recomputed_token_executions": 20_000 if base else 5_000,
        "actual_prompt_plus_output_tokens": 819_200,
        "recompute_burden": 0.20 if base else 0.05,
        "measured_window_seconds": 180.0,
    }


def test_decision_rule_go_and_digest(protocol, monkeypatch) -> None:
    outcomes = {
        f"b{block}-{arm}": _outcome(
            block, arm, throughput=100.0 if arm == "bf16_kv" else 130.0
        )
        for block in (1, 2, 3)
        for arm in ("bf16_kv", "fp8_kv")
    }
    monkeypatch.setattr(protocol, "_cell_outcome", lambda result, evidence: outcomes[result])
    cells = [(block, f"b{block}-{arm}", object()) for block in (1, 2, 3) for arm in ("bf16_kv", "fp8_kv")]
    report = protocol.evaluate_pilot(cells)
    assert report["decision"] == "GO"
    digest = report.pop("content_sha256")
    assert digest == hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def test_not_target_regime_has_precedence(protocol, monkeypatch) -> None:
    outcomes = {
        f"b{block}-{arm}": _outcome(
            block,
            arm,
            target=False,
            throughput=100.0 if arm == "bf16_kv" else 80.0,
        )
        for block in (1, 2, 3)
        for arm in ("bf16_kv", "fp8_kv")
    }
    monkeypatch.setattr(protocol, "_cell_outcome", lambda result, evidence: outcomes[result])
    cells = [(block, f"b{block}-{arm}", object()) for block in (1, 2, 3) for arm in ("bf16_kv", "fp8_kv")]
    assert protocol.evaluate_pilot(cells)["decision"] == "NOT_TARGET_REGIME"


def test_modal_launcher_is_pinned_detached_checkpointed_and_budgeted() -> None:
    source = MODAL_PATH.read_text()
    compile(source, str(MODAL_PATH), "exec")
    assert "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1" in source
    assert 'GPU = "A10G"' in source
    assert "inferpilot-fp8-pilot-results" in source
    assert "run_pilot.spawn()" in source
    assert "result_volume.commit()" in source
    assert "require_metric_capabilities=True" in source
    assert "request_timeout_s=300.0" in source
    assert "HARD_STOP_USD = 12.0" in source
    assert "WORST_CASE_NEXT_CELL_USD = 2.0" in source
    assert "PRIOR_CANARY_COST_USD = 0.0781" in source
