"""Zero-GPU proofs for the focused 7B held-out mechanism study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/fp8-mechanism-heldout-7b"


@pytest.fixture(scope="module")
def protocol():
    spec = importlib.util.spec_from_file_location("heldout_7b_protocol_test", HERE / "protocol.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_order_and_new_seed_blocks(protocol) -> None:
    assert protocol.execution_order() == (
        (1, 9401, 141, "long_context_positive", "bf16_kv"),
        (1, 9401, 141, "short_context_negative", "fp8_kv"),
        (1, 9401, 141, "long_context_positive", "fp8_kv"),
        (1, 9401, 141, "short_context_negative", "bf16_kv"),
        (2, 9402, 142, "short_context_negative", "bf16_kv"),
        (2, 9402, 142, "long_context_positive", "fp8_kv"),
        (2, 9402, 142, "short_context_negative", "fp8_kv"),
        (2, 9402, 142, "long_context_positive", "bf16_kv"),
        (3, 9403, 143, "long_context_positive", "bf16_kv"),
        (3, 9403, 143, "short_context_negative", "fp8_kv"),
        (3, 9403, 143, "long_context_positive", "fp8_kv"),
        (3, 9403, 143, "short_context_negative", "bf16_kv"),
    )


def test_configs_bind_frozen_model_workloads_and_single_pair_variable(protocol) -> None:
    configs = [protocol.config_for(*cell) for cell in protocol.execution_order()]
    assert len({config.experiment_id for config in configs}) == 12
    for config in configs:
        assert config.engine.model == "Qwen/Qwen2.5-7B-Instruct"
        assert config.engine.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
        assert config.engine.dtype == "bfloat16"
        assert config.engine.max_num_seqs == 128
        assert config.engine.max_num_batched_tokens == 2048
        assert config.engine.enable_chunked_prefill is True
        assert config.engine.enable_prefix_caching is False
        assert config.workload.num_requests == 100
        assert config.workload.warmup_requests == 4
        assert config.workload.request_rate_qps == 4.0
    for block in (1, 2, 3):
        for workload in protocol.WORKLOADS:
            cells = [
                config
                for config in configs
                if f"block-{block}" in config.tags and workload in config.tags
            ]
            assert len(cells) == 2
            left = cells[0].model_dump(mode="json")
            right = cells[1].model_dump(mode="json")
            left["experiment_id"] = right["experiment_id"] = "id"
            left["name"] = right["name"] = "name"
            left["tags"] = right["tags"] = []
            left["engine"]["kv_cache_dtype"] = right["engine"]["kv_cache_dtype"] = "varied"
            assert left == right


def _outcome(workload: str, arm: str, *, positive_target: bool = True, effect: bool = True) -> dict:
    positive = workload == "long_context_positive"
    baseline = arm == "bf16_kv"
    if positive:
        throughput = 100.0 if baseline else (130.0 if effect else 105.0)
        burden = 0.05 if baseline else 0.005
        preemptions = 5 if baseline else 1
        load_state = "overloaded" if positive_target else "near_capacity"
    else:
        throughput = 100.0 if baseline else 102.0
        burden = 0.0
        preemptions = 0
        load_state = "healthy"
    latency = 100.0 if baseline else (90.0 if positive else 102.0)
    return {
        "experiment_id": f"{workload}-{arm}",
        "arm": arm,
        "throughput_tokens_per_s": throughput,
        "ttft_p95_ms": latency,
        "tpot_p95_ms": latency,
        "e2e_p95_ms": latency,
        "successes": 100,
        "load_state": load_state,
        "kv_cache_usage_peak_perc": 1.0 if positive else 0.3,
        "preemptions": preemptions,
        "recomputed_token_executions": int(burden * 100_000),
        "actual_prompt_plus_output_tokens": 100_000,
        "recompute_burden": burden,
        "measured_window_seconds": 120.0,
    }


def _cells_and_outcomes(protocol, *, positive_target: bool = True, effect: bool = True):
    cells, outcomes = [], {}
    for block in (1, 2, 3):
        for workload in protocol.WORKLOADS:
            for arm in protocol.ARMS:
                key = f"b{block}-{workload}-{arm}"
                cells.append((block, workload, key, object()))
                outcomes[key] = _outcome(
                    workload, arm, positive_target=positive_target, effect=effect
                )
    return cells, outcomes


def test_supporting_decision_and_digest(protocol, monkeypatch) -> None:
    cells, outcomes = _cells_and_outcomes(protocol)
    monkeypatch.setattr(protocol, "mechanism_cell_outcome", lambda result, evidence: outcomes[result])
    report = protocol.evaluate_study(cells)
    assert report["decision"] == "SUPPORTS_MECHANISM_WITHIN_ENVELOPE"
    digest = report.pop("content_sha256")
    assert digest == hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def test_target_failure_and_effect_failure_are_distinct(protocol, monkeypatch) -> None:
    cells, outcomes = _cells_and_outcomes(protocol, positive_target=False)
    monkeypatch.setattr(protocol, "mechanism_cell_outcome", lambda result, evidence: outcomes[result])
    assert protocol.evaluate_study(cells)["decision"] == "NOT_TARGET_REGIME"

    cells, outcomes = _cells_and_outcomes(protocol, effect=False)
    monkeypatch.setattr(protocol, "mechanism_cell_outcome", lambda result, evidence: outcomes[result])
    assert protocol.evaluate_study(cells)["decision"] == "INCONCLUSIVE"


def test_unregistered_cell_and_retry_fail_closed(protocol) -> None:
    with pytest.raises(ValueError, match="frozen execution order"):
        protocol.config_for(1, 9401, 999, "long_context_positive", "bf16_kv")
    assert protocol.retry_allowed(["dispatch_drift_exceeded"], 1)
    assert not protocol.retry_allowed(["dispatch_drift_exceeded"], 2)
    assert not protocol.retry_allowed(["status=failed"], 1)


def test_modal_study_is_pinned_checkpointed_and_budgeted() -> None:
    source = (HERE / "modal_study.py").read_text()
    compile(source, str(HERE / "modal_study.py"), "exec")
    assert "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1" in source
    assert 'GPU = "A10G"' in source
    assert "inferpilot-fp8-mechanism-7b-results" in source
    assert "run_study.spawn()" in source
    assert "result_volume.commit()" in source
    assert "require_metric_capabilities=True" in source
    assert "request_timeout_s=1200.0" in source
    assert "PRIOR_SPEND_USD = 0.8084" in source
    assert "HARD_STOP_USD = 12.0" in source
