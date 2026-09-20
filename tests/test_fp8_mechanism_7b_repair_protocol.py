"""Zero-GPU checks for the budget-bounded 7B repair study."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from inferpilot.runner.schedule import generate_poisson_offsets

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/fp8-mechanism-repair-7b"


@pytest.fixture(scope="module")
def protocol():
    spec = importlib.util.spec_from_file_location("repair_7b_protocol_test", HERE / "protocol.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_changes_only_seeds_ids_and_short_request_count(protocol) -> None:
    configs = [protocol.config_for(*cell) for cell in protocol.execution_order()]
    assert len(configs) == 12
    assert len({config.experiment_id for config in configs}) == 12
    assert {config.workload.prompt_seed for config in configs} == {9501, 9502, 9503}
    assert {config.workload.arrival_seed for config in configs} == {151, 152, 153}
    for config in configs:
        expected = 128 if config.workload.name == "short_context_negative" else 100
        assert config.workload.num_requests == expected
        if config.workload.name == "short_context_negative":
            assert generate_poisson_offsets(
                expected, config.workload.request_rate_qps, config.workload.effective_arrival_seed
            )[-1] >= 30.0


def test_each_pair_differs_only_in_kv_dtype(protocol) -> None:
    configs = [protocol.config_for(*cell) for cell in protocol.execution_order()]
    for block in (1, 2, 3):
        for workload in protocol.WORKLOADS:
            pair = [
                config for config in configs
                if f"block-{block}" in config.tags and workload in config.tags
            ]
            assert len(pair) == 2
            left, right = (config.model_dump(mode="json") for config in pair)
            for payload in (left, right):
                payload["experiment_id"] = "id"
                payload["name"] = "name"
                payload["tags"] = []
                payload["engine"]["kv_cache_dtype"] = "varied"
            assert left == right


def test_budget_and_runtime_guards_are_frozen() -> None:
    source = (HERE / "modal_study.py").read_text()
    compile(source, str(HERE / "modal_study.py"), "exec")
    assert "INCREMENTAL_HARD_STOP_USD = 2.25" in source
    assert "WORST_CASE_NEXT_CELL_USD = 0.35" in source
    assert "CELL_WALL_TIME_LIMIT_S = 900.0" in source
    assert "max_wall_time_s=CELL_WALL_TIME_LIMIT_S" in source
    assert 'gpu=GPU' in source


def test_unregistered_seed_and_retries_fail_closed(protocol) -> None:
    with pytest.raises(ValueError, match="frozen repair execution order"):
        protocol.config_for(1, 9501, 999, "long_context_positive", "bf16_kv")
    assert protocol.retry_allowed(["dispatch_drift_exceeded"], 1)
    assert not protocol.retry_allowed(["dispatch_drift_exceeded"], 2)
    assert not protocol.retry_allowed(["load_evidence_missing"], 1)
