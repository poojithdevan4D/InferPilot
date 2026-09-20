"""The shipped example experiment config must parse and validate."""

from __future__ import annotations

import json
from pathlib import Path

from inferpilot import ExperimentConfig
from inferpilot.configuration_gate import ConfigurationGateSpec

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_experiment.json"
ASSESSMENT = Path(__file__).resolve().parents[1] / "examples" / "assessment_config.json"
GATE_SPEC = (
    Path(__file__).resolve().parents[1] / "examples" / "configuration_gate_spec.json"
)
C4_SEQ1 = Path(__file__).resolve().parents[1] / "examples" / "experiment_c4_seq1.json"
C4_SEQ2 = Path(__file__).resolve().parents[1] / "examples" / "experiment_c4_seq2.json"
C4_SEQ4 = Path(__file__).resolve().parents[1] / "examples" / "experiment_c4_seq4.json"


def test_example_config_parses() -> None:
    raw = json.loads(EXAMPLE.read_text())
    cfg = ExperimentConfig.model_validate(raw)

    assert cfg.experiment_id == "exp-0001-baseline"
    assert cfg.schema_version == "0.3.0"
    # model is operator-chosen, pinned to an exact HF revision.
    assert cfg.engine.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.engine.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    # PyTorch sampler fallback (FlashInfer JIT unsupported by local nvcc 12.4).
    assert cfg.engine.sampler_backend == "pytorch"
    # Prefix caching EXPLICITLY disabled (False, not None) -> --no-enable-prefix-caching.
    assert cfg.engine.enable_prefix_caching is False
    assert cfg.workload.num_requests == 32
    assert cfg.workload.warmup_requests == 4
    assert cfg.workload.output_tokens == 32
    assert cfg.workload.ignore_eos is True
    assert cfg.slo is None


def test_example_config_roundtrips() -> None:
    cfg = ExperimentConfig.model_validate_json(EXAMPLE.read_text())
    assert ExperimentConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_assessment_example_is_diagnosis_capable_and_operator_scoped() -> None:
    cfg = ExperimentConfig.model_validate_json(ASSESSMENT.read_text())
    assert cfg.schema_version == "0.5.0"
    assert cfg.engine.revision is not None
    assert cfg.workload.request_rate_qps == 2.0
    assert cfg.workload.num_requests >= 100
    assert cfg.workload.warmup_requests >= 1
    assert cfg.slo is not None
    assert "illustrative-slo" in cfg.tags


def test_configuration_gate_example_requires_both_precision_quality_gates() -> None:
    spec = ConfigurationGateSpec.model_validate_json(GATE_SPEC.read_text())
    assert spec.varied_engine_fields == ("kv_cache_dtype",)
    assert set(spec.required_quality_gates) == {
        "teacher_forced_kl",
        "needle_retrieval",
    }


def test_unknown_field_is_rejected() -> None:
    raw = json.loads(EXAMPLE.read_text())
    raw["engine"]["definitely_not_a_real_knob"] = True
    try:
        ExperimentConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        assert "definitely_not_a_real_knob" in str(exc)
    else:
        raise AssertionError("strict schema must reject unknown fields")


def test_controlled_scheduling_configs_differ_only_in_max_num_seqs() -> None:
    baseline = ExperimentConfig.model_validate_json(C4_SEQ1.read_text())
    midpoint = ExperimentConfig.model_validate_json(C4_SEQ2.read_text())
    candidate = ExperimentConfig.model_validate_json(C4_SEQ4.read_text())

    assert baseline.workload == midpoint.workload == candidate.workload
    assert baseline.engine.max_num_seqs == 1
    assert midpoint.engine.max_num_seqs == 2
    assert candidate.engine.max_num_seqs == 4

    baseline_engine = baseline.engine.model_dump()
    midpoint_engine = midpoint.engine.model_dump()
    candidate_engine = candidate.engine.model_dump()
    baseline_engine.pop("max_num_seqs")
    midpoint_engine.pop("max_num_seqs")
    candidate_engine.pop("max_num_seqs")
    assert baseline_engine == midpoint_engine == candidate_engine
