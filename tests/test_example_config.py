"""The shipped example experiment config must parse and validate."""

from __future__ import annotations

import json
from pathlib import Path

from inferpilot import ExperimentConfig

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example_experiment.json"


def test_example_config_parses() -> None:
    raw = json.loads(EXAMPLE.read_text())
    cfg = ExperimentConfig.model_validate(raw)

    assert cfg.experiment_id == "exp-0001-baseline"
    # model is operator-chosen (placeholder in the example), not hard-coded in schema.
    assert cfg.engine.model == "facebook/opt-125m"
    assert cfg.workload.num_requests == 64
    assert cfg.slo is None


def test_example_config_roundtrips() -> None:
    cfg = ExperimentConfig.model_validate_json(EXAMPLE.read_text())
    assert ExperimentConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_unknown_field_is_rejected() -> None:
    raw = json.loads(EXAMPLE.read_text())
    raw["engine"]["definitely_not_a_real_knob"] = True
    try:
        ExperimentConfig.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        assert "definitely_not_a_real_knob" in str(exc)
    else:
        raise AssertionError("strict schema must reject unknown fields")
