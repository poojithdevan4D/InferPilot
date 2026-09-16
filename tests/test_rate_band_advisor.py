from copy import deepcopy
import hashlib
import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferpilot.advisor import (
    AdvisorPolicy,
    AdvisorRequest,
    RateBandEvidenceReport,
    recommend,
)
from inferpilot.advisor.recommend import load_verified_policy

recommend_module = importlib.import_module("inferpilot.advisor.recommend")


def _policy() -> AdvisorPolicy:
    return AdvisorPolicy(
        policy_version="0.2.0",
        policy_id="m3",
        model="q",
        revision="abc",
        gpu_name="g",
        prompt_tokens=128,
        output_tokens=32,
        arrival_pattern="poisson-v1",
        fixed_engine_fields={"max_num_batched_tokens": 512},
        regimes=[
            {
                "request_rate_qps": 2,
                "min_rate_qps": 1.5,
                "max_rate_qps": 2.5,
                "engine_overrides": {"max_num_seqs": 2},
            },
            {
                "request_rate_qps": 6,
                "min_rate_qps": 5.5,
                "max_rate_qps": 6.5,
                "engine_overrides": {"max_num_seqs": 4},
            },
        ],
        evidence_report_sha256="a" * 64,
        applicability_report_sha256="b" * 64,
    )


def _request(rate: float) -> AdvisorRequest:
    return AdvisorRequest(
        model="q",
        revision="abc",
        gpu_name="g",
        prompt_tokens=128,
        output_tokens=32,
        arrival_pattern="poisson-v1",
        request_rate_qps=rate,
    )


@pytest.mark.parametrize(
    "rate,width", [(1.5, 2), (2.1, 2), (2.5, 2), (5.5, 4), (6.2, 4), (6.5, 4)]
)
def test_validated_band_recommends(rate: float, width: int) -> None:
    decision = recommend(_policy(), _request(rate))
    assert decision.status == "recommended"
    assert decision.engine_overrides["max_num_seqs"] == width


@pytest.mark.parametrize("rate", [1.49, 3.0, 5.49, 6.51])
def test_unvalidated_rate_abstains(rate: float) -> None:
    decision = recommend(_policy(), _request(rate))
    assert decision.status == "abstain"
    assert decision.reasons == ["unsupported_request_rate_qps"]


def test_policy_rejects_overlapping_bands() -> None:
    raw = _policy().model_dump(mode="json")
    raw["regimes"][1]["min_rate_qps"] = 2.5
    with pytest.raises(ValidationError, match="overlap"):
        AdvisorPolicy.model_validate(raw)


def test_policy_v2_requires_applicability_digest() -> None:
    raw = _policy().model_dump(mode="json")
    raw["applicability_report_sha256"] = None
    with pytest.raises(ValidationError, match="applicability evidence"):
        AdvisorPolicy.model_validate(raw)


def test_committed_evidence_roundtrips_and_rejects_tampering() -> None:
    path = Path(__file__).parents[1] / "evidence" / "m6-rate-bands.json"
    report = RateBandEvidenceReport.model_validate_json(path.read_text())
    assert len(report.bands) == 2
    assert sum(len(band.cells) for band in report.bands) == 12
    raw = report.model_dump(mode="json")
    tampered = deepcopy(raw)
    tampered["bands"][0]["cells"][0]["tpot_p95_ms"] = 9.0
    with pytest.raises(ValidationError, match="violates SLO"):
        RateBandEvidenceReport.model_validate(tampered)


def test_evidence_rejects_action_tampering() -> None:
    path = Path(__file__).parents[1] / "evidence" / "m6-rate-bands.json"
    raw = RateBandEvidenceReport.model_validate_json(path.read_text()).model_dump(mode="json")
    raw["bands"][0]["cells"][0]["max_num_seqs"] = 4
    with pytest.raises(ValidationError, match="action"):
        RateBandEvidenceReport.model_validate(raw)


def test_policy_loader_requires_and_binds_applicability_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimization = b"self-validating optimization report"
    evidence_source = Path(__file__).parents[1] / "evidence" / "m6-rate-bands.json"
    applicability = evidence_source.read_bytes()
    policy_source = Path(__file__).parents[1] / "policies" / "m3-rtx3050-qwen05b.json"
    raw = AdvisorPolicy.model_validate_json(policy_source.read_text()).model_dump(mode="json")
    raw["evidence_report_sha256"] = hashlib.sha256(optimization).hexdigest()
    policy_path = tmp_path / "policy.json"
    optimization_path = tmp_path / "optimization.json"
    applicability_path = tmp_path / "applicability.json"
    policy_path.write_text(AdvisorPolicy.model_validate(raw).model_dump_json())
    optimization_path.write_bytes(optimization)
    applicability_path.write_bytes(applicability)
    monkeypatch.setattr(
        recommend_module.PolicyBenchmarkReport,
        "model_validate_json",
        lambda _payload: object(),
    )
    with pytest.raises(ValueError, match="requires an applicability"):
        load_verified_policy(policy_path, optimization_path)
    assert load_verified_policy(policy_path, optimization_path, applicability_path).policy_version == "0.2.0"
    raw["gpu_name"] = "different GPU"
    policy_path.write_text(AdvisorPolicy.model_validate(raw).model_dump_json())
    with pytest.raises(ValueError, match="context"):
        load_verified_policy(policy_path, optimization_path, applicability_path)
