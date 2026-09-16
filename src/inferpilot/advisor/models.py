from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .._base import SchemaModel


class RateRegime(SchemaModel):
    request_rate_qps: float = Field(gt=0)
    min_rate_qps: float | None = Field(default=None, gt=0)
    max_rate_qps: float | None = Field(default=None, gt=0)
    engine_overrides: dict[str, Any] = Field(min_length=1)


class AdvisorPolicy(SchemaModel):
    policy_version: Literal["0.1.0", "0.2.0"] = "0.1.0"
    policy_id: str
    model: str
    revision: str
    gpu_name: str
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=1)
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"]
    fixed_engine_fields: dict[str, Any]
    regimes: list[RateRegime] = Field(min_length=1)
    evidence_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    applicability_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check(self):
        rates = [regime.request_rate_qps for regime in self.regimes]
        if len(set(rates)) != len(rates):
            raise ValueError("regime rates must be distinct")
        has_band = any(
            regime.min_rate_qps is not None or regime.max_rate_qps is not None
            for regime in self.regimes
        )
        if self.policy_version == "0.1.0":
            if has_band or self.applicability_report_sha256 is not None:
                raise ValueError("rate-band evidence requires policy 0.2.0")
            return self
        if self.applicability_report_sha256 is None:
            raise ValueError("policy 0.2.0 requires applicability evidence")
        for regime in self.regimes:
            if regime.min_rate_qps is None or regime.max_rate_qps is None:
                raise ValueError("policy 0.2.0 requires bounded regimes")
            if not regime.min_rate_qps <= regime.request_rate_qps <= regime.max_rate_qps:
                raise ValueError("nominal regime rate must lie inside its band")
        ordered = sorted(self.regimes, key=lambda regime: regime.min_rate_qps)
        if any(
            left.max_rate_qps >= right.min_rate_qps
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("policy rate bands overlap")
        return self


class AdvisorRequest(SchemaModel):
    model: str
    revision: str
    gpu_name: str
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=1)
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"]
    request_rate_qps: float = Field(gt=0)


class AdvisorDecision(SchemaModel):
    decision_version: Literal["0.1.0"] = "0.1.0"
    policy: AdvisorPolicy
    request: AdvisorRequest
    status: Literal["recommended", "abstain"]
    engine_overrides: dict[str, Any] | None = None
    reasons: list[str]

    @model_validator(mode="after")
    def check(self):
        from .recommend import _derive

        status, overrides, reasons = _derive(self.policy, self.request)
        if (self.status, self.engine_overrides, self.reasons) != (status, overrides, reasons):
            raise ValueError("decision is inconsistent with policy and request")
        return self
