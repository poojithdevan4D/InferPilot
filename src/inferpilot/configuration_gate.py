"""Fail-closed acceptance gate for one controlled configuration candidate.

The gate composes existing comparison, SLO, and quality contracts.  It does not
run benchmarks, collect quality data, or authorize a deployment.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, StringConstraints, model_validator

from ._base import SchemaModel
from .advisor.config_comparison import (
    ComparisonSpec,
    ConfigComparison,
    compare_configs,
)
from .comparison.fingerprint import comparison_fingerprint, exact_fingerprint
from .comparison.models import SLOCheck, SLO_RULES
from .config import EngineConfig, SLO
from .quality import KLQualityGate, NeedleQualityGate
from .results import ExperimentResult

GateVerdict = Literal["pass", "fail", "abstain"]
QualityKind = Literal["teacher_forced_kl", "needle_retrieval"]
QualityGate = Union[KLQualityGate, NeedleQualityGate]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_PRECISION_FIELDS = frozenset({"dtype", "kv_cache_dtype"})
_INSUFFICIENT_QUALITY_REASONS = frozenset(
    {"insufficient_corpus", "insufficient_probes"}
)


def _quality_kind(gate: QualityGate) -> QualityKind:
    if isinstance(gate, KLQualityGate):
        return "teacher_forced_kl"
    return "needle_retrieval"


class BoundQualityEvidence(SchemaModel):
    """One quality result bound to the exact compared experiment IDs and corpus."""

    evidence_version: Literal["0.1.0"] = "0.1.0"
    baseline_experiment_id: str = Field(min_length=1)
    candidate_experiment_id: str = Field(min_length=1)
    baseline_fingerprint: Sha256
    candidate_fingerprint: Sha256
    corpus_id: str = Field(min_length=1)
    corpus_sha256: Sha256
    kind: QualityKind
    gate: QualityGate

    @model_validator(mode="after")
    def _check(self) -> "BoundQualityEvidence":
        if self.kind != _quality_kind(self.gate):
            raise ValueError("quality evidence kind contradicts its gate")
        if self.baseline_experiment_id == self.candidate_experiment_id:
            raise ValueError("quality evidence requires distinct experiment IDs")
        return self


class ConfigurationGateSpec(SchemaModel):
    """Operator-reviewed policy fixed before accepting a candidate."""

    spec_version: Literal["0.1.0"] = "0.1.0"
    varied_engine_fields: tuple[str, ...] = Field(min_length=1)
    slo: SLO
    required_quality_gates: tuple[QualityKind, ...] = Field(min_length=1)
    comparison: ComparisonSpec = Field(default_factory=ComparisonSpec)

    @model_validator(mode="after")
    def _check(self) -> "ConfigurationGateSpec":
        if len(set(self.varied_engine_fields)) != len(self.varied_engine_fields):
            raise ValueError("varied_engine_fields must be distinct")
        unknown = sorted(set(self.varied_engine_fields) - set(EngineConfig.model_fields))
        if unknown:
            raise ValueError(f"unknown EngineConfig field(s): {unknown}")
        if len(set(self.required_quality_gates)) != len(
            self.required_quality_gates
        ):
            raise ValueError("required_quality_gates must be distinct")
        if not any(value is not None for value in self.slo.model_dump().values()):
            raise ValueError("configuration gate requires at least one SLO constraint")
        if _PRECISION_FIELDS.intersection(self.varied_engine_fields) and not {
            "teacher_forced_kl",
            "needle_retrieval",
        }.issubset(self.required_quality_gates):
            raise ValueError(
                "precision-changing candidates require both teacher-forced KL "
                "and needle-retrieval quality gates"
            )
        return self


def _validate_context(
    spec: ConfigurationGateSpec,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> None:
    fields = spec.varied_engine_fields
    if baseline.config.experiment_id == candidate.config.experiment_id:
        raise ValueError("baseline and candidate experiment IDs must be distinct")
    if baseline.config.slo != spec.slo or candidate.config.slo != spec.slo:
        raise ValueError("both run configs must carry the gate's exact SLO")
    if comparison_fingerprint(baseline, fields) != comparison_fingerprint(
        candidate, fields
    ):
        raise ValueError(
            "baseline and candidate differ outside the declared engine fields"
        )
    unchanged = [
        field
        for field in fields
        if getattr(baseline.config.engine, field)
        == getattr(candidate.config.engine, field)
    ]
    if unchanged:
        raise ValueError(f"declared engine field(s) did not change: {unchanged}")


def _quality_status(
    spec: ConfigurationGateSpec,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    evidence: tuple[BoundQualityEvidence, ...],
) -> tuple[list[str], list[str]]:
    observed: dict[QualityKind, BoundQualityEvidence] = {}
    for item in evidence:
        if item.baseline_experiment_id != baseline.config.experiment_id:
            raise ValueError("quality evidence baseline experiment ID mismatch")
        if item.candidate_experiment_id != candidate.config.experiment_id:
            raise ValueError("quality evidence candidate experiment ID mismatch")
        if item.baseline_fingerprint != exact_fingerprint(baseline):
            raise ValueError("quality evidence baseline fingerprint mismatch")
        if item.candidate_fingerprint != exact_fingerprint(candidate):
            raise ValueError("quality evidence candidate fingerprint mismatch")
        if item.kind in observed:
            raise ValueError(f"duplicate quality evidence kind: {item.kind}")
        if item.kind not in spec.required_quality_gates:
            raise ValueError(f"undeclared quality evidence kind: {item.kind}")
        observed[item.kind] = item

    failures: list[str] = []
    unknowns: list[str] = []
    for kind in spec.required_quality_gates:
        item = observed.get(kind)
        if item is None:
            unknowns.append(f"quality_evidence_missing:{kind}")
            continue
        if item.gate.passed:
            continue
        if _INSUFFICIENT_QUALITY_REASONS.intersection(item.gate.reasons):
            unknowns.append(f"quality_evidence_insufficient:{kind}")
        else:
            failures.append(f"quality_gate_failed:{kind}")
    return failures, unknowns


def _slo_checks(spec: ConfigurationGateSpec, candidate: ExperimentResult) -> list[SLOCheck]:
    aggregates = candidate.aggregates
    if aggregates is None:
        return []
    checks: list[SLOCheck] = []
    for slo_field, threshold in spec.slo.model_dump().items():
        if threshold is None:
            continue
        metric, operator = SLO_RULES[slo_field]
        observed = getattr(aggregates, metric)
        if observed is None:
            continue
        passed = observed <= threshold if operator == "<=" else observed >= threshold
        checks.append(
            SLOCheck(
                slo_field=slo_field,
                metric=metric,
                operator=operator,
                threshold=threshold,
                observed_worst=observed,
                passed=passed,
            )
        )
    return checks


def _derive(
    spec: ConfigurationGateSpec,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    quality_evidence: tuple[BoundQualityEvidence, ...],
) -> tuple[Optional[ConfigComparison], tuple[SLOCheck, ...], GateVerdict, tuple[str, ...]]:
    _validate_context(spec, baseline, candidate)
    quality_failures, quality_unknowns = _quality_status(
        spec, baseline, candidate, quality_evidence
    )
    failures: list[str] = []
    unknowns: list[str] = []

    if not baseline.is_baseline_eligible:
        unknowns.append("baseline_ineligible")
    if not candidate.is_baseline_eligible:
        unknowns.append("candidate_ineligible")

    comparison = None
    checks: list[SLOCheck] = []
    if not unknowns:
        comparison = compare_configs(
            spec.comparison,
            baseline,
            candidate,
            incumbent_load_evidence=baseline.load_evidence,
            candidate_load_evidence=candidate.load_evidence,
        )
        checks = _slo_checks(spec, candidate)
        if len(checks) != sum(
            value is not None for value in spec.slo.model_dump().values()
        ):
            unknowns.append("candidate_slo_metric_missing")
        elif any(not check.passed for check in checks):
            failures.append("candidate_slo_failed")

        if comparison.verdict in {
            "candidate_infeasible",
            "incumbent_kept",
            "inconclusive_tradeoff",
        }:
            failures.append(f"performance:{comparison.verdict}")
        elif comparison.verdict == "inconclusive_evidence":
            unknowns.append("performance:inconclusive_evidence")

    failures.extend(quality_failures)
    unknowns.extend(quality_unknowns)
    if failures:
        verdict: GateVerdict = "fail"
        reasons = failures + unknowns
    elif unknowns:
        verdict = "abstain"
        reasons = unknowns
    else:
        verdict = "pass"
        reasons = ["candidate_dominates", "candidate_slo_passed", "quality_gates_passed"]
    return comparison, tuple(checks), verdict, tuple(reasons)


class ConfigurationGateReport(SchemaModel):
    """Self-validating PASS/FAIL/ABSTAIN decision for one observed candidate."""

    report_version: Literal["0.1.0"] = "0.1.0"
    spec: ConfigurationGateSpec
    baseline: ExperimentResult
    candidate: ExperimentResult
    quality_evidence: tuple[BoundQualityEvidence, ...] = ()
    comparison: Optional[ConfigComparison] = None
    candidate_slo_checks: tuple[SLOCheck, ...] = ()
    verdict: GateVerdict
    reasons: tuple[str, ...]
    deployment_authorized: Literal[False] = False
    interpretation: Literal[
        "Observed-pair acceptance only; operator review and a controlled canary remain required."
    ] = (
        "Observed-pair acceptance only; operator review and a controlled canary remain required."
    )

    @model_validator(mode="after")
    def _check(self) -> "ConfigurationGateReport":
        expected = _derive(
            self.spec,
            self.baseline,
            self.candidate,
            self.quality_evidence,
        )
        actual = (
            self.comparison,
            self.candidate_slo_checks,
            self.verdict,
            self.reasons,
        )
        if actual != expected:
            raise ValueError("configuration gate report contradicts its evidence")
        return self


def evaluate_configuration_gate(
    spec: ConfigurationGateSpec,
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    quality_evidence: tuple[BoundQualityEvidence, ...] = (),
) -> ConfigurationGateReport:
    comparison, checks, verdict, reasons = _derive(
        spec, baseline, candidate, quality_evidence
    )
    return ConfigurationGateReport(
        spec=spec,
        baseline=baseline,
        candidate=candidate,
        quality_evidence=quality_evidence,
        comparison=comparison,
        candidate_slo_checks=checks,
        verdict=verdict,
        reasons=reasons,
    )


def bind_quality_evidence(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    *,
    corpus_id: str,
    corpus_sha256: str,
    gate: QualityGate,
) -> BoundQualityEvidence:
    """Bind an already measured quality gate to exact run configurations."""
    return BoundQualityEvidence(
        baseline_experiment_id=baseline.config.experiment_id,
        candidate_experiment_id=candidate.config.experiment_id,
        baseline_fingerprint=exact_fingerprint(baseline),
        candidate_fingerprint=exact_fingerprint(candidate),
        corpus_id=corpus_id,
        corpus_sha256=corpus_sha256,
        kind=_quality_kind(gate),
        gate=gate,
    )
