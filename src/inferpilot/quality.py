"""Quality gate: never report a precision-reducing win without a quality check.

The biggest blind spot in config tuning is that capacity levers (fp8/int8 KV, weight
quantization) can SILENTLY change outputs — a "+52% throughput" that quietly degrades
answers is not a win. This gate makes quality a first-class, fail-closed precondition.

Signal: under greedy decoding (temperature=0), a quality-preserving change should produce
(near-)identical output tokens to the baseline on the same prompts. Position-wise token
agreement is therefore a direct, cheap, model-agnostic quality proxy — no logits needed.
Low agreement ⇒ the lever changed the generative distribution ⇒ block the recommendation.

(For long-context fp8 specifically, the known failure is retrieval-accuracy collapse beyond
~100k tokens; greedy-agreement on representative prompts catches exactly that drift.)
"""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import Field, model_validator

from ._base import SchemaModel


class QualitySpec(SchemaModel):
    """greedy_token_agreement is a cheap DRIFT PRE-FILTER only (over-flags open-ended gen due to
    the greedy cascade); the defensible quality VERDICT is teacher-forced KL (see KLQualitySpec)
    complemented by needle-in-haystack (see NeedleQualitySpec). Per DeepSeek review 2026-09-19."""

    quality_version: Literal["0.1.0"] = "0.1.0"
    metric: Literal["greedy_token_agreement"] = "greedy_token_agreement"
    min_agreement: float = Field(default=0.99, ge=0, le=1, description="Min position-wise token match.")
    min_pairs: int = Field(default=8, ge=1, description="Min prompts compared for a valid gate.")


class KLQualitySpec(SchemaModel):
    """Teacher-forced per-position KL gate — the defensible primary quality verdict.

    Both configs consume the SAME reference token history (teacher-forced), so this measures
    pure distributional shift without the autoregressive-cascade artifact. Thresholds from the
    published consensus: mean KL < 0.01 = distribution-lossless; but the gate is noise-floor
    relative (a KL of 0.008 is meaningless if two bf16 seeds differ by 0.006)."""

    kl_version: Literal["0.1.0"] = "0.1.0"
    mean_kl_max: float = Field(default=0.01, gt=0)
    p99_kl_max: float = Field(default=0.1, gt=0)
    noise_floor_multiple: float = Field(default=3.0, ge=1.0,
                                        description="Allow KL up to this × the seed-to-seed noise floor.")
    min_tokens: int = Field(default=4000, ge=1, description="Min corpus tokens for a valid gate.")


class NeedleQualitySpec(SchemaModel):
    """Needle-in-haystack retrieval accuracy at max context — the non-negotiable complement.

    KL is structurally blind to long-range retrieval collapse (the 91%->13% fp8 failure), which
    contributes ~nothing to average next-token loss. Required for any long-context deployment."""

    needle_version: Literal["0.1.0"] = "0.1.0"
    min_accuracy: float = Field(default=0.9, ge=0, le=1)
    max_drop_vs_baseline: float = Field(default=0.02, ge=0,
                                        description="Max allowed accuracy drop vs the bf16 baseline.")
    min_probes: int = Field(default=10, ge=1)


class KLQualityGate(SchemaModel):
    """Self-validating teacher-forced-KL quality verdict (noise-floor relative)."""

    spec: KLQualitySpec
    tokens_compared: int = Field(ge=0)
    mean_kl: float = Field(ge=0)
    p99_kl: float = Field(ge=0)
    noise_floor_kl: float = Field(ge=0, description="Mean KL between two bf16 seeds (measurement floor).")
    passed: bool
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "KLQualityGate":
        reasons = _kl_reasons(self.spec, self.tokens_compared, self.mean_kl, self.p99_kl, self.noise_floor_kl)
        if self.passed != (not reasons) or self.reasons != reasons:
            raise ValueError("KL quality gate is inconsistent with its evidence")
        return self


class NeedleQualityGate(SchemaModel):
    """Self-validating long-context retrieval-accuracy verdict."""

    spec: NeedleQualitySpec
    num_probes: int = Field(ge=0)
    candidate_accuracy: float = Field(ge=0, le=1)
    baseline_accuracy: float = Field(ge=0, le=1)
    passed: bool
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "NeedleQualityGate":
        reasons = _needle_reasons(self.spec, self.num_probes, self.candidate_accuracy, self.baseline_accuracy)
        if self.passed != (not reasons) or self.reasons != reasons:
            raise ValueError("needle quality gate is inconsistent with its evidence")
        return self


def _kl_reasons(spec: KLQualitySpec, tokens: int, mean_kl: float, p99_kl: float, floor: float) -> list[str]:
    reasons: list[str] = []
    if tokens < spec.min_tokens:
        reasons.append("insufficient_corpus")
    mean_budget = max(spec.mean_kl_max, floor * spec.noise_floor_multiple)
    p99_budget = max(spec.p99_kl_max, floor * spec.noise_floor_multiple)
    if mean_kl > mean_budget:
        reasons.append("mean_kl_above_threshold")
    if p99_kl > p99_budget:
        reasons.append("p99_kl_above_threshold")  # damage concentrated on few positions (long-ctx risk)
    return reasons


def _needle_reasons(spec: NeedleQualitySpec, n: int, cand: float, base: float) -> list[str]:
    reasons: list[str] = []
    if n < spec.min_probes:
        reasons.append("insufficient_probes")
    if cand < spec.min_accuracy:
        reasons.append("accuracy_below_floor")
    if base - cand > spec.max_drop_vs_baseline:
        reasons.append("regressed_vs_baseline")
    return reasons


def evaluate_kl_quality(spec: KLQualitySpec, tokens_compared: int, mean_kl: float, p99_kl: float,
                        noise_floor_kl: float) -> KLQualityGate:
    reasons = _kl_reasons(spec, tokens_compared, mean_kl, p99_kl, noise_floor_kl)
    return KLQualityGate(spec=spec, tokens_compared=tokens_compared, mean_kl=mean_kl, p99_kl=p99_kl,
                         noise_floor_kl=noise_floor_kl, passed=not reasons, reasons=reasons)


def evaluate_needle_quality(spec: NeedleQualitySpec, num_probes: int, candidate_accuracy: float,
                            baseline_accuracy: float) -> NeedleQualityGate:
    reasons = _needle_reasons(spec, num_probes, candidate_accuracy, baseline_accuracy)
    return NeedleQualityGate(spec=spec, num_probes=num_probes, candidate_accuracy=candidate_accuracy,
                             baseline_accuracy=baseline_accuracy, passed=not reasons, reasons=reasons)


def greedy_token_agreement(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[int, int]:
    """(matching, total) position-wise token matches over paired (baseline, candidate) outputs.

    Compared over the overlapping prefix length; a length mismatch counts the missing tail as
    non-matching (a divergence), so truncation/early-stop is penalised."""
    matching = total = 0
    for base, cand in pairs:
        n = max(len(base), len(cand))
        total += n
        for i in range(min(len(base), len(cand))):
            if base[i] == cand[i]:
                matching += 1
    return matching, total


class QualityGate(SchemaModel):
    """Self-validating pass/fail on output-quality preservation for a candidate config."""

    spec: QualitySpec
    num_pairs: int = Field(ge=0)
    matching_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    agreement_rate: float = Field(ge=0, le=1)
    passed: bool
    reasons: list[str]

    @model_validator(mode="after")
    def _check(self) -> "QualityGate":
        rate = self.matching_tokens / self.total_tokens if self.total_tokens else 0.0
        reasons: list[str] = []
        if self.num_pairs < self.spec.min_pairs:
            reasons.append("insufficient_pairs")
        if self.total_tokens == 0:
            reasons.append("no_tokens_compared")
        if rate < self.spec.min_agreement:
            reasons.append("agreement_below_threshold")
        passed = not reasons
        if abs(self.agreement_rate - rate) > 1e-9 or self.passed != passed or self.reasons != reasons:
            raise ValueError("quality gate is inconsistent with its evidence")
        return self


def evaluate_quality(
    spec: QualitySpec, pairs: Sequence[tuple[Sequence[int], Sequence[int]]]
) -> QualityGate:
    """Compare paired greedy outputs (baseline vs candidate) into a fail-closed quality verdict."""
    matching, total = greedy_token_agreement(pairs)
    rate = matching / total if total else 0.0
    reasons: list[str] = []
    if len(pairs) < spec.min_pairs:
        reasons.append("insufficient_pairs")
    if total == 0:
        reasons.append("no_tokens_compared")
    if rate < spec.min_agreement:
        reasons.append("agreement_below_threshold")
    return QualityGate(
        spec=spec, num_pairs=len(pairs), matching_tokens=matching, total_tokens=total,
        agreement_rate=rate, passed=not reasons, reasons=reasons,
    )
