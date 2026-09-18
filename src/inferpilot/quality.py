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
    quality_version: Literal["0.1.0"] = "0.1.0"
    metric: Literal["greedy_token_agreement"] = "greedy_token_agreement"
    min_agreement: float = Field(default=0.99, ge=0, le=1, description="Min position-wise token match.")
    min_pairs: int = Field(default=8, ge=1, description="Min prompts compared for a valid gate.")


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
