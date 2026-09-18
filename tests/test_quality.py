"""Quality gate: fail-closed output-preservation check for precision-reducing levers."""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from inferpilot import QualityGate, QualitySpec, evaluate_quality, greedy_token_agreement


def _pairs(n, mismatch_every=0):
    out=[]
    for k in range(n):
        base=list(range(20))
        cand=list(base)
        if mismatch_every and k % mismatch_every == 0:
            cand[0]=999  # one differing token
        out.append((base,cand))
    return out


def test_identical_outputs_pass() -> None:
    g=evaluate_quality(QualitySpec(), _pairs(10))
    assert g.passed and g.agreement_rate==1.0


def test_divergent_outputs_fail() -> None:
    # every pair differs in 1/20 tokens -> agreement 0.95 < 0.99 default
    g=evaluate_quality(QualitySpec(), _pairs(10, mismatch_every=1))
    assert not g.passed and "agreement_below_threshold" in g.reasons


def test_length_mismatch_penalised() -> None:
    pairs=[(list(range(20)), list(range(10))) for _ in range(10)]  # candidate truncated
    g=evaluate_quality(QualitySpec(), pairs)
    assert g.agreement_rate==0.5 and not g.passed


def test_too_few_pairs_is_failclosed() -> None:
    g=evaluate_quality(QualitySpec(), _pairs(3))
    assert not g.passed and "insufficient_pairs" in g.reasons


def test_roundtrip_and_tamper() -> None:
    g=evaluate_quality(QualitySpec(), _pairs(10))
    assert QualityGate.model_validate_json(g.model_dump_json())==g
    raw=g.model_dump(mode="json"); raw["passed"]=False
    with pytest.raises(ValidationError, match="inconsistent with its evidence"):
        QualityGate.model_validate(raw)


from inferpilot import (KLQualitySpec, NeedleQualitySpec, evaluate_kl_quality, evaluate_needle_quality)


def test_kl_gate_passes_below_threshold() -> None:
    g = evaluate_kl_quality(KLQualitySpec(), tokens_compared=8000, mean_kl=0.006, p99_kl=0.05, noise_floor_kl=0.004)
    assert g.passed


def test_kl_gate_noise_floor_relative() -> None:
    # mean_kl 0.02 > 0.01 abs, but floor 0.01 * 3 = 0.03 budget -> passes (within noise)
    g = evaluate_kl_quality(KLQualitySpec(), tokens_compared=8000, mean_kl=0.02, p99_kl=0.05, noise_floor_kl=0.01)
    assert g.passed


def test_kl_gate_fails_high_p99_concentrated_damage() -> None:
    g = evaluate_kl_quality(KLQualitySpec(), tokens_compared=8000, mean_kl=0.005, p99_kl=1.0, noise_floor_kl=0.002)
    assert not g.passed and "p99_kl_above_threshold" in g.reasons


def test_needle_gate_catches_longctx_regression() -> None:
    g = evaluate_needle_quality(NeedleQualitySpec(), num_probes=20, candidate_accuracy=0.13, baseline_accuracy=0.91)
    assert not g.passed and "regressed_vs_baseline" in g.reasons and "accuracy_below_floor" in g.reasons


def test_needle_gate_passes_preserved() -> None:
    g = evaluate_needle_quality(NeedleQualitySpec(), num_probes=20, candidate_accuracy=0.92, baseline_accuracy=0.93)
    assert g.passed
