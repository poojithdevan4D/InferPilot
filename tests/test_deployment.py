"""Deployment-fit physics: which GPU, how much KV headroom, fp8 effect."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferpilot import FitAnalysis, ModelFootprint, analyze_fit

# Qwen2.5-14B: 48 layers, 8 KV heads, head_dim 128, bf16 (from HF config)
Q14B = ModelFootprint(
    model="Qwen/Qwen2.5-14B-Instruct", params_billions=14.77,
    num_layers=48, num_kv_heads=8, head_dim=128, weight_dtype="bf16",
)


def test_kv_bytes_per_token_matches_measured() -> None:
    assert Q14B.kv_bytes_per_token("bf16") == 2 * 48 * 8 * 128 * 2  # 192 KiB/token
    assert Q14B.kv_bytes_per_token("fp8") == Q14B.kv_bytes_per_token("bf16") / 2


def test_14b_fits_a100_40g_with_tight_kv() -> None:
    fit = analyze_fit(Q14B, "A100-40GB", context_tokens=2560)
    assert fit.fits
    assert 25 < fit.weights_gb_per_gpu < 32     # ~29.5 GB bf16 weights
    assert 0 < fit.kv_budget_gb < 8             # tight KV headroom
    assert fit.max_concurrent_requests >= 1


def test_fp8_kv_roughly_doubles_concurrency() -> None:
    bf16 = analyze_fit(Q14B, "A100-40GB", context_tokens=2560, kv_dtype="bf16")
    fp8 = analyze_fit(Q14B, "A100-40GB", context_tokens=2560, kv_dtype="fp8")
    assert fp8.max_concurrent_requests >= 2 * bf16.max_concurrent_requests - 1


def test_does_not_fit_when_weights_exceed_vram() -> None:
    fit = analyze_fit(Q14B, "A10G", context_tokens=2560)  # 24GB < ~29.5GB weights
    assert not fit.fits and fit.max_concurrent_requests == 0


def test_tensor_parallel_frees_kv_headroom() -> None:
    tp1 = analyze_fit(Q14B, "A100-40GB", context_tokens=2560, tensor_parallel=1)
    tp2 = analyze_fit(Q14B, "A100-40GB", context_tokens=2560, tensor_parallel=2)
    assert tp2.kv_budget_gb > tp1.kv_budget_gb
    assert tp2.weights_gb_per_gpu < tp1.weights_gb_per_gpu


def test_unknown_gpu_needs_explicit_vram() -> None:
    with pytest.raises(ValueError, match="unknown GPU"):
        analyze_fit(Q14B, "MadeUpGPU", context_tokens=2560)
    assert analyze_fit(Q14B, "MadeUpGPU", context_tokens=2560, gpu_vram_gb=48.0).fits


def test_fit_is_self_validating() -> None:
    fit = analyze_fit(Q14B, "A100-40GB", context_tokens=2560)
    assert FitAnalysis.model_validate_json(fit.model_dump_json()) == fit
    raw = fit.model_dump(mode="json")
    raw["max_concurrent_requests"] = 9999
    with pytest.raises(ValidationError, match="inconsistent with model/GPU/context"):
        FitAnalysis.model_validate(raw)
