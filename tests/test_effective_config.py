"""Parsing vLLM's resolved config from startup logs, and fidelity checks."""

from __future__ import annotations

from inferpilot import EngineConfig
from inferpilot.runner.effective_config import check_fidelity, parse_effective_config

# Realistic vLLM 0.29 startup log fragment.
LOG = (
    "INFO 09-14 14:50:16 [api_utils.py:286] non-default args: "
    "{'host': '127.0.0.1', 'port': 34747, 'model': 'Qwen/Qwen2.5-0.5B-Instruct', "
    "'revision': '7ae557604adf67be50417f59c2c2f167def9a775', 'max_model_len': 2048, "
    "'gpu_memory_utilization': 0.85, 'max_num_batched_tokens': 2048, 'max_num_seqs': 1, "
    "'generation_config': 'vllm'}\n"
    "INFO 09-14 14:50:54 [core.py:123] Initializing a V1 LLM engine (v0.29.0) with config: "
    "model='Qwen/Qwen2.5-0.5B-Instruct', "
    "revision=7ae557604adf67be50417f59c2c2f167def9a775, max_seq_len=2048, "
    "enable_prefix_caching=False, enable_chunked_prefill=True, kv_cache_dtype=auto, seed=0, "
    "served_model_name=Qwen/Qwen2.5-0.5B-Instruct\n"
    "INFO 09-14 14:50:57 [topk_topp_sampler.py:46] FlashInfer top-p/top-k sampling "
    "disabled via VLLM_USE_FLASHINFER_SAMPLER=0.\n"
)


def _engine(**kw) -> EngineConfig:
    base = dict(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        sampler_backend="pytorch",
    )
    base.update(kw)
    return EngineConfig(**base)


def test_parse_resolves_all_key_fields() -> None:
    eff = parse_effective_config(LOG)
    assert eff.revision == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert eff.max_model_len == 2048
    assert eff.max_num_seqs == 1
    assert eff.max_num_batched_tokens == 2048
    assert eff.enable_prefix_caching is False
    assert eff.enable_chunked_prefill is True
    assert eff.kv_cache_dtype == "auto"
    assert eff.gpu_memory_utilization == 0.85
    assert eff.sampler_backend == "pytorch"
    assert eff.generation_config == "vllm"


def test_fidelity_matches_when_config_agrees() -> None:
    eff = parse_effective_config(LOG)
    mismatches, unverified = check_fidelity(_engine(), eff)
    assert mismatches == []
    assert unverified == []


def test_fidelity_detects_prefix_caching_mismatch() -> None:
    # Requested True but the log resolved False.
    eff = parse_effective_config(LOG)
    mismatches, _ = check_fidelity(_engine(enable_prefix_caching=True), eff)
    assert any("enable_prefix_caching" in m for m in mismatches)


def test_fidelity_detects_model_and_batched_token_mismatches() -> None:
    eff = parse_effective_config(LOG)
    mismatches, _ = check_fidelity(
        _engine(model="other/model", max_num_batched_tokens=1024), eff
    )
    assert any("model" in mismatch for mismatch in mismatches)
    assert any("max_num_batched_tokens" in mismatch for mismatch in mismatches)


def test_auto_dtype_is_not_treated_as_an_explicit_resolved_dtype() -> None:
    eff = parse_effective_config(LOG)
    mismatches, unverified = check_fidelity(_engine(dtype="auto"), eff)
    assert mismatches == []
    assert "dtype" not in unverified


def test_fidelity_marks_unparsed_field_unverified() -> None:
    # A log with no resolvable max_seq_len / non-default max_model_len.
    partial = (
        "with config: model='m', revision=abc, enable_prefix_caching=False, "
        "enable_chunked_prefill=True, kv_cache_dtype=auto\n"
        "VLLM_USE_FLASHINFER_SAMPLER=0\n"
    )
    eff = parse_effective_config(partial)
    mismatches, unverified = check_fidelity(_engine(revision="abc"), eff)
    assert "max_model_len" in unverified  # requested 2048 but not resolvable
    assert mismatches == []  # unverified is not a mismatch
