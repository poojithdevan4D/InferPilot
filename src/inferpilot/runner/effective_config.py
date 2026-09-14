"""Parse vLLM's resolved configuration from startup logs and check fidelity.

We do not trust the input config to describe what actually ran. vLLM logs both a
``non-default args: {...}`` dict and a resolved ``... with config: ...`` line at
startup; this module parses those into an :class:`EffectiveConfig` and compares
the resolved values against the explicitly-requested ones.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from ..config import EngineConfig
from ..effective import EffectiveConfig

# Keys we verify (requested -> resolved). Order defines report order.
FIDELITY_KEYS = (
    "model",
    "revision",
    "dtype",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "sampler_backend",
    "kv_cache_dtype",
    "gpu_memory_utilization",
    "generation_config",
)


def _search(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _parse_non_default_args(text: str) -> dict:
    m = re.search(r"non-default args:\s*(\{.*\})", text)
    if not m:
        return {}
    try:
        value = ast.literal_eval(m.group(1))
        return value if isinstance(value, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _detect_sampler_backend(text: str) -> Optional[str]:
    if "VLLM_USE_FLASHINFER_SAMPLER=0" in text:
        return "pytorch"
    if "VLLM_USE_FLASHINFER_SAMPLER=1" in text:
        return "flashinfer"
    return None


def parse_effective_config(log_text: str) -> EffectiveConfig:
    """Best-effort parse of the resolved config from vLLM startup logs."""
    args = _parse_non_default_args(log_text)

    # Resolved values from the "with config:" line (targeted regexes avoid the
    # comma-splitting hazard of that very long line).
    resolved_pc = _search(r"enable_prefix_caching=(True|False)", log_text)
    resolved_cp = _search(r"enable_chunked_prefill=(True|False)", log_text)
    kv_dtype = _search(r"kv_cache_dtype=([^\s,]+)", log_text) or args.get("kv_cache_dtype")
    max_seq = _search(r"max_seq_len=(\d+)", log_text)
    revision = args.get("revision") or _search(r"revision=([0-9A-Za-z._-]+)", log_text)

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if v == "True":
            return True
        if v == "False":
            return False
        return None

    gpu_util = args.get("gpu_memory_utilization")
    if gpu_util is None:
        g = _search(r"Desired GPU memory utilization is \(([0-9.]+)", log_text)
        gpu_util = float(g) if g else None

    kwargs = dict(
        model=args.get("model") or _search(r"served_model_name=([^\s,]+)", log_text),
        revision=revision,
        dtype=_search(r"dtype=torch\.([^\s,]+)", log_text) or args.get("dtype"),
        max_model_len=int(max_seq) if max_seq else args.get("max_model_len"),
        max_num_seqs=args.get("max_num_seqs"),
        max_num_batched_tokens=args.get("max_num_batched_tokens"),
        enable_prefix_caching=_to_bool(resolved_pc) if resolved_pc is not None else _to_bool(args.get("enable_prefix_caching")),
        enable_chunked_prefill=_to_bool(resolved_cp) if resolved_cp is not None else _to_bool(args.get("enable_chunked_prefill")),
        sampler_backend=_detect_sampler_backend(log_text),
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=float(gpu_util) if gpu_util is not None else None,
        generation_config=args.get("generation_config"),
    )
    try:
        return EffectiveConfig(**kwargs)
    except Exception:
        # A malformed parse must not crash the run; record nothing rather than guess.
        return EffectiveConfig(source="startup_log_parse_error")


def _requested(engine: EngineConfig) -> dict:
    """Explicitly-requested values (None/'auto' => not explicitly configured)."""
    req = {
        "model": engine.model,
        "revision": engine.revision,
        "dtype": engine.dtype,
        "max_model_len": engine.max_model_len,
        "max_num_seqs": engine.max_num_seqs,
        "max_num_batched_tokens": engine.max_num_batched_tokens,
        "enable_prefix_caching": engine.enable_prefix_caching,
        "enable_chunked_prefill": engine.enable_chunked_prefill,
        "kv_cache_dtype": engine.kv_cache_dtype,
        "gpu_memory_utilization": engine.gpu_memory_utilization,
        "sampler_backend": engine.sampler_backend,
        "generation_config": engine.generation_config,
    }
    # sampler_backend "auto" means "engine default" => not explicitly configured.
    if req["sampler_backend"] == "auto":
        req["sampler_backend"] = None
    if req["dtype"] == "auto":
        req["dtype"] = None
    return req


def check_fidelity(
    engine: EngineConfig, effective: EffectiveConfig
) -> tuple[list[str], list[str]]:
    """Compare requested vs resolved.

    Returns ``(mismatches, unverified)``. ``mismatches`` are explicitly-requested
    values that differ from the resolved value; ``unverified`` are requested keys
    whose resolved value could not be parsed (so cannot be confirmed).
    """
    requested = _requested(engine)
    mismatches: list[str] = []
    unverified: list[str] = []

    for key in FIDELITY_KEYS:
        want = requested.get(key)
        if want is None:  # not explicitly configured -> nothing to verify
            continue
        got = getattr(effective, key)
        if got is None:
            unverified.append(key)
            continue
        if key == "gpu_memory_utilization":
            if abs(float(want) - float(got)) > 1e-6:
                mismatches.append(f"{key}: requested={want} resolved={got}")
        elif key == "kv_cache_dtype":
            if str(want).lower() != str(got).lower():
                mismatches.append(f"{key}: requested={want} resolved={got}")
        else:
            if want != got:
                mismatches.append(f"{key}: requested={want} resolved={got}")

    return mismatches, unverified
