"""Pinned runtime decisions for the Milestone-1 benchmark environment.

These are the *serving-environment* pins, deliberately kept separate from the
contracts package (which stays pure-Python, no GPU/serving deps). vLLM is NOT a
project dependency; it is invoked as a subprocess and must be installed in a
dedicated Python 3.12 environment (see README).
"""

from __future__ import annotations

# Benchmark environment.
BENCH_PYTHON_VERSION = "3.12"

# Serving engine, pinned exactly.
PINNED_VLLM_VERSION = "0.29.0"

# Default model + exact Hugging Face revision (resolved from the Hub main branch).
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
