"""InferPilot Milestone-1 benchmark runner.

Vertical slice: start a vLLM server, run a fixed characterization workload,
measure raw per-request timings, aggregate, and persist one immutable result.

Concerns are separated into modules:
    server        — process management (shell-free subprocess + health poll)
    workload_gen  — deterministic prompt construction
    client        — async streaming measurement client
    aggregate     — pure percentile / aggregate computation
    artifacts     — unique run dir + immutable result JSON
    orchestrator  — glue + outcome classification (COMPLETED/FAILED/OOM/TIMEOUT)
"""

from __future__ import annotations

from .aggregate import compute_aggregates, percentile
from .client import GenerationParams, run_requests, run_requests_open_loop
from .schedule import (
    BATCHED_POISSON_VERSION,
    POISSON_VERSION,
    generate_batched_poisson_offsets,
    generate_poisson_offsets,
)
from .defaults import (
    BENCH_PYTHON_VERSION,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    PINNED_VLLM_VERSION,
)
from .effective_config import check_fidelity, parse_effective_config
from .hardware import discover_hardware, discover_toolchain
from .orchestrator import capture_environment, run_experiment
from .server import (
    ManagedServer,
    ServerReadinessTimeout,
    ServerStartupError,
    build_server_env,
    build_vllm_command,
    default_vllm_executable,
    find_free_port,
)
from .telemetry import TelemetrySampler
from .load_evidence import build_load_evidence, intended_replay_digest
from .mechanism_evidence import (
    build_mechanism_evidence,
    parse_scheduler_iterations,
    telemetry_digest,
)
from .workload_gen import GeneratedWorkload, generate_workload

__all__ = [
    "run_experiment",
    "capture_environment",
    "discover_hardware",
    "discover_toolchain",
    "build_server_env",
    "default_vllm_executable",
    "parse_effective_config",
    "check_fidelity",
    "TelemetrySampler",
    "build_load_evidence",
    "intended_replay_digest",
    "build_mechanism_evidence",
    "parse_scheduler_iterations",
    "telemetry_digest",
    "compute_aggregates",
    "percentile",
    "GenerationParams",
    "run_requests",
    "run_requests_open_loop",
    "generate_poisson_offsets",
    "generate_batched_poisson_offsets",
    "POISSON_VERSION",
    "BATCHED_POISSON_VERSION",
    "ManagedServer",
    "ServerReadinessTimeout",
    "ServerStartupError",
    "build_vllm_command",
    "find_free_port",
    "GeneratedWorkload",
    "generate_workload",
    "BENCH_PYTHON_VERSION",
    "PINNED_VLLM_VERSION",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_REVISION",
]
