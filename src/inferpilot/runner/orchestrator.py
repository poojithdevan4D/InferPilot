"""Runner orchestration: config in, one immutable result artifact out.

Glue only — it wires together the independent concerns (process management,
workload generation, measurement, aggregation, artifact storage) and classifies
the outcome as COMPLETED / FAILED / OOM / TIMEOUT. It contains no optimization,
no SLO evaluation, no GPU polling, and no decision logic.
"""

from __future__ import annotations

import asyncio
import platform
import socket
import traceback as _tb
from datetime import datetime, timezone
from importlib import metadata as _md
from pathlib import Path
from time import monotonic
from typing import Callable, Optional

from ..config import ExperimentConfig
from ..environment import EnvironmentMetadata, HardwareInfo
from ..results import AggregateMetrics, ExperimentResult
from ..status import ExperimentStatus, FailureRecord
from .aggregate import compute_aggregates
from .artifacts import create_run_dir, server_log_paths, write_result
from .client import GenerationParams, run_requests
from .server import (
    ManagedServer,
    ServerReadinessTimeout,
    ServerStartupError,
    build_vllm_command,
    find_free_port,
)
from .workload_gen import generate_workload

CommandBuilder = Callable[[int], list[str]]

_OOM_SIGNATURES = ("out of memory", "outofmemory", "cuda error: out of memory")


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _now() -> datetime:
    # Wall-clock, used ONLY for provenance timestamps (started/finished/occurred).
    # Never used for any duration/latency calculation — those use a monotonic clock.
    return datetime.now(timezone.utc)


def capture_environment() -> EnvironmentMetadata:
    """Best-effort environment snapshot. No GPU polling (out of scope)."""
    return EnvironmentMetadata(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=_pkg_version("torch"),
        vllm_version=_pkg_version("vllm"),
        hardware=HardwareInfo(),  # left unpopulated; discovery is future work
        captured_at=_now(),
    )


def _looks_like_oom(server_stderr: str) -> bool:
    text = server_stderr.lower()
    return any(sig in text for sig in _OOM_SIGNATURES)


def _failure_result(
    config: ExperimentConfig,
    environment: EnvironmentMetadata,
    status: ExperimentStatus,
    error_type: str,
    message: str,
    started_at: datetime,
    *,
    traceback: Optional[str] = None,
    measurements: Optional[list] = None,
    aggregates: Optional[AggregateMetrics] = None,
) -> ExperimentResult:
    return ExperimentResult(
        config=config,
        environment=environment,
        status=status,
        measurements=measurements or [],
        aggregates=aggregates,
        failure=FailureRecord(
            status=status,
            error_type=error_type,
            message=message,
            traceback=traceback,
            occurred_at=_now(),
        ),
        started_at=started_at,
        finished_at=_now(),
    )


def _generation_params(config: ExperimentConfig, request_timeout_s: float) -> GenerationParams:
    engine = config.engine
    workload = config.workload
    return GenerationParams(
        model=engine.model,
        max_tokens=workload.output_tokens,
        temperature=workload.temperature,
        seed=workload.seed,
        ignore_eos=workload.ignore_eos,
        min_tokens=workload.output_tokens if workload.ignore_eos else None,
        request_timeout_s=request_timeout_s,
    )


def run_experiment(
    config: ExperimentConfig,
    output_dir: str = "runs",
    *,
    command_builder: Optional[CommandBuilder] = None,
    host: str = "127.0.0.1",
    ready_timeout_s: float = 300.0,
    request_timeout_s: float = 120.0,
    terminate_timeout_s: float = 15.0,
) -> ExperimentResult:
    """Run one experiment end-to-end and persist an immutable result artifact.

    ``command_builder`` maps a chosen port to the server argv; defaults to the
    real vLLM command. Tests inject a fake-server builder so no GPU/model/vLLM is
    required.
    """
    workload = config.workload
    environment = capture_environment()
    started_at = _now()

    run_dir: Path = create_run_dir(output_dir, config.experiment_id)
    stdout_path, stderr_path = server_log_paths(run_dir)

    port = find_free_port(host)
    builder = command_builder or (lambda p: build_vllm_command(config.engine, p, host))
    command = builder(port)

    server = ManagedServer(
        command,
        host=host,
        port=port,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        ready_timeout_s=ready_timeout_s,
        terminate_timeout_s=terminate_timeout_s,
    )

    result: Optional[ExperimentResult] = None
    try:
        server.start()

        try:
            server.wait_until_ready()
            ready = True
        except ServerReadinessTimeout as exc:
            ready = False
            if _looks_like_oom(server.read_stderr()):
                result = _failure_result(
                    config, environment, ExperimentStatus.OOM, "CudaOOM",
                    f"server OOM before readiness: {exc}", started_at,
                )
            else:
                result = _failure_result(
                    config, environment, ExperimentStatus.TIMEOUT, "ServerReadinessTimeout",
                    str(exc), started_at,
                )
        except ServerStartupError as exc:
            ready = False
            status = ExperimentStatus.OOM if _looks_like_oom(server.read_stderr()) else ExperimentStatus.FAILED
            result = _failure_result(
                config, environment, status,
                "CudaOOM" if status is ExperimentStatus.OOM else "ServerStartupError",
                str(exc), started_at,
            )

        if ready:
            gen = generate_workload(workload)
            params = _generation_params(config, request_timeout_s)
            concurrency = workload.max_concurrency or 1

            async def _run_all() -> tuple[list, float]:
                t0 = monotonic()
                if gen.warmup_prompts:
                    await run_requests(
                        server.base_url, gen.warmup_prompts, params,
                        t0=t0, max_concurrency=concurrency, id_prefix="warmup",
                    )
                m_start = monotonic()
                measured = await run_requests(
                    server.base_url, gen.measured_prompts, params,
                    t0=t0, max_concurrency=concurrency, id_prefix="measured",
                )
                m_end = monotonic()
                return measured, (m_end - m_start)

            measured, duration_s = asyncio.run(_run_all())
            aggregates = compute_aggregates(measured, duration_s)
            result = ExperimentResult(
                config=config,
                environment=environment,
                status=ExperimentStatus.COMPLETED,
                measurements=measured,
                aggregates=aggregates,
                started_at=started_at,
                finished_at=_now(),
            )

    except KeyboardInterrupt:
        if result is None:
            result = _failure_result(
                config, environment, ExperimentStatus.FAILED, "Interrupted",
                "run interrupted by user", started_at,
            )
        raise
    except Exception as exc:  # unexpected: still record a structured FAILED result
        if result is None:
            result = _failure_result(
                config, environment, ExperimentStatus.FAILED, type(exc).__name__,
                str(exc), started_at, traceback=_tb.format_exc(),
            )
    finally:
        server.stop()
        if result is not None:
            write_result(run_dir, result)

    assert result is not None
    return result
