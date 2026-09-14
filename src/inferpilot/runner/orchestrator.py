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

from ..config import EngineConfig, ExperimentConfig
from ..environment import EnvironmentMetadata
from ..measurements import RequestMeasurement
from ..results import AggregateMetrics, ExperimentResult
from ..status import ExperimentStatus, FailureRecord
from ..effective import EffectiveConfig
from ..telemetry import ResourceTelemetry
from .aggregate import compute_aggregates
from .artifacts import (
    create_run_dir,
    server_log_paths,
    write_arrivals,
    write_lifecycle,
    write_phases,
    write_result,
    write_telemetry,
    write_warmup,
)
from .phases import PhaseTimer
from .client import GenerationParams, run_requests, run_requests_open_loop
from .schedule import POISSON_VERSION, generate_poisson_offsets
from .effective_config import check_fidelity, parse_effective_config
from .hardware import discover_hardware, discover_toolchain
from .server import (
    ManagedServer,
    ServerReadinessTimeout,
    ServerStartupError,
    build_server_env,
    build_vllm_command,
    find_free_port,
)
from .telemetry import TelemetrySampler
from .workload_gen import generate_workload

CommandBuilder = Callable[[int], list[str]]

_OOM_SIGNATURES = ("out of memory", "outofmemory", "cuda error: out of memory")


class _WarmupFailed(Exception):
    """Raised when any warm-up request fails; carries the warm-up measurements."""

    def __init__(self, measurements: list[RequestMeasurement]) -> None:
        super().__init__("warm-up request(s) failed")
        self.measurements = measurements


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _now() -> datetime:
    # Wall-clock, used ONLY for provenance timestamps (started/finished/occurred).
    # Never used for any duration/latency calculation — those use a monotonic clock.
    return datetime.now(timezone.utc)


def capture_environment(
    engine: Optional[EngineConfig] = None,
    runtime_overrides: Optional[dict] = None,
) -> EnvironmentMetadata:
    """One-time best-effort environment + hardware + toolchain snapshot.

    No GPU polling. ``engine``/``runtime_overrides`` record which sampler backend
    was requested and the exact env overrides applied to the server child.
    """
    return EnvironmentMetadata(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=_pkg_version("torch"),
        vllm_version=_pkg_version("vllm"),
        hardware=discover_hardware(),
        toolchain=discover_toolchain(),
        effective_sampler_backend=engine.sampler_backend if engine is not None else None,
        runtime_overrides=dict(runtime_overrides) if runtime_overrides else {},
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
    effective_config: Optional[EffectiveConfig] = None,
) -> ExperimentResult:
    return ExperimentResult(
        config=config,
        environment=environment,
        status=status,
        measurements=measurements or [],
        aggregates=aggregates,
        effective_config=effective_config,
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
    env_overrides = build_server_env(config.engine)
    environment = capture_environment(config.engine, env_overrides)
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
        env=env_overrides,
    )

    result: Optional[ExperimentResult] = None
    timer = PhaseTimer()
    try:
        timer.launch()
        server.start()

        try:
            server.wait_until_ready()
            timer.mark("server_ready")
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
            # Configuration fidelity: verify what vLLM actually resolved vs what
            # we asked for. Do not trust the input config to describe what ran.
            timer.mark("config_verify_start")
            log_text = server.read_stdout() + "\n" + server.read_stderr()
            effective = parse_effective_config(log_text)
            mismatches, unverified = check_fidelity(config.engine, effective)
            effective = effective.model_copy(
                update={
                    "unverified_fields": unverified,
                    "verified": not mismatches and not unverified,
                }
            )
            timer.mark("config_verify_end")

        if ready and (mismatches or unverified):
            # Fail BEFORE measurement — a config drift invalidates the benchmark.
            result = _failure_result(
                config, environment, ExperimentStatus.FAILED, "ConfigFidelityMismatch",
                "resolved config could not be verified: "
                + "; ".join(mismatches + [f"{field}: unresolved" for field in unverified]),
                started_at, effective_config=effective,
            )
        elif ready:
            gen = generate_workload(workload)
            params = _generation_params(config, request_timeout_s)
            concurrency = workload.max_concurrency or 1
            open_loop = workload.request_rate_qps is not None
            scheduled_offsets = (
                generate_poisson_offsets(
                    workload.num_requests, workload.request_rate_qps, workload.seed
                )
                if open_loop
                else None
            )

            async def _run_all() -> tuple[list, list, float, ResourceTelemetry, list, Optional[dict]]:
                t0 = monotonic()
                warmups: list[RequestMeasurement] = []
                if gen.warmup_prompts:
                    # Warm-ups are always sequential and finish before the measured
                    # arrival clock / telemetry window begin.
                    timer.mark("warmup_start")
                    warmups = await run_requests(
                        server.base_url, gen.warmup_prompts, params,
                        t0=t0, max_concurrency=1 if open_loop else concurrency,
                        id_prefix="warmup",
                    )
                    timer.mark("warmup_end")
                    if any(not m.success for m in warmups):
                        raise _WarmupFailed(warmups)
                # Measured window (telemetry + arrival origin) starts here.
                arrival_origin = monotonic()
                timer.mark_at("measured_start", arrival_origin)
                sampler = TelemetrySampler(server.base_url, t0=arrival_origin)
                arrivals: Optional[dict] = None
                try:
                    sampler.start()
                    if open_loop:
                        measured, dispatch = await run_requests_open_loop(
                            server.base_url, gen.measured_prompts, params,
                            t0=arrival_origin, offsets=scheduled_offsets,
                            id_prefix="measured",
                        )
                        arrivals = {
                            "algorithm": POISSON_VERSION,
                            "request_rate_qps": workload.request_rate_qps,
                            "seed": workload.seed,
                            "scheduled_offsets_s": scheduled_offsets,
                            "actual_dispatch_offsets_s": dispatch,
                        }
                    else:
                        measured = await run_requests(
                            server.base_url, gen.measured_prompts, params,
                            t0=arrival_origin, max_concurrency=concurrency,
                            id_prefix="measured",
                        )
                    m_end = monotonic()
                    timer.mark_at("measured_end", m_end)
                finally:
                    telemetry = sampler.stop()
                return (
                    warmups, measured, (m_end - arrival_origin),
                    telemetry, sampler.samples, arrivals,
                )

            try:
                warmups, measured, duration_s, telemetry, samples, arrivals = asyncio.run(_run_all())
            except _WarmupFailed as wf:
                # Preserve warm-up diagnostics as a separate artifact; never mix
                # warm-ups into measured aggregates.
                write_warmup(run_dir, wf.measurements)
                failed = [m for m in wf.measurements if not m.success]
                first_error = failed[0].error if failed else "unknown"
                result = _failure_result(
                    config, environment, ExperimentStatus.FAILED, "WarmupFailure",
                    f"{len(failed)}/{len(wf.measurements)} warm-up request(s) failed "
                    f"before measured execution; first error: {first_error}",
                    started_at, effective_config=effective,
                )
            else:
                timer.mark("finalize_start")
                if warmups:
                    write_warmup(run_dir, warmups)
                write_telemetry(run_dir, samples)
                if arrivals is not None:
                    write_arrivals(run_dir, arrivals)
                aggregates = compute_aggregates(measured, duration_s)
                aggregates = aggregates.model_copy(
                    update={"gpu_memory_peak_mb": telemetry.peak_gpu_memory_mb}
                )
                result = ExperimentResult(
                    config=config,
                    environment=environment,
                    status=ExperimentStatus.COMPLETED,
                    measurements=measured,
                    aggregates=aggregates,
                    effective_config=effective,
                    telemetry=telemetry,
                    started_at=started_at,
                    finished_at=_now(),
                )
                timer.mark("finalize_end")

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
        # stop() records the teardown boundary; classify errors before vs after it
        # so intentional-teardown noise is not confused with startup/measurement
        # failures. Logs are classified, never suppressed.
        timer.mark("teardown_start")
        server.stop()
        timer.mark("teardown_end")
        try:
            write_lifecycle(run_dir, server.classify_log_errors())
        except (OSError, FileExistsError):
            pass
        if result is not None:
            write_result(run_dir, result)
        # Phase timing is written AFTER cleanup on every terminal path, without
        # weakening always-cleanup behavior (its failure never masks the result).
        if timer.started:
            try:
                agg_duration = (
                    result.aggregates.duration_s
                    if result is not None
                    and result.status is ExperimentStatus.COMPLETED
                    and result.aggregates is not None
                    else None
                )
                terminal = result.status.value if result is not None else "failed"
                write_phases(run_dir, timer.build(terminal, agg_duration))
            except (OSError, FileExistsError, ValueError):
                pass

    assert result is not None
    return result
