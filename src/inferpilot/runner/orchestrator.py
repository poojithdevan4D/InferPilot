"""Runner orchestration: config in, one immutable result artifact out.

Glue only — it wires together the independent concerns (process management,
workload generation, measurement, aggregation, artifact storage) and classifies
the outcome as COMPLETED / FAILED / OOM / TIMEOUT. It contains no optimization,
no SLO evaluation, no GPU polling, and no decision logic.
"""

from __future__ import annotations

import asyncio
import math
import platform
import signal
import socket
import threading
import traceback as _tb
from datetime import datetime, timezone
from importlib import metadata as _md
from pathlib import Path
from time import monotonic
from typing import Callable, Optional, Sequence

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
    write_mechanism_evidence,
    write_metrics_capabilities,
    write_phases,
    write_result,
    write_telemetry,
    write_warmup,
)
from .phases import PhaseTimer
from .client import GenerationParams, run_requests, run_requests_open_loop
from .schedule import (
    BATCHED_POISSON_VERSION,
    generate_batched_poisson_offsets,
    generate_poisson_offsets,
)
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
from .load_evidence import build_load_evidence, intended_replay_digest
from .metrics_capabilities import (
    MetricRequirement,
    fetch_metrics_capability_report,
)
from .mechanism_evidence import build_mechanism_evidence
from .workload_gen import generate_workload

CommandBuilder = Callable[[int], list[str]]
RunDirectoryCallback = Callable[[Path], None]

_OOM_SIGNATURES = ("out of memory", "outofmemory", "cuda error: out of memory")


class _WarmupFailed(Exception):
    """Raised when any warm-up request fails; carries the warm-up measurements."""

    def __init__(self, measurements: list[RequestMeasurement]) -> None:
        super().__init__("warm-up request(s) failed")
        self.measurements = measurements


class _ExperimentDeadlineExceeded(Exception):
    """Internal signal used to unwind through the runner's cleanup path."""


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
        effective_sampler_backend=engine.sampler_backend
        if engine is not None
        else None,
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


def _generation_params(
    config: ExperimentConfig, request_timeout_s: float
) -> GenerationParams:
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
    metric_requirements: Optional[Sequence[MetricRequirement]] = None,
    require_metric_capabilities: bool = False,
    max_wall_time_s: Optional[float] = None,
    on_run_dir: Optional[RunDirectoryCallback] = None,
) -> ExperimentResult:
    """Run one experiment end-to-end and persist an immutable result artifact.

    ``command_builder`` maps a chosen port to the server argv; defaults to the
    real vLLM command. Tests inject a fake-server builder so no GPU/model/vLLM is
    required. ``metric_requirements`` captures an immutable name-level preflight;
    set ``require_metric_capabilities`` to stop before measurement when any
    required family is absent. Name presence still does not prove metric semantics.
    """
    if require_metric_capabilities and metric_requirements is None:
        raise ValueError(
            "required metric capabilities need explicit metric requirements"
        )
    if max_wall_time_s is not None:
        if max_wall_time_s <= 0 or not math.isfinite(max_wall_time_s):
            raise ValueError("max_wall_time_s must be finite and positive")
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("max_wall_time_s requires the main thread")
        if not hasattr(signal, "setitimer"):
            raise ValueError("max_wall_time_s requires a POSIX interval timer")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise ValueError("max_wall_time_s refuses to replace an active process timer")
    mechanism_requested = bool(
        metric_requirements
        and any(
            requirement.semantic == "recomputed_token_executions"
            for requirement in metric_requirements
        )
    )
    if mechanism_requested and config.engine.max_num_seqs is None:
        raise ValueError("mechanism evidence requires an explicit max_num_seqs")
    workload = config.workload
    env_overrides = build_server_env(config.engine)
    environment = capture_environment(config.engine, env_overrides)
    started_at = _now()

    run_dir: Path = create_run_dir(output_dir, config.experiment_id)
    if on_run_dir is not None:
        on_run_dir(run_dir)
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
    previous_alarm_handler = None
    # Set once the measured window completes; stays set even if a later
    # aggregation/finalization step fails, so phase timing keeps the real
    # measured duration on FAILED finalization paths.
    measured_duration_s: Optional[float] = None
    try:
        if max_wall_time_s is not None:

            def _deadline_handler(_signum, _frame):
                raise _ExperimentDeadlineExceeded

            previous_alarm_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, max_wall_time_s)
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
                    config,
                    environment,
                    ExperimentStatus.OOM,
                    "CudaOOM",
                    f"server OOM before readiness: {exc}",
                    started_at,
                )
            else:
                result = _failure_result(
                    config,
                    environment,
                    ExperimentStatus.TIMEOUT,
                    "ServerReadinessTimeout",
                    str(exc),
                    started_at,
                )
        except ServerStartupError as exc:
            ready = False
            status = (
                ExperimentStatus.OOM
                if _looks_like_oom(server.read_stderr())
                else ExperimentStatus.FAILED
            )
            result = _failure_result(
                config,
                environment,
                status,
                "CudaOOM" if status is ExperimentStatus.OOM else "ServerStartupError",
                str(exc),
                started_at,
            )

        if ready:
            if metric_requirements is not None:
                capability_report = fetch_metrics_capability_report(
                    server.base_url,
                    requirements=metric_requirements,
                )
                write_metrics_capabilities(run_dir, capability_report)
                if require_metric_capabilities and not capability_report.ready:
                    result = _failure_result(
                        config,
                        environment,
                        ExperimentStatus.FAILED,
                        "MetricCapabilityMismatch",
                        "required metric semantics are absent: "
                        + ", ".join(capability_report.missing_required),
                        started_at,
                    )
                    ready = False
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
                config,
                environment,
                ExperimentStatus.FAILED,
                "ConfigFidelityMismatch",
                "resolved config could not be verified: "
                + "; ".join(
                    mismatches + [f"{field}: unresolved" for field in unverified]
                ),
                started_at,
                effective_config=effective,
            )
        elif ready:
            gen = generate_workload(workload)
            params = _generation_params(config, request_timeout_s)
            concurrency = workload.max_concurrency or 1
            open_loop = workload.request_rate_qps is not None
            scheduled_offsets = None
            if open_loop:
                if workload.arrival_pattern == BATCHED_POISSON_VERSION:
                    scheduled_offsets = generate_batched_poisson_offsets(
                        workload.num_requests,
                        workload.request_rate_qps,
                        workload.effective_arrival_seed,
                        workload.burst_size,
                    )
                else:
                    scheduled_offsets = generate_poisson_offsets(
                        workload.num_requests,
                        workload.request_rate_qps,
                        workload.effective_arrival_seed,
                    )

            async def _run_all() -> tuple[
                list,
                list,
                float,
                ResourceTelemetry,
                list,
                Optional[dict],
                tuple[int, int],
                tuple[int, int],
                Optional[str],
            ]:
                t0 = monotonic()
                warmups: list[RequestMeasurement] = []
                if gen.warmup_prompts:
                    # Warm-ups are always sequential and finish before the measured
                    # arrival clock / telemetry window begin.
                    timer.mark("warmup_start")
                    warmups = await run_requests(
                        server.base_url,
                        gen.warmup_prompts,
                        params,
                        t0=t0,
                        max_concurrency=1 if open_loop else concurrency,
                        id_prefix="warmup",
                    )
                    timer.mark("warmup_end")
                    if any(not m.success for m in warmups):
                        raise _WarmupFailed(warmups)
                # Prime the counter boundary before starting the arrival clock;
                # the metrics scrape itself must not create dispatch drift.
                sampler = TelemetrySampler(server.base_url, t0=monotonic())
                arrival_origin = sampler.start()
                timer.mark_at("measured_start", arrival_origin)
                log_start = server.log_offsets()
                arrivals: Optional[dict] = None
                try:
                    if open_loop:
                        measured, dispatch = await run_requests_open_loop(
                            server.base_url,
                            gen.measured_prompts,
                            params,
                            t0=arrival_origin,
                            offsets=scheduled_offsets,
                            id_prefix="measured",
                        )
                        arrivals = {
                            "algorithm": workload.arrival_pattern,
                            "request_rate_qps": workload.request_rate_qps,
                            # `seed` keeps its historical meaning: the seed that
                            # produced these offsets (= the effective arrival seed;
                            # identical to workload.seed for legacy 0.3.0 runs).
                            "seed": workload.effective_arrival_seed,
                            "arrival_seed": workload.effective_arrival_seed,
                            "burst_size": workload.burst_size,
                            "scheduled_offsets_s": scheduled_offsets,
                            "actual_dispatch_offsets_s": dispatch,
                        }
                    else:
                        measured = await run_requests(
                            server.base_url,
                            gen.measured_prompts,
                            params,
                            t0=arrival_origin,
                            max_concurrency=concurrency,
                            id_prefix="measured",
                        )
                    m_end = monotonic()
                    timer.mark_at("measured_end", m_end)
                    log_end = server.log_offsets()
                finally:
                    telemetry = sampler.stop()
                return (
                    warmups,
                    measured,
                    (m_end - arrival_origin),
                    telemetry,
                    sampler.samples,
                    arrivals,
                    log_start,
                    log_end,
                    sampler.recompute_metric_name,
                )

            try:
                (
                    warmups,
                    measured,
                    duration_s,
                    telemetry,
                    samples,
                    arrivals,
                    log_start,
                    log_end,
                    recompute_metric_name,
                ) = asyncio.run(_run_all())
                # Measured window finished; retain its duration for phase timing
                # even if aggregation/finalization below raises.
                measured_duration_s = duration_s
            except _WarmupFailed as wf:
                # Preserve warm-up diagnostics as a separate artifact; never mix
                # warm-ups into measured aggregates.
                write_warmup(run_dir, wf.measurements)
                failed = [m for m in wf.measurements if not m.success]
                first_error = failed[0].error if failed else "unknown"
                result = _failure_result(
                    config,
                    environment,
                    ExperimentStatus.FAILED,
                    "WarmupFailure",
                    f"{len(failed)}/{len(wf.measurements)} warm-up request(s) failed "
                    f"before measured execution; first error: {first_error}",
                    started_at,
                    effective_config=effective,
                )
            else:
                timer.mark("finalize_start")
                if warmups:
                    write_warmup(run_dir, warmups)
                write_telemetry(run_dir, samples)
                if mechanism_requested:
                    stdout_slice = server.read_log_slice(
                        stdout_path, log_start[0], log_end[0]
                    )
                    stderr_slice = server.read_log_slice(
                        stderr_path, log_start[1], log_end[1]
                    )
                    evidence = build_mechanism_evidence(
                        experiment_id=config.experiment_id,
                        measured_duration_s=duration_s,
                        sample_interval_s=telemetry.sample_interval_s,
                        max_num_seqs=config.engine.max_num_seqs,
                        telemetry_samples=samples,
                        counter_metric_name=(
                            recompute_metric_name or "unobserved-recomputation-counter"
                        ),
                        stdout_slice=stdout_slice,
                        stderr_slice=stderr_slice,
                        stdout_bounds=(log_start[0], log_end[0]),
                        stderr_bounds=(log_start[1], log_end[1]),
                    )
                    write_mechanism_evidence(run_dir, evidence)
                if arrivals is not None:
                    write_arrivals(run_dir, arrivals)
                aggregates = compute_aggregates(measured, duration_s)
                aggregates = aggregates.model_copy(
                    update={"gpu_memory_peak_mb": telemetry.peak_gpu_memory_mb}
                )
                load_evidence = None
                # The load window ends at the first complete five-second
                # boundary covering the last offered arrival, excluding the
                # post-arrival drain tail from steady-state classification.
                if (
                    open_loop
                    and scheduled_offsets
                    and len(measured) >= 100
                    and duration_s >= 30
                ):
                    evidence_end = min(
                        duration_s,
                        max(30.0, math.ceil(scheduled_offsets[-1] / 5.0) * 5.0),
                    )
                    load_evidence = build_load_evidence(
                        experiment_id=config.experiment_id,
                        measured_window_t0_s=0.0,
                        measured_window_end_s=evidence_end,
                        measurements=measured,
                        telemetry_samples=samples,
                        requested_output_tokens=workload.output_tokens,
                        coverage_complete=len(measured) == workload.num_requests,
                        # Synthetic open-loop work is stationary by construction;
                        # successful warm-up excludes engine-startup transients.
                        steady_state=bool(warmups),
                        source="runner-measured-window-v1",
                        replay_sha256=intended_replay_digest(
                            gen.measured_prompts,
                            scheduled_offsets,
                            workload.output_tokens,
                        ),
                    )
                result = ExperimentResult(
                    config=config,
                    environment=environment,
                    status=ExperimentStatus.COMPLETED,
                    measurements=measured,
                    aggregates=aggregates,
                    effective_config=effective,
                    telemetry=telemetry,
                    load_evidence=load_evidence,
                    started_at=started_at,
                    finished_at=_now(),
                )
                timer.mark("finalize_end")

    except _ExperimentDeadlineExceeded:
        result = _failure_result(
            config,
            environment,
            ExperimentStatus.TIMEOUT,
            "ExperimentDeadlineExceeded",
            f"experiment exceeded the hard {max_wall_time_s:g}s wall-time budget",
            started_at,
        )
    except KeyboardInterrupt:
        if result is None:
            result = _failure_result(
                config,
                environment,
                ExperimentStatus.FAILED,
                "Interrupted",
                "run interrupted by user",
                started_at,
            )
        raise
    except Exception as exc:  # unexpected: still record a structured FAILED result
        if result is None:
            result = _failure_result(
                config,
                environment,
                ExperimentStatus.FAILED,
                type(exc).__name__,
                str(exc),
                started_at,
                traceback=_tb.format_exc(),
            )
    finally:
        if max_wall_time_s is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            assert previous_alarm_handler is not None
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        # stop() records the teardown boundary; classify errors before vs after it
        # so intentional-teardown noise is not confused with startup/measurement
        # failures. Logs are classified, never suppressed.
        timer.mark("teardown_start")
        # A stop() error must never prevent result/phase persistence or leave the
        # teardown boundary unrecorded.
        try:
            server.stop()
        except Exception:  # best-effort cleanup; boundary still recorded below
            pass
        timer.mark("teardown_end")
        try:
            write_lifecycle(run_dir, server.classify_log_errors())
        except (OSError, FileExistsError, ValueError):
            pass
        if result is not None:
            write_result(run_dir, result)
        # Phase timing is written AFTER cleanup on every terminal path, without
        # weakening always-cleanup behavior (its failure never masks the result).
        # aggregate_duration_s is bound to a completed measured window, not to a
        # COMPLETED terminal status, so it survives finalization failures.
        if timer.started:
            try:
                terminal = result.status.value if result is not None else "failed"
                write_phases(run_dir, timer.build(terminal, measured_duration_s))
            except (OSError, FileExistsError, ValueError):
                pass

    assert result is not None
    return result
