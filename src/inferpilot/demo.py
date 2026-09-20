"""Deterministic, GPU-free product demo.

The fixture is synthetic and intentionally lives outside the scientific evidence
corpus.  It exercises the same public contracts and decision path as a real run.
"""

from __future__ import annotations

import hashlib

from .advisor.capacity_advisory import OperatorEconomics
from .config import EngineConfig, ExperimentConfig, SLO
from .effective import EffectiveConfig
from .environment import EnvironmentMetadata, HardwareInfo
from .evidence_card import OptimizationEvidenceCard, build_evidence_card
from .measurements import RequestMeasurement
from .results import ExperimentResult
from .runner.aggregate import compute_aggregates
from .saturation import LoadEvidence, measurement_digest
from .status import ExperimentStatus
from .telemetry import ResourceTelemetry
from .workload import WorkloadSpec


def _requests() -> list[RequestMeasurement]:
    rows: list[RequestMeasurement] = []
    for index in range(120):
        # Three arrivals per second, with a deliberately growing completion delay.
        # TTFT is flat: the aligned request census, not a latency-shape guess,
        # establishes the growing backlog.
        e2e_ms = 700.0 + index * 100.0
        rows.append(
            RequestMeasurement(
                request_id=f"demo-{index:03d}",
                prompt_tokens=128,
                output_tokens=32,
                start_time_s=index / 3,
                end_time_s=index / 3 + e2e_ms / 1000,
                success=True,
                ttft_ms=100.0,
                tpot_ms=(e2e_ms - 100.0) / 31,
                e2e_latency_ms=e2e_ms,
            )
        )
    return rows


def build_demo_card() -> OptimizationEvidenceCard:
    """Build one synthetic card through the real diagnosis/recommendation path."""

    rows = _requests()
    workload = WorkloadSpec(
        name="synthetic-kv-pressure-demo",
        num_requests=len(rows),
        prompt_tokens=128,
        output_tokens=32,
        request_rate_qps=3.0,
        seed=7,
    )
    config = ExperimentConfig(
        experiment_id="inferpilot-synthetic-quickstart",
        name="InferPilot synthetic quickstart",
        description="GPU-free demonstration only; not empirical evidence.",
        engine=EngineConfig(
            model="synthetic/model",
            revision="synthetic-revision",
            max_num_seqs=4,
            max_num_batched_tokens=512,
            kv_cache_dtype="auto",
        ),
        workload=workload,
        tags=["synthetic-demo", "no-gpu"],
    )
    duration_s = max(row.end_time_s for row in rows)
    aggregates = compute_aggregates(rows, duration_s).model_copy(
        update={"gpu_memory_peak_mb": 20_000}
    )
    telemetry = ResourceTelemetry(
        sample_interval_s=1.0,
        num_samples=50,
        peak_gpu_memory_mb=20_000,
        gpu_utilization_mean_pct=96.0,
        gpu_utilization_peak_pct=100.0,
        kv_cache_usage_mean_perc=0.96,
        kv_cache_usage_peak_perc=0.99,
        preemptions_total=4,
    )
    boundaries = [0.0, 10.0, 20.0, 30.0, 40.0]
    offered = [
        sum(
            row.output_tokens
            for row in rows
            if begin <= row.start_time_s < end
        )
        for begin, end in zip(boundaries, boundaries[1:])
    ]
    delivered = [
        sum(
            row.output_tokens
            for row in rows
            if begin <= row.end_time_s < end
        )
        for begin, end in zip(boundaries, boundaries[1:])
    ]
    load = LoadEvidence(
        experiment_id=config.experiment_id,
        measurement_sha256=measurement_digest(rows),
        replay_sha256=hashlib.sha256(b"inferpilot-synthetic-quickstart-v1").hexdigest(),
        source="inferpilot-synthetic-demo-v1",
        boundaries_s=boundaries,
        coverage_complete=True,
        steady_state=True,
        offered_output_tokens=offered,
        delivered_output_tokens=delivered,
        gpu_utilization_mean_pct=96.0,
        kv_cache_usage_peak_perc=0.99,
        preemptions=4,
    )
    result = ExperimentResult(
        config=config,
        environment=EnvironmentMetadata(
            hostname="synthetic-demo",
            platform="synthetic",
            python_version="synthetic",
            vllm_version="synthetic",
            hardware=HardwareInfo(
                gpu_name="Synthetic 24 GB GPU",
                gpu_count=1,
                gpu_memory_total_mb=24_576,
            ),
        ),
        status=ExperimentStatus.COMPLETED,
        measurements=rows,
        aggregates=aggregates,
        effective_config=EffectiveConfig(
            model=config.engine.model,
            revision=config.engine.revision,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            kv_cache_dtype="auto",
            source="synthetic-demo",
            verified=True,
        ),
        telemetry=telemetry,
        load_evidence=load,
    )
    return build_evidence_card(
        result,
        SLO(ttft_p95_ms=500, tpot_p95_ms=50),
        OperatorEconomics(gpu_cost_per_hour_usd=1.0, target_qps=3.0),
        total_occupancy_s=60.0,
    )
