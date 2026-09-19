"""Five-minute, GPU-free demonstration of InferPilot's fail-closed load contract."""

from __future__ import annotations

from inferpilot import RequestMeasurement, ResourceSample
from inferpilot.runner import build_load_evidence
from inferpilot.saturation import assess_load_state


def measurements(*, overloaded: bool) -> list[RequestMeasurement]:
    rows = []
    for index in range(200):
        start = index / 2
        latency = 0.5 + (index * 0.25 if overloaded else 0)
        rows.append(RequestMeasurement(
            request_id=str(index), prompt_tokens=128, output_tokens=32,
            start_time_s=start, end_time_s=start + latency,
            ttft_ms=100, tpot_ms=(latency * 1000 - 100) / 31,
            e2e_latency_ms=latency * 1000, success=True,
        ))
    return rows


def evidence(rows: list[RequestMeasurement]):
    samples = [ResourceSample(
        t_s=float(second), gpu_utilization_pct=80, kv_cache_usage_perc=.5,
        num_requests_waiting=0, num_requests_running=0,
        num_preemptions_total=0,
    ) for second in range(101)]
    return build_load_evidence(
        experiment_id="demo", measured_window_t0_s=0,
        measured_window_end_s=100, measurements=rows,
        telemetry_samples=samples, requested_output_tokens=32,
        coverage_complete=True, steady_state=True,
    )


def main() -> int:
    print("InferPilot aligned-load demo (synthetic, deterministic, no GPU)\n")
    for label, overloaded in (("keeps up", False), ("persistent backlog", True)):
        rows = measurements(overloaded=overloaded)
        assessment = assess_load_state(rows, evidence=evidence(rows))
        print(f"{label:20} -> {assessment.state:12}  {', '.join(assessment.reasons)}")
    rows = measurements(overloaded=False)
    missing = assess_load_state(rows)
    print(f"{'missing evidence':20} -> {missing.state:12}  {', '.join(missing.reasons)}")
    print("\nThe system calls overload only from persistent conserved backlog growth;")
    print("without aligned evidence it abstains instead of guessing from GPU/KV snapshots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
