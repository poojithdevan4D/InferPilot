"""Resume-safe execution driver for the preregistered C7 rate-regime study."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from inferpilot import ExperimentConfig, ExperimentResult
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment


ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs/c7-rate-regimes"
STORE_DIR = OUT_DIR / "store"
MANIFEST = OUT_DIR / "manifest.jsonl"
MAX_DRIFT_P95_MS = 10.0
MAX_DRIFT_SINGLE_MS = 250.0

# Fixed before measurement. Rate order and sequence order are deliberately
# interleaved to distribute time/thermal drift across the matrix.
EXECUTION_ORDER = (
    (6, 4, (1, 2, 3, 4)),
    (6, 6, (2, 3, 4, 1)),
    (6, 8, (3, 4, 1, 2)),
    (7, 8, (4, 3, 2, 1)),
    (7, 6, (3, 2, 1, 4)),
    (7, 4, (2, 1, 4, 3)),
    (8, 6, (2, 4, 1, 3)),
    (8, 4, (4, 1, 3, 2)),
    (8, 8, (1, 3, 2, 4)),
)


def _percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]


def _config(rate: int, seed: int, seqs: int) -> ExperimentConfig:
    path = CFG_DIR / f"qps{rate}-s{seed}-seq{seqs}.json"
    return ExperimentConfig.model_validate_json(path.read_text())


def _load_result(run_dir: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json((run_dir / "result.json").read_text())


def validate_run(result: ExperimentResult, run_dir: Path, expected_rate: int) -> dict:
    problems: list[str] = []
    if result.status.value != "completed":
        problems.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        problems.append("not_baseline_eligible")
    aggregate = result.aggregates
    counts = None if aggregate is None else (
        aggregate.num_requests, aggregate.num_successful, aggregate.num_failed
    )
    if counts != (256, 256, 0):
        problems.append(f"counts={counts}")
    effective = result.effective_config
    if effective is None or not effective.verified or effective.unverified_fields:
        problems.append("effective_not_verified")
    telemetry = result.telemetry
    if (
        telemetry is None
        or telemetry.error is not None
        or telemetry.num_samples < 1
        or telemetry.peak_gpu_memory_mb is None
    ):
        problems.append("telemetry_incomplete")

    lifecycle_path = run_dir / "lifecycle.json"
    if not lifecycle_path.is_file():
        problems.append("lifecycle_missing")
    elif json.loads(lifecycle_path.read_text()).get("pre_teardown"):
        problems.append("pre_teardown_errors")

    arrivals_path = run_dir / "arrivals.json"
    drift: dict = {}
    if not arrivals_path.is_file():
        problems.append("arrivals_missing")
    else:
        arrivals = json.loads(arrivals_path.read_text())
        scheduled = arrivals.get("scheduled_offsets_s", [])
        actual = arrivals.get("actual_dispatch_offsets_s", [])
        if len(scheduled) != 256 or len(actual) != 256:
            problems.append(f"arrival_counts={(len(scheduled), len(actual))}")
        else:
            drifts = [(a - s) * 1000 for s, a in zip(scheduled, actual)]
            p95 = _percentile(drifts, 0.95)
            maximum = max(drifts)
            drift = {
                "nominal_qps": arrivals.get("request_rate_qps"),
                "seed": arrivals.get("seed"),
                "scheduled_span_s": round(scheduled[-1], 6),
                "realized_schedule_qps": round(255 / scheduled[-1], 6),
                "realized_dispatch_qps": round(255 / actual[-1], 6),
                "drift_mean_ms": round(statistics.mean(drifts), 3),
                "drift_p95_ms": round(p95, 3) if p95 is not None else None,
                "drift_max_ms": round(maximum, 3),
            }
            if arrivals.get("seed") != result.config.workload.seed:
                problems.append("arrival_seed_mismatch")
            if arrivals.get("request_rate_qps") != float(expected_rate):
                problems.append("arrival_rate_mismatch")
            if p95 is not None and p95 > MAX_DRIFT_P95_MS:
                problems.append(f"dispatch_drift_p95_ms={p95:.3f}")
            if maximum > MAX_DRIFT_SINGLE_MS:
                problems.append(f"dispatch_drift_max_ms={maximum:.3f}")
    return {"problems": problems, "drift": drift}


def _records() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    records = []
    for line_number, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
    return records


def _accepted_ids() -> set[str]:
    accepted: set[str] = set()
    for record in _records():
        if not record.get("accepted"):
            continue
        run_dir = ROOT / record["run_dir"]
        result = _load_result(run_dir)
        check = validate_run(result, run_dir, record["nominal_qps"])
        if check["problems"]:
            raise ValueError(
                f"previously accepted run no longer validates: {run_dir}: {check['problems']}"
            )
        accepted.add(result.config.experiment_id)
    return accepted


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE_DIR)
    accepted = _accepted_ids()
    with MANIFEST.open("a") as manifest:
        for seed, rate, seq_order in EXECUTION_ORDER:
            for seqs in seq_order:
                config = _config(rate, seed, seqs)
                if config.experiment_id in accepted:
                    print(f"SKIP accepted {config.experiment_id}", flush=True)
                    continue
                before = set(OUT_DIR.glob(f"{config.experiment_id}-*"))
                print(f"RUN qps={rate} seed={seed} seqs={seqs} ({config.experiment_id})", flush=True)
                result = run_experiment(
                    config, str(OUT_DIR), ready_timeout_s=900.0, request_timeout_s=300.0
                )
                created = set(OUT_DIR.glob(f"{config.experiment_id}-*")) - before
                if len(created) != 1:
                    raise RuntimeError(
                        f"expected one new run directory for {config.experiment_id}; got {created}"
                    )
                run_dir = created.pop()
                check = validate_run(result, run_dir, rate)
                record = {
                    "experiment_id": config.experiment_id,
                    "nominal_qps": rate,
                    "seed": seed,
                    "max_num_seqs": seqs,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "status": result.status.value,
                    "eligible": result.is_baseline_eligible,
                    "accepted": not check["problems"],
                    **check,
                }
                if not check["problems"]:
                    ingested = store.ingest(run_dir)
                    record["store_run_id"] = ingested.run_id
                    record["store_created"] = ingested.created
                    accepted.add(config.experiment_id)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print(json.dumps(record, indent=2), flush=True)
                if check["problems"]:
                    print(f"ABORT: invalid cell {config.experiment_id}; evidence preserved", file=sys.stderr)
                    return 1

    expected = {
        f"c7-qps{rate}-s{seed}-seq{seqs}"
        for seed in (6, 7, 8) for rate in (4, 6, 8) for seqs in (1, 2, 3, 4)
    }
    if accepted != expected:
        raise RuntimeError(f"study incomplete; missing {sorted(expected - accepted)}")
    print("C7 COMPLETE: 36/36 cells accepted and ingested", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
