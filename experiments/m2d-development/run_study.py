"""Resume-safe runner for the preregistered M2D development pilot."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from inferpilot import ExperimentConfig, ExperimentResult, RunnerPhaseTiming
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment

from generate_configs import CANDIDATES, experiment_id


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/m2d-development"
STORE = OUT / "store"
MANIFEST = OUT / "manifest.jsonl"
MAX_DRIFT_P95_MS = 10.0
MAX_DRIFT_SINGLE_MS = 250.0

# Fixed before measurement; candidate orders are deliberately varied.
EXECUTION_ORDER = (
    (10, "prefill", CANDIDATES),
    (10, "decode", tuple(reversed(CANDIDATES))),
    (10, "burst", CANDIDATES[1:] + CANDIDATES[:1]),
    (11, "burst", tuple(reversed(CANDIDATES))),
    (11, "prefill", CANDIDATES[2:] + CANDIDATES[:2]),
    (11, "decode", CANDIDATES),
    (12, "decode", CANDIDATES[1:] + CANDIDATES[:1]),
    (12, "burst", CANDIDATES[2:] + CANDIDATES[:2]),
    (12, "prefill", tuple(reversed(CANDIDATES))),
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _load(run_dir: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json((run_dir / "result.json").read_text())


def validate_run(result: ExperimentResult, run_dir: Path) -> dict:
    problems: list[str] = []
    aggregate = result.aggregates
    if result.status.value != "completed":
        problems.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        problems.append("not_baseline_eligible")
    if aggregate is None or (
        aggregate.num_requests, aggregate.num_successful, aggregate.num_failed
    ) != (128, 128, 0):
        problems.append("request_counts_invalid")
    if result.effective_config is None or not result.effective_config.verified:
        problems.append("effective_not_verified")
    if result.telemetry is None or result.telemetry.error is not None:
        problems.append("telemetry_incomplete")

    lifecycle_path = run_dir / "lifecycle.json"
    if not lifecycle_path.is_file() or json.loads(lifecycle_path.read_text()).get("pre_teardown"):
        problems.append("lifecycle_invalid")
    phases_path = run_dir / "phases.json"
    if not phases_path.is_file():
        problems.append("phases_missing")
    else:
        try:
            timing = RunnerPhaseTiming.model_validate_json(phases_path.read_text())
            if not all(
                span.completed for span in timing.phases
                if span.name in {"server_startup", "measured_window", "teardown", "total_occupancy"}
            ):
                problems.append("phases_incomplete")
        except ValueError:
            problems.append("phases_invalid")

    drift = {}
    arrivals_path = run_dir / "arrivals.json"
    if not arrivals_path.is_file():
        problems.append("arrivals_missing")
    else:
        arrivals = json.loads(arrivals_path.read_text())
        scheduled = arrivals.get("scheduled_offsets_s", [])
        actual = arrivals.get("actual_dispatch_offsets_s", [])
        if len(scheduled) != 128 or len(actual) != 128:
            problems.append("arrival_counts_invalid")
        else:
            drifts = [(a - s) * 1000 for s, a in zip(scheduled, actual)]
            p95, maximum = _percentile(drifts, 0.95), max(drifts)
            drift = {
                "algorithm": arrivals.get("algorithm"),
                "arrival_seed": arrivals.get("arrival_seed"),
                "burst_size": arrivals.get("burst_size"),
                "scheduled_span_s": round(scheduled[-1], 6),
                "drift_mean_ms": round(statistics.mean(drifts), 3),
                "drift_p95_ms": round(p95, 3), "drift_max_ms": round(maximum, 3),
            }
            workload = result.config.workload
            if arrivals.get("algorithm") != workload.arrival_pattern:
                problems.append("arrival_algorithm_mismatch")
            if arrivals.get("arrival_seed") != workload.effective_arrival_seed:
                problems.append("arrival_seed_mismatch")
            if arrivals.get("burst_size") != workload.burst_size:
                problems.append("burst_size_mismatch")
            if p95 > MAX_DRIFT_P95_MS or maximum > MAX_DRIFT_SINGLE_MS:
                problems.append("dispatch_drift_exceeded")
    return {"problems": problems, "drift": drift}


def _records() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    return [json.loads(line) for line in MANIFEST.read_text().splitlines()]


def _accepted() -> set[str]:
    accepted: set[str] = set()
    for record in _records():
        if record.get("accepted"):
            run_dir = ROOT / record["run_dir"]
            check = validate_run(_load(run_dir), run_dir)
            if check["problems"]:
                raise ValueError(f"accepted run no longer validates: {run_dir}: {check['problems']}")
            accepted.add(record["experiment_id"])
    return accepted


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE)
    accepted = _accepted()
    with MANIFEST.open("a") as manifest:
        for arrival_seed, shape, order in EXECUTION_ORDER:
            for seqs, tokens in order:
                eid = experiment_id(shape, arrival_seed, seqs, tokens)
                if eid in accepted:
                    print(f"SKIP accepted {eid}", flush=True)
                    continue
                config = ExperimentConfig.model_validate_json((HERE / f"{eid}.json").read_text())
                before = set(OUT.glob(f"{eid}-*"))
                print(f"RUN {eid}", flush=True)
                result = run_experiment(
                    config, str(OUT), ready_timeout_s=900.0, request_timeout_s=600.0
                )
                created = set(OUT.glob(f"{eid}-*")) - before
                if len(created) != 1:
                    raise RuntimeError(f"expected one run directory for {eid}; got {created}")
                run_dir = created.pop()
                check = validate_run(result, run_dir)
                record = {
                    "experiment_id": eid, "shape": shape, "arrival_seed": arrival_seed,
                    "max_num_seqs": seqs, "max_num_batched_tokens": tokens,
                    "run_dir": str(run_dir.relative_to(ROOT)), "status": result.status.value,
                    "eligible": result.is_baseline_eligible,
                    "accepted": not check["problems"], **check,
                }
                if not check["problems"]:
                    ingested = store.ingest(run_dir)
                    record.update(store_run_id=ingested.run_id, store_created=ingested.created)
                    accepted.add(eid)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print(json.dumps(record, indent=2), flush=True)
                if check["problems"]:
                    print(f"ABORT invalid cell {eid}; evidence preserved", file=sys.stderr)
                    return 1
    expected = {
        experiment_id(shape, seed, seqs, tokens)
        for seed in (10, 11, 12) for shape in ("prefill", "decode", "burst")
        for seqs, tokens in CANDIDATES
    }
    if accepted != expected:
        raise RuntimeError(f"study incomplete; missing={sorted(expected - accepted)}")
    print("M2D DEVELOPMENT COMPLETE: 36/36 accepted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
