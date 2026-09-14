"""Resume-safe driver for the preregistered C6 GPU study.

Runs one observation for every cell in the five-seed by four-candidate design,
validates it before acceptance, and ingests eligible bundles into the local
content-addressed store. Raw evidence remains under ``runs/`` and is gitignored.
"""

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
OUT_DIR = ROOT / "runs/c6-openloop-blocked"
STORE_DIR = OUT_DIR / "store"
MANIFEST = OUT_DIR / "manifest.jsonl"

# The first four rows are a Latin square; the fifth is a fixed reverse row.
# This order was committed before measurement and must not be changed in response
# to intermediate results.
EXECUTION_ORDER = (
    (1, (1, 2, 3, 4)),
    (2, (2, 3, 4, 1)),
    (3, (3, 4, 1, 2)),
    (4, (4, 1, 2, 3)),
    (5, (4, 3, 2, 1)),
)

MAX_DRIFT_P95_MS = 10.0
MAX_DRIFT_SINGLE_MS = 250.0


def _percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))
    return ordered[index]


def _config(seed: int, seqs: int) -> ExperimentConfig:
    path = CFG_DIR / f"s{seed}-seq{seqs}.json"
    return ExperimentConfig.model_validate_json(path.read_text())


def _load_result(run_dir: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json((run_dir / "result.json").read_text())


def validate_run(result: ExperimentResult, run_dir: Path) -> dict:
    problems: list[str] = []
    if result.status.value != "completed":
        problems.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        problems.append("not_baseline_eligible")

    aggregate = result.aggregates
    counts = None if aggregate is None else (
        aggregate.num_requests,
        aggregate.num_successful,
        aggregate.num_failed,
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
    else:
        lifecycle = json.loads(lifecycle_path.read_text())
        if lifecycle.get("pre_teardown"):
            problems.append(f"pre_teardown_errors={len(lifecycle['pre_teardown'])}")

    arrivals_path = run_dir / "arrivals.json"
    drift: dict = {}
    if not arrivals_path.is_file():
        problems.append("arrivals_missing")
    else:
        arrivals = json.loads(arrivals_path.read_text())
        scheduled = arrivals.get("scheduled_offsets_s", [])
        actual = arrivals.get("actual_dispatch_offsets_s", [])
        if len(scheduled) != 256 or len(actual) != len(scheduled):
            problems.append(f"arrival_counts={(len(scheduled), len(actual))}")
        else:
            drifts = [(observed - expected) * 1000 for expected, observed in zip(scheduled, actual)]
            scheduled_span = scheduled[-1]
            actual_span = actual[-1]
            p95 = _percentile(drifts, 0.95)
            maximum = max(drifts)
            drift = {
                "nominal_qps": arrivals.get("request_rate_qps"),
                "seed": arrivals.get("seed"),
                "n": len(scheduled),
                "scheduled_span_s": round(scheduled_span, 6),
                "realized_schedule_qps": round(255 / scheduled_span, 6),
                "realized_dispatch_qps": round(255 / actual_span, 6),
                "drift_mean_ms": round(statistics.mean(drifts), 3),
                "drift_p95_ms": round(p95, 3) if p95 is not None else None,
                "drift_max_ms": round(maximum, 3),
            }
            if arrivals.get("seed") != result.config.workload.seed:
                problems.append("arrival_seed_mismatch")
            if arrivals.get("request_rate_qps") != 8.0:
                problems.append("arrival_rate_mismatch")
            if p95 is not None and p95 > MAX_DRIFT_P95_MS:
                problems.append(f"dispatch_drift_p95_ms={p95:.3f}")
            if maximum > MAX_DRIFT_SINGLE_MS:
                problems.append(f"dispatch_drift_max_ms={maximum:.3f}")

    return {"problems": problems, "drift": drift}


def _manifest_records() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    records: list[dict] = []
    for line_number, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
    return records


def _accepted_ids() -> set[str]:
    accepted: set[str] = set()
    for record in _manifest_records():
        if not record.get("accepted"):
            continue
        run_dir = Path(record["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        result = _load_result(run_dir)
        check = validate_run(result, run_dir)
        if check["problems"]:
            raise ValueError(
                f"previously accepted run no longer validates: {run_dir}: {check['problems']}"
            )
        accepted.add(result.config.experiment_id)
    return accepted


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE_DIR)
    accepted = _accepted_ids()

    with MANIFEST.open("a") as manifest:
        for seed, sequence_order in EXECUTION_ORDER:
            for seqs in sequence_order:
                config = _config(seed, seqs)
                if config.experiment_id in accepted:
                    print(f"SKIP accepted {config.experiment_id}", flush=True)
                    continue

                before = set(OUT_DIR.glob(f"{config.experiment_id}-*"))
                print(
                    f"RUN seed={seed} max_num_seqs={seqs} ({config.experiment_id})",
                    flush=True,
                )
                result = run_experiment(
                    config,
                    str(OUT_DIR),
                    ready_timeout_s=900.0,
                    request_timeout_s=300.0,
                )
                created = set(OUT_DIR.glob(f"{config.experiment_id}-*")) - before
                if len(created) != 1:
                    raise RuntimeError(
                        f"expected exactly one new run directory for {config.experiment_id}; "
                        f"found {sorted(str(path) for path in created)}"
                    )
                run_dir = created.pop()
                check = validate_run(result, run_dir)
                record = {
                    "experiment_id": config.experiment_id,
                    "seed": seed,
                    "max_num_seqs": seqs,
                    "run_dir": _relative(run_dir),
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
                    print(
                        f"ABORT: {config.experiment_id} invalid; evidence preserved",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1

    expected = {f"c6-s{seed}-seq{seqs}" for seed, _ in EXECUTION_ORDER for seqs in range(1, 5)}
    if accepted != expected:
        missing = sorted(expected - accepted)
        raise RuntimeError(f"study incomplete; missing accepted cells: {missing}")
    print("C6 COMPLETE: 20/20 cells accepted and ingested", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
