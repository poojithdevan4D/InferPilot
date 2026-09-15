"""Resume-safe executor for the sealed M2D held-out corpus."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

from inferpilot import ExperimentConfig, ExperimentResult, RunnerPhaseTiming
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/m2d-heldout"
STORE = OUT / "store"
MANIFEST = OUT / "manifest.jsonl"
SHAPES = ("prefill", "decode", "burst")
SEEDS = (50, 51, 52)
CANDIDATES = ((1, 512), (1, 2048), (4, 512), (4, 2048))
MAX_DRIFT_P95_MS = 10.0
MAX_ATTEMPTS = 2


def experiment_id(shape: str, seed: int, seqs: int, tokens: int) -> str:
    return f"m2d-held-{shape}-a{seed}-seq{seqs}-tok{tokens}"


def execution_order() -> tuple[tuple[str, int, int, int], ...]:
    return tuple(
        (shape, seed, seqs, tokens)
        for seed in SEEDS
        for seqs, tokens in CANDIDATES
        for shape in SHAPES
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _load(run_dir: Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json((run_dir / "result.json").read_text())


def validate_run(result: ExperimentResult, run_dir: Path) -> dict:
    """Evaluate only preregistered validity gates, never performance outcomes."""
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
            required = {"server_startup", "measured_window", "teardown", "total_occupancy"}
            phases = {span.name: span for span in timing.phases}
            if not required.issubset(phases) or not all(phases[name].completed for name in required):
                problems.append("phases_incomplete")
        except ValueError:
            problems.append("phases_invalid")

    drift: dict = {}
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
            drifts = [(actual_i - scheduled_i) * 1000 for scheduled_i, actual_i in zip(scheduled, actual)]
            p95 = _percentile(drifts, 0.95)
            drift = {
                "drift_mean_ms": round(statistics.mean(drifts), 3),
                "drift_p95_ms": round(p95, 3),
                "drift_max_ms": round(max(drifts), 3),
            }
            workload = result.config.workload
            if arrivals.get("algorithm") != workload.arrival_pattern:
                problems.append("arrival_algorithm_mismatch")
            if arrivals.get("arrival_seed") != workload.effective_arrival_seed:
                problems.append("arrival_seed_mismatch")
            if arrivals.get("burst_size") != workload.burst_size:
                problems.append("burst_size_mismatch")
            if p95 > MAX_DRIFT_P95_MS:
                problems.append("dispatch_drift_exceeded")
    return {"problems": problems, "drift": drift}


def _records() -> list[dict]:
    if not MANIFEST.is_file():
        return []
    records = []
    for line_number, line in enumerate(MANIFEST.read_text().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
    return records


def retry_allowed(problems: list[str], attempt: int) -> bool:
    return problems == ["dispatch_drift_exceeded"] and attempt < MAX_ATTEMPTS


def _validate_manifest_order(records: list[dict]) -> None:
    order = execution_order()
    cursor = 0
    invalid_shapes: set[str] = set()
    attempts: dict[str, int] = {}
    for record in records:
        while cursor < len(order) and order[cursor][0] in invalid_shapes:
            cursor += 1
        if cursor == len(order):
            raise ValueError("manifest contains records after sealed execution ended")
        shape, seed, seqs, tokens = order[cursor]
        expected = experiment_id(shape, seed, seqs, tokens)
        if record.get("experiment_id") != expected:
            raise ValueError(f"manifest violates sealed order: expected {expected}")
        attempt = attempts.get(expected, 0) + 1
        if record.get("attempt") != attempt:
            raise ValueError(f"manifest attempt mismatch for {expected}: expected {attempt}")
        attempts[expected] = attempt
        allowed = retry_allowed(record.get("problems", []), attempt)
        if record.get("retry_allowed") != allowed:
            raise ValueError(f"manifest retry decision mismatch for {expected}")
        if allowed:
            continue
        if not record.get("accepted"):
            invalid_shapes.add(shape)
        cursor += 1


def _resume_state(records: list[dict]) -> tuple[set[str], set[str], dict[str, int]]:
    accepted: set[str] = set()
    invalid_shapes: set[str] = set()
    attempts: dict[str, int] = {}
    for record in records:
        eid = record["experiment_id"]
        attempts[eid] = max(attempts.get(eid, 0), record["attempt"])
        if record.get("accepted"):
            run_dir = ROOT / record["run_dir"]
            check = validate_run(_load(run_dir), run_dir)
            if check["problems"]:
                raise ValueError(f"accepted run no longer validates: {run_dir}: {check['problems']}")
            accepted.add(eid)
        elif record.get("terminal"):
            invalid_shapes.add(record["shape"])
    return accepted, invalid_shapes, attempts


def main() -> int:
    subprocess.run(
        [sys.executable, str(HERE / "generate_corpus.py"), "--check"],
        cwd=ROOT,
        check=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE)
    records = _records()
    _validate_manifest_order(records)
    accepted, invalid_shapes, attempts = _resume_state(records)
    with MANIFEST.open("a") as manifest:
        for shape, seed, seqs, tokens in execution_order():
            eid = experiment_id(shape, seed, seqs, tokens)
            if shape in invalid_shapes or eid in accepted:
                continue
            while eid not in accepted and shape not in invalid_shapes:
                attempt = attempts.get(eid, 0) + 1
                config = ExperimentConfig.model_validate_json((HERE / f"{eid}.json").read_text())
                before = set(OUT.glob(f"{eid}-*"))
                print(f"RUN {eid} attempt={attempt}", flush=True)
                result = run_experiment(
                    config, str(OUT), ready_timeout_s=900.0, request_timeout_s=600.0
                )
                created = set(OUT.glob(f"{eid}-*")) - before
                if len(created) != 1:
                    raise RuntimeError(f"expected one run directory for {eid}; got {created}")
                run_dir = created.pop()
                check = validate_run(result, run_dir)
                accepted_now = not check["problems"]
                may_retry = retry_allowed(check["problems"], attempt)
                terminal = not accepted_now and not may_retry
                record = {
                    "experiment_id": eid,
                    "shape": shape,
                    "arrival_seed": seed,
                    "max_num_seqs": seqs,
                    "max_num_batched_tokens": tokens,
                    "attempt": attempt,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "status": result.status.value,
                    "eligible": result.is_baseline_eligible,
                    "accepted": accepted_now,
                    "retry_allowed": may_retry,
                    "terminal": terminal,
                    **check,
                }
                if accepted_now:
                    ingested = store.ingest(run_dir)
                    record.update(store_run_id=ingested.run_id, store_created=ingested.created)
                    accepted.add(eid)
                elif terminal:
                    invalid_shapes.add(shape)
                attempts[eid] = attempt
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                print(json.dumps(record, indent=2), flush=True)
                if may_retry:
                    print(
                        f"RETRY preregistered sole drift failure: {eid}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif terminal:
                    print(
                        f"INVALIDATE shape={shape}; evidence preserved",
                        file=sys.stderr,
                        flush=True,
                    )

    expected = {experiment_id(shape, seed, seqs, tokens) for shape, seed, seqs, tokens in execution_order()}
    missing = expected - accepted
    if invalid_shapes:
        print(f"M2D HELD-OUT INVALID shapes={sorted(invalid_shapes)}; accepted={len(accepted)}/36", flush=True)
        return 2
    if missing:
        raise RuntimeError(f"held-out study incomplete; missing={sorted(missing)}")
    print("M2D HELD-OUT COLLECTION COMPLETE: 36/36 accepted", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
