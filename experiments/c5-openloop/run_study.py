"""Driver for the c5 open-loop scheduling study.

Runs 3 interleaved repetitions across seq1..seq4 (max_num_seqs 1..4), validating
every run and aborting immediately on an invalid run or material dispatch drift.
Artifacts land under runs/c5-openloop/ (gitignored); a manifest.jsonl records
each run's checks and drift. Real GPU + vLLM required.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from inferpilot import ExperimentConfig
from inferpilot.runner.orchestrator import run_experiment

CFG_DIR = Path("experiments/c5-openloop")
OUT_DIR = Path("runs/c5-openloop")
MANIFEST = OUT_DIR / "manifest.jsonl"

ORDER = [
    ["seq1", "seq2", "seq3", "seq4"],
    ["seq4", "seq3", "seq2", "seq1"],
    ["seq2", "seq4", "seq1", "seq3"],
]

MAX_DRIFT_ABORT_S = 1.5  # abort if worst dispatch drift exceeds this


def _pctl(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = min(len(xs) - 1, int(q * (len(xs) - 1)))
    return xs[i]


def validate(result, run_dir: Path) -> dict:
    problems = []
    if result.status.value != "completed":
        problems.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        problems.append("not_baseline_eligible")
    agg = result.aggregates
    if agg is None or agg.num_requests != 256 or agg.num_successful != 256 or agg.num_failed != 0:
        problems.append(f"counts={None if agg is None else (agg.num_requests, agg.num_successful, agg.num_failed)}")
    eff = result.effective_config
    if eff is None or not eff.verified or eff.unverified_fields:
        problems.append("effective_not_verified")
    tel = result.telemetry
    if tel is None or tel.error is not None or tel.num_samples < 1 or tel.peak_gpu_memory_mb is None:
        problems.append("telemetry_incomplete")

    lifecycle = json.loads((run_dir / "lifecycle.json").read_text())
    if lifecycle.get("pre_teardown"):
        problems.append(f"pre_teardown_errors={len(lifecycle['pre_teardown'])}")

    arrivals_path = run_dir / "arrivals.json"
    drift = {}
    if not arrivals_path.exists():
        problems.append("arrivals_missing")
    else:
        a = json.loads(arrivals_path.read_text())
        sched = a["scheduled_offsets_s"]
        actual = a["actual_dispatch_offsets_s"]
        drifts = [act - sch for sch, act in zip(sched, actual)]
        sched_span = sched[-1] if sched else 0.0
        actual_span = actual[-1] if actual else 0.0
        drift = {
            "nominal_qps": a["request_rate_qps"],
            "n": len(sched),
            "scheduled_span_s": round(sched_span, 3),
            "realized_schedule_qps": round((len(sched) - 1) / sched_span, 4) if sched_span else None,
            "realized_dispatch_qps": round((len(actual) - 1) / actual_span, 4) if actual_span else None,
            "drift_mean_ms": round(statistics.mean(drifts) * 1000, 2) if drifts else None,
            "drift_p95_ms": round(_pctl(drifts, 0.95) * 1000, 2) if drifts else None,
            "drift_max_ms": round(max(drifts) * 1000, 2) if drifts else None,
            "drift_min_ms": round(min(drifts) * 1000, 2) if drifts else None,
        }
        if drift["drift_max_ms"] is not None and drift["drift_max_ms"] > MAX_DRIFT_ABORT_S * 1000:
            problems.append(f"material_dispatch_drift_max_ms={drift['drift_max_ms']}")
    return {"problems": problems, "drift": drift}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {p.stem: ExperimentConfig.model_validate_json(p.read_text())
               for p in CFG_DIR.glob("seq*.json")}
    with open(MANIFEST, "w") as mf:
        for rep, row in enumerate(ORDER, 1):
            for name in row:
                print(f"[rep {rep}] running {name} (max_num_seqs={configs[name].engine.max_num_seqs})", flush=True)
                result = run_experiment(
                    configs[name], str(OUT_DIR),
                    ready_timeout_s=900.0, request_timeout_s=300.0,
                )
                run_dir = sorted(OUT_DIR.glob(f"{configs[name].experiment_id}-*"),
                                 key=lambda p: p.stat().st_mtime)[-1]
                v = validate(result, run_dir)
                rec = {"rep": rep, "experiment_id": configs[name].experiment_id,
                       "max_num_seqs": configs[name].engine.max_num_seqs,
                       "run_dir": str(run_dir), "status": result.status.value,
                       "eligible": result.is_baseline_eligible, **v}
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                print("  ->", json.dumps({"problems": v["problems"], "drift": v["drift"]}), flush=True)
                if v["problems"]:
                    print(f"ABORT: {name} rep{rep} invalid: {v['problems']}", flush=True)
                    return 1
    print("STUDY COMPLETE: 12/12 runs valid", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
