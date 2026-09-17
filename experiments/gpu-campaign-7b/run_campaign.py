"""Pure validation/store/manifest helpers for the GPU campaign.

GPU work runs on Modal via the `campaign` local_entrypoint in modal_app.py; this
module holds everything that runs LOCALLY: config selection, the M3 validator,
evidence-store ingest, and the resume-safe manifest. It must NOT import `modal`
(importing it here would make run_cell a serialized function and force a local/
image Python-version match — see modal_app.campaign).

Run via:  modal run experiments/gpu-campaign-7b/modal_app.py::campaign --phase phase1
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from inferpilot import ExperimentConfig, ExperimentResult, RunnerPhaseTiming
from inferpilot.comparison.store import ResultStore

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/gpu-campaign-7b"
MANIFEST = OUT / "manifest.jsonl"
STORE = OUT / "store"
MAX_ATTEMPTS = 2

_g = importlib.util.spec_from_file_location("g7_gen", HERE / "generate_configs.py")
GEN = importlib.util.module_from_spec(_g)
_g.loader.exec_module(GEN)


def _percentile(xs, q):
    o = sorted(xs)
    return o[min(len(o) - 1, int(q * (len(o) - 1)))]


def validate(result: ExperimentResult, run_dir: Path, rate: float, seed: int) -> dict:
    """Same acceptance rules as M3 (counts, verified config, telemetry, phases, drift)."""
    p, a = [], result.aggregates
    if result.status.value != "completed":
        p.append(f"status={result.status.value}")
    if not result.is_baseline_eligible:
        p.append("not_baseline_eligible")
    if a is None or (a.num_requests, a.num_successful, a.num_failed) != (256, 256, 0):
        p.append("request_counts_invalid")
    if result.effective_config is None or not result.effective_config.verified or result.effective_config.unverified_fields:
        p.append("effective_not_verified")
    if result.telemetry is None or result.telemetry.error is not None or result.telemetry.num_samples < 1:
        p.append("telemetry_incomplete")
    lp = run_dir / "lifecycle.json"
    if not lp.is_file() or json.loads(lp.read_text()).get("pre_teardown"):
        p.append("lifecycle_invalid")
    pp = run_dir / "phases.json"
    if not pp.is_file():
        p.append("phases_missing")
    else:
        try:
            timing = RunnerPhaseTiming.model_validate_json(pp.read_text())
            by = {x.name: x for x in timing.phases}
            required = {"server_startup", "measured_window", "teardown", "total_occupancy"}
            if not required.issubset(by) or not all(by[x].completed for x in required):
                p.append("phases_incomplete")
        except ValueError:
            p.append("phases_invalid")
    drift, ap = {}, run_dir / "arrivals.json"
    if not ap.is_file():
        p.append("arrivals_missing")
    else:
        ar = json.loads(ap.read_text())
        scheduled, actual = ar.get("scheduled_offsets_s", []), ar.get("actual_dispatch_offsets_s", [])
        if len(scheduled) != 256 or len(actual) != 256:
            p.append("arrival_counts_invalid")
        else:
            ds = [(x - y) * 1000 for y, x in zip(scheduled, actual)]
            drift = {
                "drift_mean_ms": round(statistics.mean(ds), 3),
                "drift_p95_ms": round(_percentile(ds, 0.95), 3),
                "drift_max_ms": round(max(ds), 3),
            }
            if ar.get("algorithm") != "poisson-v1" or ar.get("arrival_seed") != seed or ar.get("request_rate_qps") != float(rate):
                p.append("arrival_provenance_mismatch")
            if drift["drift_p95_ms"] > 10:
                p.append("dispatch_drift_exceeded")
    return {"problems": p, "drift": drift}


def _retry(problems, attempt):
    return problems == ["dispatch_drift_exceeded"] and attempt < MAX_ATTEMPTS


def _records():
    return [json.loads(x) for x in MANIFEST.read_text().splitlines()] if MANIFEST.is_file() else []


def _accepted_ids():
    return {r["experiment_id"] for r in _records() if r["accepted"]}


def run_ids(eids: list[str], runner) -> int:
    """Run each config id via `runner(config_json)->payload`, validate locally, append
    manifest. Resume-safe: already-accepted cells are skipped. `runner` is
    modal_app.run_cell.remote, injected by the campaign entrypoint so this module
    never imports modal."""
    OUT.mkdir(parents=True, exist_ok=True)
    store = ResultStore(STORE)
    accepted = _accepted_ids()
    attempts: dict[str, int] = {}
    for r in _records():
        attempts[r["experiment_id"]] = max(attempts.get(r["experiment_id"], 0), r["attempt"])
    with MANIFEST.open("a") as manifest:
        for eid in eids:
            if eid in accepted:
                print(f"SKIP {eid} (already accepted)", flush=True)
                continue
            cfg = ExperimentConfig.model_validate_json((HERE / f"{eid}.json").read_text())
            rate, seed = cfg.workload.request_rate_qps, cfg.workload.arrival_seed
            width = cfg.engine.max_num_seqs
            while eid not in accepted:
                attempt = attempts.get(eid, 0) + 1
                print(f"RUN {eid} attempt={attempt}", flush=True)
                payload = runner(cfg.model_dump_json())
                run_dir = OUT / payload["run_dir_name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                for name, text in payload["artifacts"].items():
                    (run_dir / name).write_text(text)
                result = ExperimentResult.model_validate_json((run_dir / "result.json").read_text())
                check = validate(result, run_dir, rate, seed)
                ok = not check["problems"]
                again = _retry(check["problems"], attempt)
                rec = {
                    "experiment_id": eid, "request_rate_qps": rate, "arrival_seed": seed,
                    "max_num_seqs": width, "attempt": attempt,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "accepted": ok, "retry_allowed": again, **check,
                }
                if ok:
                    rec["store_run_id"] = store.ingest(run_dir).run_id
                    accepted.add(eid)
                attempts[eid] = attempt
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                print(json.dumps(rec, indent=2), flush=True)
                if not ok and not again:
                    print(f"CELL {eid} INVALID; stopping phase (terminal per stopping rule)", file=sys.stderr)
                    return 1
    return 0


def _phase_ids(phase: str, widths: tuple[int, ...]) -> list[str]:
    if phase == "phase0":
        return [GEN.dev_id(6, 60, 1)]
    if phase == "phase1":
        return [GEN.default_id(r, s) for r in GEN.RATES for s in GEN.DEV_SEEDS]
    if phase == "phase2":
        return [GEN.dev_id(r, s, w) for r in GEN.RATES for s in GEN.DEV_SEEDS for w in GEN.WIDTHS]
    if phase == "phase3":
        return [GEN.held_id(r, s, w) for r in GEN.RATES for s in GEN.HELD_SEEDS for w in widths]
    raise SystemExit(f"unknown phase {phase}")


def freeze_check() -> None:
    """Assert on-disk configs match the generator before spending GPU."""
    import subprocess
    subprocess.run([sys.executable, str(HERE / "generate_configs.py"), "--check"], check=True)
