"""Checkpointed Modal runner for the budget-bounded 7B repair study."""

from __future__ import annotations

import glob
import importlib.util
import io
import json
import os
import tarfile
import time
from pathlib import Path

import modal

APP_NAME = "inferpilot-fp8-mechanism-7b-repair-v1"
GPU = "A10G"
IMAGE = (
    "ghcr.io/poojithdevan4d/vllm-inferpilot@"
    "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1"
)
REMOTE_CACHE = "/root/.cache/huggingface"
REMOTE_RESULTS = "/root/inferpilot-7b-mechanism-repair-results"
STUDY_DIR = "repair-v1"

# Conservative all-resource estimate: current A10 is $1.10/h; the extra
# $0.25/h covers CPU and memory.  The performance study may consume at most
# $2.25 of the user's $3 incremental authorization, reserving $0.75 for quality.
ESTIMATED_ALL_RESOURCE_USD_PER_HOUR = 1.35
INCREMENTAL_HARD_STOP_USD = 2.25
WORST_CASE_NEXT_CELL_USD = 0.35
CELL_WALL_TIME_LIMIT_S = 900.0

try:
    root = Path(__file__).resolve().parents[2]
    wheels = sorted(glob.glob(str(root / "dist" / "inferpilot-*.whl")))
    wheel = Path(wheels[-1]) if wheels else None
    repair_protocol = Path(__file__).with_name("protocol.py")
    base_protocol = root / "experiments" / "fp8-mechanism-heldout-7b" / "protocol.py"
except IndexError:
    wheel = repair_protocol = base_protocol = None

image = modal.Image.from_registry(
    IMAGE,
    add_python="3.12",
    setup_dockerfile_commands=["ENTRYPOINT []"],
).env({"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/root"})
if wheel is not None:
    image = image.add_local_file(wheel, f"/wheels/{wheel.name}", copy=True).run_commands(
        f"python -m pip install /wheels/{wheel.name}"
    )
if repair_protocol is not None and base_protocol is not None:
    image = image.add_local_file(
        repair_protocol, "/root/fp8-mechanism-repair-7b/protocol.py", copy=True
    ).add_local_file(
        base_protocol, "/root/fp8-mechanism-heldout-7b/protocol.py", copy=True
    )

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name(
    "inferpilot-fp8-mechanism-7b-repair-results", create_if_missing=True
)


def _protocol():
    path = Path("/root/fp8-mechanism-repair-7b/protocol.py")
    spec = importlib.util.spec_from_file_location("repair_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen repair protocol")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def _append(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _archive(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(directory, arcname=".")
    return buffer.getvalue()


@app.function(
    image=image,
    gpu=GPU,
    timeout=7200,
    volumes={REMOTE_CACHE: hf_cache, REMOTE_RESULTS: result_volume},
)
def run_study() -> dict:
    from inferpilot import ExperimentResult, MechanismEvidence
    from inferpilot.runner.mechanism_study import validate_mechanism_cell
    from inferpilot.runner.metrics_capabilities import FP8_MECHANISM_REQUIREMENTS
    from inferpilot.runner.orchestrator import run_experiment

    protocol = _protocol()
    os.environ.setdefault("HF_HOME", REMOTE_CACHE)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", REMOTE_CACHE)
    study = Path(REMOTE_RESULTS) / STUDY_DIR
    study.mkdir(parents=True, exist_ok=True)
    manifest = study / "manifest.jsonl"
    records = _records(manifest)
    accepted = {row["experiment_id"]: row for row in records if row.get("accepted")}
    attempts: dict[str, int] = {}
    for row in records:
        attempts[row["experiment_id"]] = max(attempts.get(row["experiment_id"], 0), row["attempt"])
    incremental_cost = sum(row.get("estimated_cell_cost_usd", 0.0) for row in records)

    for block, prompt_seed, arrival_seed, workload, arm in protocol.execution_order():
        config = protocol.config_for(block, prompt_seed, arrival_seed, workload, arm)
        eid = config.experiment_id
        if eid in accepted:
            continue
        while eid not in accepted:
            if incremental_cost + WORST_CASE_NEXT_CELL_USD > INCREMENTAL_HARD_STOP_USD:
                summary = {
                    "status": "BUDGET_STOP",
                    "accepted_cells": len(accepted),
                    "estimated_incremental_cost_usd": incremental_cost,
                }
                (study / "status.json").write_text(json.dumps(summary, indent=2))
                result_volume.commit()
                return summary
            attempt = attempts.get(eid, 0) + 1
            started = time.monotonic()
            before = set(study.iterdir())
            result = run_experiment(
                config,
                study,
                ready_timeout_s=300.0,
                request_timeout_s=600.0,
                max_wall_time_s=CELL_WALL_TIME_LIMIT_S,
                metric_requirements=FP8_MECHANISM_REQUIREMENTS,
                require_metric_capabilities=True,
            )
            created = sorted(
                path for path in set(study.iterdir()) - before
                if path.is_dir() and path.name.startswith(eid + "-")
            )
            if len(created) != 1:
                raise RuntimeError(f"expected one new bundle for {eid}; found {created}")
            run_dir = created[0]
            check = validate_mechanism_cell(result, run_dir)
            cell_cost = (
                (time.monotonic() - started) / 3600 * ESTIMATED_ALL_RESOURCE_USD_PER_HOUR
            )
            incremental_cost += cell_cost
            again = protocol.retry_allowed(check["problems"], attempt)
            record = {
                "experiment_id": eid,
                "block": block,
                "workload": workload,
                "arm": arm,
                "prompt_seed": prompt_seed,
                "arrival_seed": arrival_seed,
                "attempt": attempt,
                "run_dir": run_dir.name,
                "status": result.status.value,
                "accepted": not check["problems"],
                "retry_allowed": again,
                "problems": check["problems"],
                "drift": check["drift"],
                "load_state": check["load_state"],
                "estimated_cell_cost_usd": cell_cost,
                "estimated_incremental_cost_usd": incremental_cost,
            }
            _append(manifest, record)
            attempts[eid] = attempt
            if record["accepted"]:
                accepted[eid] = record
            (study / "status.json").write_text(json.dumps({
                "status": "RUNNING",
                "accepted_cells": len(accepted),
                "total_cells": 12,
                "estimated_incremental_cost_usd": incremental_cost,
                "last_record": record,
            }, indent=2))
            result_volume.commit()
            if check["problems"] and not again:
                summary = {
                    "status": "INVALID_STOP",
                    "accepted_cells": len(accepted),
                    "estimated_incremental_cost_usd": incremental_cost,
                    "terminal_record": record,
                }
                (study / "status.json").write_text(json.dumps(summary, indent=2))
                result_volume.commit()
                return summary

    cells = []
    for block, prompt_seed, arrival_seed, workload, arm in protocol.execution_order():
        eid = protocol.config_for(block, prompt_seed, arrival_seed, workload, arm).experiment_id
        run_dir = study / accepted[eid]["run_dir"]
        result = ExperimentResult.model_validate_json((run_dir / "result.json").read_text())
        evidence = MechanismEvidence.model_validate_json(
            (run_dir / "mechanism-evidence.json").read_text()
        )
        if validate_mechanism_cell(result, run_dir)["problems"]:
            raise ValueError(f"accepted bundle no longer validates: {run_dir}")
        cells.append((block, workload, result, evidence))
    report = protocol.evaluate_study(cells)
    (study / "decision-report.json").write_text(json.dumps(report, indent=2))
    summary = {
        "status": "COMPLETE",
        "accepted_cells": 12,
        "decision": report["decision"],
        "estimated_incremental_cost_usd": incremental_cost,
    }
    (study / "status.json").write_text(json.dumps(summary, indent=2))
    archive_path = study / "repair-v1.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    archive_path.write_bytes(_archive(study))
    result_volume.commit()
    return summary


@app.function(image=image, timeout=120, volumes={REMOTE_RESULTS: result_volume})
def read_status(include_archive: bool = False) -> dict:
    study = Path(REMOTE_RESULTS) / STUDY_DIR
    status_path = study / "status.json"
    result = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "NOT_STARTED"}
    archive_path = study / "repair-v1.tar.gz"
    if include_archive and archive_path.is_file():
        result["archive"] = archive_path.read_bytes()
    return result


@app.local_entrypoint()
def main(phase: str = "launch", output_dir: str = "runs/fp8-mechanism-7b-repair") -> None:
    if phase == "launch":
        # Keep the local invocation attached. Some Modal client versions stop an
        # ephemeral app before a spawned call starts when the entrypoint exits.
        # Per-cell and cumulative spend guards remain inside run_study.
        print(json.dumps(run_study.remote(), indent=2))
        return
    if phase not in {"status", "download"}:
        raise SystemExit("phase must be launch, status, or download")
    result = read_status.remote(include_archive=phase == "download")
    archive = result.pop("archive", None)
    print(json.dumps(result, indent=2))
    if phase == "download":
        if archive is None:
            raise SystemExit("completed archive is not available")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            bundle.extractall(destination, filter="data")
