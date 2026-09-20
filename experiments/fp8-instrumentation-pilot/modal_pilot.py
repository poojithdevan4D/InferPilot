"""Detached, checkpointed Modal launcher for the six-cell fp8 pilot.

This app is separate from the discarded canary and from every other Modal app.
It runs only the preregistered six cells, sequentially, with a fresh vLLM server
per cell.  Evidence is committed to a dedicated Volume after every attempt.
"""

from __future__ import annotations

import glob
import io
import json
import os
import tarfile
import time
from pathlib import Path

import modal

APP_NAME = "inferpilot-fp8-performance-pilot-v1"
GPU = "A10G"
IMAGE = (
    "ghcr.io/poojithdevan4d/vllm-inferpilot@"
    "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1"
)
REMOTE_CACHE = "/root/.cache/huggingface"
REMOTE_RESULTS = "/root/inferpilot-pilot-results"
STUDY_DIR = "pilot-v1"
A10G_USD_PER_HOUR = 1.10
HARD_STOP_USD = 12.0
WORST_CASE_NEXT_CELL_USD = 2.0
# Passing discarded semantic canary, documented in the repository.  It counts
# against the protocol budget even though it cannot enter pilot outcomes.
PRIOR_CANARY_COST_USD = 0.0781

try:
    ROOT = Path(__file__).resolve().parents[2]
    WHEELS = sorted(glob.glob(str(ROOT / "dist" / "inferpilot-*.whl")))
    WHEEL = Path(WHEELS[-1]) if WHEELS else None
    candidate_protocol = Path(__file__).with_name("pilot_protocol.py")
    PROTOCOL = candidate_protocol if candidate_protocol.is_file() else None
except IndexError:
    # Remote module imports use /root/modal_pilot.py; both files are already in
    # the image assembled by the local import.
    WHEEL = None
    PROTOCOL = None

image = modal.Image.from_registry(
    IMAGE,
    add_python="3.12",
    setup_dockerfile_commands=["ENTRYPOINT []"],
).env({"PYTHONPATH": "/usr/local/lib/python3.12/dist-packages:/root"})
if WHEEL is not None:
    image = image.add_local_file(WHEEL, f"/wheels/{WHEEL.name}", copy=True).run_commands(
        f"python -m pip install /wheels/{WHEEL.name}"
    )
if PROTOCOL is not None:
    image = image.add_local_file(PROTOCOL, "/root/pilot_protocol.py", copy=True)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name("inferpilot-fp8-pilot-results", create_if_missing=True)


def _records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    timeout=14_400,
    volumes={REMOTE_CACHE: hf_cache, REMOTE_RESULTS: result_volume},
)
def run_pilot() -> dict:
    """Run/resume the frozen matrix, stopping before comparison on invalid evidence."""
    from inferpilot import ExperimentResult, MechanismEvidence
    from inferpilot.runner.metrics_capabilities import FP8_MECHANISM_REQUIREMENTS
    from inferpilot.runner.orchestrator import run_experiment
    from pilot_protocol import config_for, evaluate_pilot, execution_order, retry_allowed, validate_cell

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
    cumulative_cost = PRIOR_CANARY_COST_USD + sum(
        row.get("estimated_cell_cost_usd", 0.0) for row in records
    )

    for block, prompt_seed, arrival_seed, arm in execution_order():
        config = config_for(block, prompt_seed, arrival_seed, arm)
        eid = config.experiment_id
        if eid in accepted:
            continue
        while eid not in accepted:
            if cumulative_cost + WORST_CASE_NEXT_CELL_USD > HARD_STOP_USD:
                summary = {
                    "status": "BUDGET_STOP",
                    "accepted_cells": len(accepted),
                    "estimated_cost_usd": cumulative_cost,
                }
                (study / "status.json").write_text(json.dumps(summary, indent=2))
                result_volume.commit()
                return summary

            attempt = attempts.get(eid, 0) + 1
            cell_started = time.monotonic()
            before = set(study.iterdir())
            result = run_experiment(
                config,
                study,
                ready_timeout_s=900.0,
                request_timeout_s=300.0,
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
            check = validate_cell(result, run_dir)
            cell_cost = (time.monotonic() - cell_started) / 3600 * A10G_USD_PER_HOUR
            cumulative_cost += cell_cost
            again = retry_allowed(check["problems"], attempt)
            record = {
                "experiment_id": eid,
                "block": block,
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
                "estimated_cumulative_cost_usd": cumulative_cost,
            }
            _append(manifest, record)
            records.append(record)
            attempts[eid] = attempt
            if record["accepted"]:
                accepted[eid] = record
            (study / "status.json").write_text(json.dumps({
                "status": "RUNNING",
                "accepted_cells": len(accepted),
                "last_record": record,
            }, indent=2))
            result_volume.commit()
            if check["problems"] and not again:
                summary = {
                    "status": "INVALID_STOP",
                    "accepted_cells": len(accepted),
                    "terminal_record": record,
                }
                (study / "status.json").write_text(json.dumps(summary, indent=2))
                result_volume.commit()
                return summary

    cells = []
    for block, _prompt_seed, _arrival_seed, arm in execution_order():
        record = accepted[config_for(block, _prompt_seed, _arrival_seed, arm).experiment_id]
        run_dir = study / record["run_dir"]
        result = ExperimentResult.model_validate_json((run_dir / "result.json").read_text())
        evidence = MechanismEvidence.model_validate_json(
            (run_dir / "mechanism-evidence.json").read_text()
        )
        # Revalidate accepted evidence on resume before outcome inspection.
        if validate_cell(result, run_dir)["problems"]:
            raise ValueError(f"accepted bundle no longer validates: {run_dir}")
        cells.append((block, result, evidence))
    report = evaluate_pilot(cells)
    (study / "pilot-decision-report.json").write_text(json.dumps(report, indent=2))
    summary = {
        "status": "COMPLETE",
        "accepted_cells": 6,
        "decision": report["decision"],
        "content_sha256": report["content_sha256"],
    }
    (study / "status.json").write_text(json.dumps(summary, indent=2))
    archive_path = study / "pilot-v1.tar.gz"
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
    if include_archive and (study / "pilot-v1.tar.gz").is_file():
        result["archive"] = (study / "pilot-v1.tar.gz").read_bytes()
    return result


@app.local_entrypoint()
def main(phase: str = "launch", output_dir: str = "runs/fp8-instrumentation-pilot") -> None:
    if phase == "launch":
        call = run_pilot.spawn()
        print(json.dumps({"dispatched": True, "function_call_id": call.object_id}, indent=2))
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
