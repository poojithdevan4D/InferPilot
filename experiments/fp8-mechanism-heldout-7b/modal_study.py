"""Detached, checkpointed Modal execution for the focused 7B held-out study."""

from __future__ import annotations

import glob
import io
import json
import os
import tarfile
import time
from pathlib import Path

import modal

APP_NAME = "inferpilot-fp8-mechanism-7b-heldout-v1"
GPU = "A10G"
IMAGE = (
    "ghcr.io/poojithdevan4d/vllm-inferpilot@"
    "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1"
)
REMOTE_CACHE = "/root/.cache/huggingface"
REMOTE_RESULTS = "/root/inferpilot-7b-mechanism-results"
STUDY_DIR = "heldout-v1"
A10G_USD_PER_HOUR = 1.10
PRIOR_SPEND_USD = 0.8084
HARD_STOP_USD = 12.0
WORST_CASE_NEXT_CELL_USD = 2.0

try:
    ROOT = Path(__file__).resolve().parents[2]
    wheels = sorted(glob.glob(str(ROOT / "dist" / "inferpilot-*.whl")))
    WHEEL = Path(wheels[-1]) if wheels else None
    candidate = Path(__file__).with_name("protocol.py")
    PROTOCOL = candidate if candidate.is_file() else None
except IndexError:
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
    image = image.add_local_file(PROTOCOL, "/root/heldout_protocol.py", copy=True)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)
result_volume = modal.Volume.from_name(
    "inferpilot-fp8-mechanism-7b-results", create_if_missing=True
)


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
    timeout=21_600,
    volumes={REMOTE_CACHE: hf_cache, REMOTE_RESULTS: result_volume},
)
def run_study() -> dict:
    from heldout_protocol import config_for, evaluate_study, execution_order, retry_allowed
    from inferpilot import ExperimentResult, MechanismEvidence
    from inferpilot.runner.mechanism_study import validate_mechanism_cell
    from inferpilot.runner.metrics_capabilities import FP8_MECHANISM_REQUIREMENTS
    from inferpilot.runner.orchestrator import run_experiment

    os.environ.setdefault("HF_HOME", REMOTE_CACHE)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", REMOTE_CACHE)
    study = Path(REMOTE_RESULTS) / STUDY_DIR
    study.mkdir(parents=True, exist_ok=True)
    manifest = study / "manifest.jsonl"
    records = _records(manifest)
    accepted = {row["experiment_id"]: row for row in records if row.get("accepted")}
    attempts: dict[str, int] = {}
    for row in records:
        attempts[row["experiment_id"]] = max(
            attempts.get(row["experiment_id"], 0), row["attempt"]
        )
    cumulative_cost = PRIOR_SPEND_USD + sum(
        row.get("estimated_cell_cost_usd", 0.0) for row in records
    )

    for block, prompt_seed, arrival_seed, workload, arm in execution_order():
        config = config_for(block, prompt_seed, arrival_seed, workload, arm)
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
                request_timeout_s=1200.0,
                metric_requirements=FP8_MECHANISM_REQUIREMENTS,
                require_metric_capabilities=True,
            )
            created = sorted(
                path
                for path in set(study.iterdir()) - before
                if path.is_dir() and path.name.startswith(eid + "-")
            )
            if len(created) != 1:
                raise RuntimeError(f"expected one new bundle for {eid}; found {created}")
            run_dir = created[0]
            check = validate_mechanism_cell(result, run_dir)
            cell_cost = (time.monotonic() - cell_started) / 3600 * A10G_USD_PER_HOUR
            cumulative_cost += cell_cost
            again = retry_allowed(check["problems"], attempt)
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
                "estimated_cumulative_cost_usd": cumulative_cost,
            }
            _append(manifest, record)
            records.append(record)
            attempts[eid] = attempt
            if record["accepted"]:
                accepted[eid] = record
            (study / "status.json").write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "accepted_cells": len(accepted),
                        "total_cells": 12,
                        "last_record": record,
                    },
                    indent=2,
                )
            )
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
    for block, prompt_seed, arrival_seed, workload, arm in execution_order():
        eid = config_for(block, prompt_seed, arrival_seed, workload, arm).experiment_id
        run_dir = study / accepted[eid]["run_dir"]
        result = ExperimentResult.model_validate_json((run_dir / "result.json").read_text())
        evidence = MechanismEvidence.model_validate_json(
            (run_dir / "mechanism-evidence.json").read_text()
        )
        if validate_mechanism_cell(result, run_dir)["problems"]:
            raise ValueError(f"accepted bundle no longer validates: {run_dir}")
        cells.append((block, workload, result, evidence))
    report = evaluate_study(cells)
    (study / "heldout-decision-report.json").write_text(json.dumps(report, indent=2))
    summary = {
        "status": "COMPLETE",
        "accepted_cells": 12,
        "decision": report["decision"],
        "content_sha256": report["content_sha256"],
        "estimated_cumulative_cost_usd": cumulative_cost,
    }
    (study / "status.json").write_text(json.dumps(summary, indent=2))
    archive_path = study / "heldout-v1.tar.gz"
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
    archive_path = study / "heldout-v1.tar.gz"
    if include_archive and archive_path.is_file():
        result["archive"] = archive_path.read_bytes()
    return result


@app.local_entrypoint()
def main(
    phase: str = "launch",
    output_dir: str = "runs/fp8-mechanism-7b-heldout",
) -> None:
    if phase == "launch":
        call = run_study.spawn()
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
