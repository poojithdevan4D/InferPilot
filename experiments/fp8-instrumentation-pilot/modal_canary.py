"""Digest-pinned Modal launcher for the discarded fp8 mechanism canary.

This app is deliberately separate from every historical Modal campaign.  It
only runs a CPU image probe or the two-arm semantic canary; it cannot launch the
six-cell performance pilot.
"""

from __future__ import annotations

import glob
import io
import json
import os
import tarfile
from pathlib import Path

import modal

APP_NAME = "inferpilot-fp8-mechanism-canary"
GPU = "A10G"
IMAGE = (
    "ghcr.io/poojithdevan4d/vllm-inferpilot@"
    "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1"
)
MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
REMOTE_CACHE = "/root/.cache/huggingface"

try:
    ROOT = Path(__file__).resolve().parents[2]
    WHEELS = sorted(glob.glob(str(ROOT / "dist" / "inferpilot-*.whl")))
    WHEEL = Path(WHEELS[-1]) if WHEELS else None
except IndexError:
    # The remote module is mounted at /root/modal_canary.py; InferPilot is
    # already installed in the image built by the local import.
    WHEEL = None

image = modal.Image.from_registry(
    IMAGE,
    setup_dockerfile_commands=[
        "RUN ln -s $(command -v python3) /usr/local/bin/python",
        "ENTRYPOINT []",
    ],
)
if WHEEL is not None:
    image = image.add_local_file(
        WHEEL,
        f"/wheels/{WHEEL.name}",
        copy=True,
    ).run_commands(f"python -m pip install /wheels/{WHEEL.name}")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)


def _archive(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(directory, arcname=".")
    return buffer.getvalue()


def _config(experiment_id: str, *, pressure: bool):
    from inferpilot import EngineConfig, ExperimentConfig, WorkloadSpec

    engine = EngineConfig(
        model=MODEL,
        revision=MODEL_REVISION,
        dtype="bfloat16",
        max_model_len=9216,
        max_num_seqs=48,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.50,
        kv_cache_dtype="auto",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        sampler_backend="pytorch",
        generation_config="vllm",
        extra_args={
            "async-scheduling": False,
            "enable-logging-iteration-details": True,
        },
    )
    workload = WorkloadSpec(
        name="forced-preemption" if pressure else "low-pressure-control",
        num_requests=48 if pressure else 4,
        warmup_requests=1,
        prompt_tokens=7680 if pressure else 1024,
        output_tokens=256 if pressure else 64,
        max_concurrency=48 if pressure else 1,
        temperature=0.0,
        ignore_eos=True,
        seed=9200 if pressure else 9100,
    )
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_id,
        description=(
            "Discarded real-server mechanism canary; never included in pilot outcomes."
        ),
        engine=engine,
        workload=workload,
        tags=["fp8-instrumentation", "discarded-canary", "pressure" if pressure else "control"],
    )


@app.function(image=image, timeout=600)
def probe() -> dict:
    """Verify the pinned image and patch without allocating a GPU."""
    import inspect
    import platform

    import vllm
    from vllm.v1.metrics import loggers

    source = inspect.getsource(loggers.PrometheusStatLogger)
    return {
        "image": IMAGE,
        "python": platform.python_version(),
        "vllm": vllm.__version__,
        "recomputed_counter_in_logger": "recomputed_token_executions" in source,
        "ready": (
            vllm.__version__ == "0.29.0"
            and "recomputed_token_executions" in source
        ),
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=2400,
    volumes={REMOTE_CACHE: hf_cache},
)
def run_semantic_canary() -> dict:
    """Run fresh-server control and pressure arms and return every artifact."""
    import tempfile

    from inferpilot import MechanismEvidence, evaluate_mechanism_canary
    from inferpilot.runner.artifacts import MECHANISM_EVIDENCE_FILENAME
    from inferpilot.runner.metrics_capabilities import FP8_MECHANISM_REQUIREMENTS
    from inferpilot.runner.orchestrator import run_experiment

    os.environ.setdefault("HF_HOME", REMOTE_CACHE)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", REMOTE_CACHE)

    with tempfile.TemporaryDirectory(prefix="inferpilot-real-canary-") as raw:
        output = Path(raw)
        summaries = []
        evidence = []
        for name, pressure in (
            ("fp8-mechanism-control", False),
            ("fp8-mechanism-pressure", True),
        ):
            before = set(output.iterdir())
            result = run_experiment(
                _config(name, pressure=pressure),
                output,
                ready_timeout_s=900.0,
                request_timeout_s=600.0,
                metric_requirements=FP8_MECHANISM_REQUIREMENTS,
                require_metric_capabilities=True,
            )
            created = sorted(set(output.iterdir()) - before)
            run_dir = created[0] if len(created) == 1 else None
            evidence_path = (
                run_dir / MECHANISM_EVIDENCE_FILENAME if run_dir is not None else None
            )
            item = {
                "experiment_id": name,
                "status": result.status.value,
                "failure": result.failure.model_dump(mode="json") if result.failure else None,
                "run_dir": run_dir.name if run_dir is not None else None,
                "mechanism_evidence_present": bool(
                    evidence_path is not None and evidence_path.exists()
                ),
            }
            summaries.append(item)
            if evidence_path is not None and evidence_path.exists():
                evidence.append(MechanismEvidence.model_validate_json(evidence_path.read_text()))

        verdict = None
        if len(evidence) == 2 and all(item["status"] == "completed" for item in summaries):
            verdict = evaluate_mechanism_canary(evidence[0], evidence[1])
            report_path = output / "mechanism-canary-report.json"
            report_path.write_text(verdict.model_dump_json(indent=2))

        summary = {
            "image": IMAGE,
            "gpu": GPU,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "cells": summaries,
            "passed": bool(verdict and verdict.passed),
            "reasons": list(verdict.reasons) if verdict else ["canary_evidence_unavailable"],
            "control_recomputed_tokens": (
                evidence[0].recomputed_token_executions if len(evidence) == 2 else None
            ),
            "pressure_recomputed_tokens": (
                evidence[1].recomputed_token_executions if len(evidence) == 2 else None
            ),
            "pressure_preemptions": evidence[1].preemptions if len(evidence) == 2 else None,
            "control_iterations": len(evidence[0].iterations) if len(evidence) == 2 else None,
            "pressure_iterations": len(evidence[1].iterations) if len(evidence) == 2 else None,
        }
        (output / "modal-canary-summary.json").write_text(json.dumps(summary, indent=2))
        return {"summary": summary, "archive": _archive(output)}


@app.local_entrypoint()
def main(
    phase: str = "probe",
    output_dir: str = "runs/fp8-instrumentation-canary",
) -> None:
    """Run `probe` or `canary`; always persist returned canary evidence locally."""
    if phase == "probe":
        result = probe.remote()
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit("pinned image probe failed")
        return
    if phase != "canary":
        raise SystemExit("phase must be probe or canary")

    result = run_semantic_canary.remote()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    print(json.dumps(result["summary"], indent=2))
    if not result["summary"]["passed"]:
        raise SystemExit("semantic canary failed; performance pilot remains blocked")
