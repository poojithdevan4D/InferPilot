"""Modal app for the GPU campaign: GPU work runs here, evidence validation stays local.

Two functions:
  probe()          - cheap (~seconds): report GPU, nvcc, torch.cuda, vllm version.
                     Validates the container BEFORE any model download / paid run.
  run_cell(config) - run one ExperimentConfig end-to-end on the GPU and return the
                     run-dir artifacts (result.json, arrivals.json, phases.json,
                     lifecycle.json, server logs) as a dict for local validation.

The image pins Python 3.12 + vLLM 0.29.0 (matching the local bench env) on a CUDA
12.6 *devel* base (nvcc >= 12.6, which unblocks the FlashInfer sampler JIT the RTX
3050 could not build). The HuggingFace cache is a persisted Volume so the 7B weights
download once and are reused across every cell (critical for cost).

Run the probe:   modal run experiments/gpu-campaign-7b/modal_app.py::probe
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import modal

APP_NAME = "inferpilot-gpu-campaign"
# GPU is read at import so it must be set BEFORE `modal run` imports this module,
# e.g. INFERPILOT_GPU=A100-40GB modal run ...::campaign --campaign gpu-campaign-14b-kv
GPU = os.environ.get("INFERPILOT_GPU", "A10G")

# Resolve the wheel only when running locally (build/deploy). Inside the container
# this module lives at /root/modal_app.py (no parents[2]) and the package is already
# installed, so a failure here must NOT crash import.
try:
    _WHEEL = sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "dist" / "inferpilot-*.whl")))
    _WHEEL_PATH = _WHEEL[-1] if _WHEEL else None
except IndexError:
    _WHEEL_PATH = None

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.29.0")
)
if _WHEEL_PATH:
    _WHEEL_NAME = os.path.basename(_WHEEL_PATH)  # pip needs the versioned filename
    image = image.add_local_file(
        _WHEEL_PATH, f"/wheels/{_WHEEL_NAME}", copy=True
    ).run_commands(f"pip install /wheels/{_WHEEL_NAME}")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)
CACHE = "/root/.cache/huggingface"


@app.local_entrypoint()
def main():
    """Print the probe result locally (modal run modal_app.py)."""
    import json

    print(json.dumps(probe.remote(), indent=2))


@app.local_entrypoint()
def campaign(phase: str, widths: str = "", campaign: str = "gpu-campaign-7b"):
    """Drive a campaign phase: run cells on the GPU, validate + store locally.

    modal run experiments/gpu-campaign-7b/modal_app.py::campaign --phase phase1
    modal run .../modal_app.py::campaign --phase phase1 --campaign gpu-campaign-7b-decode
    """
    import importlib.util
    import os

    os.environ["G7_CAMPAIGN"] = campaign  # picks experiments/<campaign>/ + runs/<campaign>/
    spec = importlib.util.spec_from_file_location(
        "g7_campaign", Path(__file__).resolve().parent / "run_campaign.py"
    )
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    rc.freeze_check()
    ws = tuple(int(w) for w in widths.split(",") if w) if widths else rc.GEN.WIDTHS
    eids = rc._phase_ids(phase, ws)
    code = rc.run_ids(eids, lambda cj: run_cell.remote(cj))
    if code != 0:
        raise SystemExit(code)


@app.local_entrypoint()
def cell(config: str, out: str = "runs/gpu-campaign-7b-smoke"):
    """Run one config on the GPU and pull its artifacts back locally.

    modal run experiments/gpu-campaign-7b/modal_app.py::cell --config <path>
    """
    import json
    from pathlib import Path

    payload = run_cell.remote(Path(config).read_text())
    dest = Path(out) / payload["run_dir_name"]
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in payload["artifacts"].items():
        (dest / name).write_text(text)
    result = json.loads(payload["artifacts"].get("result.json", "{}"))
    agg = result.get("aggregates") or {}
    print(json.dumps({
        "experiment_id": payload["experiment_id"],
        "status": payload["status"],
        "wrote_to": str(dest),
        "num_requests": agg.get("num_requests"),
        "num_successful": agg.get("num_successful"),
        "num_failed": agg.get("num_failed"),
        "ttft_p95_ms": agg.get("ttft_p95_ms"),
        "tpot_p95_ms": agg.get("tpot_p95_ms"),
    }, indent=2))


@app.function(image=image, gpu=GPU, timeout=600)
def probe() -> dict:
    """Cheap environment validation — no model download, no serving."""
    import subprocess

    import torch

    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception as exc:  # noqa: BLE001
        vllm_version = f"IMPORT FAILED: {exc}"
    try:
        from inferpilot.runner.server import default_vllm_executable
        vllm_exe = default_vllm_executable()
    except Exception as exc:  # noqa: BLE001
        vllm_exe = f"resolve failed: {exc}"
    return {
        "gpu": str(smi),
        "torch": str(torch.__version__),  # torch subclass type -> cast so local unpickle needs no torch
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": str(torch.version.cuda),
        "nvcc": nvcc.strip().splitlines()[-1] if nvcc else "MISSING",
        "vllm": str(vllm_version),
        "vllm_executable": str(vllm_exe),
    }


@app.function(image=image, gpu=GPU, timeout=3600, volumes={CACHE: hf_cache})
def run_cell(config_json: str, ready_timeout_s: float = 900, request_timeout_s: float = 600) -> dict:
    """Run one experiment on the GPU; return run-dir artifacts for local validation."""
    os.environ.setdefault("HF_HOME", CACHE)
    from inferpilot import ExperimentConfig
    from inferpilot.runner.orchestrator import run_experiment

    config = ExperimentConfig.model_validate_json(config_json)
    out = "/tmp/campaign-runs"
    before = set(glob.glob(f"{out}/{config.experiment_id}-*"))
    result = run_experiment(
        config, out, ready_timeout_s=ready_timeout_s, request_timeout_s=request_timeout_s
    )
    made = set(glob.glob(f"{out}/{config.experiment_id}-*")) - before
    if len(made) != 1:
        raise RuntimeError(f"expected one run directory, got {sorted(made)}")
    run_dir = Path(made.pop())
    hf_cache.commit()

    artifacts: dict[str, str] = {}
    for name in (
        "result.json", "arrivals.json", "phases.json", "lifecycle.json",
        "server.stdout.log", "server.stderr.log",
    ):
        p = run_dir / name
        if p.is_file():
            artifacts[name] = p.read_text()
    return {
        "experiment_id": config.experiment_id,
        "run_dir_name": run_dir.name,
        "status": result.status.value,
        "artifacts": artifacts,
    }
