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


@app.function(image=image, gpu=GPU, timeout=1800, volumes={CACHE: hf_cache})
def quality_probe(model: str, kv_cache_dtype: str, prompts: list, max_tokens: int = 128) -> list:
    """Greedy-decode the given prompts offline at a given kv_cache_dtype; return output token-ids.

    Used to compare baseline (bf16 KV) vs fp8 KV outputs on identical prompts (temperature=0),
    so a throughput win can be checked for silent output drift."""
    import os as _os

    _os.environ.setdefault("HF_HOME", CACHE)
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, kv_cache_dtype=kv_cache_dtype, max_model_len=4096,
              gpu_memory_utilization=0.90, dtype="auto")
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
    # preserve input order; return both token-ids (for agreement) and text (for QA scoring)
    by_prompt = {o.prompt: {"tokens": list(o.outputs[0].token_ids), "text": o.outputs[0].text}
                 for o in outs}
    return [by_prompt[p] for p in prompts]


@app.local_entrypoint()
def quality_qa(model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Measure ACTUAL quality: does fp8 KV still answer factual questions correctly vs bf16?

    Canonical-answer task (unlike open-ended generation, exact-match on the answer is meaningful).
    modal run experiments/gpu-campaign-7b/modal_app.py::quality_qa
    """
    import json

    qa = [
        ("What is the capital of France? Answer in one word.", "paris"),
        ("What is 17 multiplied by 23? Give only the number.", "391"),
        ("Who wrote the play Romeo and Juliet? Last name only.", "shakespeare"),
        ("What is the chemical symbol for gold?", "au"),
        ("How many continents are there on Earth?", "seven"),
        ("What planet is known as the Red Planet?", "mars"),
        ("What is the square root of 144?", "12"),
        ("In what year did World War II end?", "1945"),
        ("What is the largest ocean on Earth?", "pacific"),
        ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
        ("What is the freezing point of water in Celsius?", "0"),
        ("Who painted the Mona Lisa? Last name only.", "vinci"),
        ("What is the powerhouse of the cell?", "mitochondria"),
        ("How many sides does a hexagon have?", "6"),
        ("What is the capital of Japan?", "tokyo"),
        ("What is 100 divided by 4?", "25"),
    ]
    prompts = [q for q, _ in qa]
    answers = [a for _, a in qa]
    base = quality_probe.remote(model, "auto", prompts, 48)
    fp8 = quality_probe.remote(model, "fp8", prompts, 48)

    def score(outs):
        return [ans in o["text"].strip().lower() for o, ans in zip(outs, answers)]

    b_ok, f_ok = score(base), score(fp8)
    both = sum(1 for b, f in zip(b_ok, f_ok) if b and f)
    regressions = [prompts[i] for i in range(len(qa)) if b_ok[i] and not f_ok[i]]
    print(json.dumps({
        "model": model, "n": len(qa),
        "bf16_correct": sum(b_ok), "fp8_correct": sum(f_ok),
        "fp8_regressions_vs_bf16": len(regressions),
        "regressed_questions": regressions,
    }, indent=2))


@app.local_entrypoint()
def quality(model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Compare bf16-KV vs fp8-KV greedy outputs on identical prompts; print token agreement.

    modal run experiments/gpu-campaign-7b/modal_app.py::quality
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from inferpilot import QualitySpec, evaluate_quality

    # 16 fixed, varied, long-ish instruction prompts (identical for both configs)
    topics = [
        "the causes of the French Revolution", "how photosynthesis works",
        "the architecture of a transformer neural network", "the plot of Hamlet",
        "how vaccines train the immune system", "the economics of inflation",
        "the theory of plate tectonics", "how a CPU pipeline executes instructions",
        "the history of the Roman Empire", "how RSA public-key cryptography works",
        "the water cycle in detail", "the rules and strategy of chess",
        "how black holes form and evaporate", "the process of protein synthesis",
        "the causes of the 2008 financial crisis", "how GPS determines position",
    ]
    prompts = [
        f"You are a meticulous teacher. In several detailed paragraphs, explain {t}. "
        f"Cover the key mechanisms, give concrete examples, and note common misconceptions. "
        f"Be precise and thorough." for t in topics
    ]
    base = quality_probe.remote(model, "auto", prompts, 128)
    fp8 = quality_probe.remote(model, "fp8", prompts, 128)
    gate = evaluate_quality(QualitySpec(), [(b["tokens"], f["tokens"]) for b, f in zip(base, fp8)])
    print(json.dumps({
        "model": model, "num_prompts": len(prompts),
        "agreement_rate": round(gate.agreement_rate, 4),
        "matching": gate.matching_tokens, "total": gate.total_tokens,
        "passed": gate.passed, "reasons": gate.reasons,
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
