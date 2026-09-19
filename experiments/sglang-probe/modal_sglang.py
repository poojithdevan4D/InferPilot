"""Minimal SGLang throughput probe: does the fp8/preemption law generalize to a 2nd engine?

Loads Qwen2.5-3B under SGLang with bf16 vs fp8 KV cache, drives a KV-pressured long-context
batch, and reports tokens/s. If fp8 wins by ~40-50% under KV pressure, the law is engine-general.

    modal run experiments/sglang-probe/modal_sglang.py::sglang_fp8
"""
from __future__ import annotations
import modal, os

GPU = os.environ.get("INFERPILOT_GPU", "A10G")
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-devel-ubuntu22.04", add_python="3.12")
    .pip_install("sglang[all]>=0.4.0", "transformers")
)
app = modal.App("inferpilot-sglang-probe")
hf_cache = modal.Volume.from_name("inferpilot-hf-cache", create_if_missing=True)
CACHE = "/root/.cache/huggingface"


@app.function(image=image, gpu=GPU, timeout=2400, volumes={CACHE: hf_cache})
def sglang_throughput(model: str, kv_dtype: str, n: int, prompt_tokens: int, out_tokens: int) -> dict:
    import os as _os, time
    _os.environ.setdefault("HF_HOME", CACHE)
    import sglang as sgl
    # long filler prompt to induce KV pressure
    filler = ("routine administrative notes on logistics scheduling and inventory were recorded "
              "during the quarterly review of regional operations and supply chains. ")
    prompt = (filler * ((prompt_tokens // 15) + 1))
    prompts = [prompt + f" Request {i}. Summarize the above." for i in range(n)]
    llm = sgl.Engine(model_path=model, kv_cache_dtype=kv_dtype, mem_fraction_static=0.9,
                     context_length=16000, disable_cuda_graph=True)
    t0 = time.monotonic()
    outs = llm.generate(prompts, {"temperature": 0.0, "max_new_tokens": out_tokens})
    dt = time.monotonic() - t0
    llm.shutdown()
    total_out = sum(len(o.get("meta_info", {}).get("output_token_logprobs", []) or []) or out_tokens for o in outs)
    return {"kv_dtype": kv_dtype, "n": n, "wall_s": round(dt, 2),
            "req_per_s": round(n / dt, 3), "approx_tok_per_s": round(n * out_tokens / dt, 1)}


@app.local_entrypoint()
def sglang_fp8(model: str = "Qwen/Qwen2.5-3B-Instruct"):
    import json
    base = sglang_throughput.remote(model, "auto", 48, 5000, 256)
    fp8 = sglang_throughput.remote(model, "fp8_e5m2", 48, 5000, 256)
    gain = (fp8["req_per_s"] / base["req_per_s"] - 1) * 100 if base["req_per_s"] else 0
    print(json.dumps({"engine": "sglang", "bf16": base, "fp8": fp8,
                      "fp8_req_gain_pct": round(gain, 1)}, indent=2))
