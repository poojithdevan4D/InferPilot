# InferPilot is engine-agnostic by construction (vLLM today, SGLang/TRT-LLM next)

InferPilot's **reasoning layer** depends on *measured signals*, not on any engine's internals:

- `detect_saturation` — TTFT across arrival order (any engine's per-request timings).
- `BottleneckDiagnosis` — GPU util, KV-cache usage %, **preemption count**, saturation. Every
  continuous-batching engine (vLLM, SGLang, TRT-LLM) exposes these (SGLang and TRT-LLM both have
  Prometheus metrics for KV/cache usage and preemption/eviction).
- `analyze_fit` / `recommend_scale` — pure model-architecture + GPU arithmetic; engine-independent.
- `CapacityAdvisory`, `compare_configs`, quality gates — operate on `ExperimentResult` (timings +
  aggregates + telemetry); no engine coupling.

The **only** engine-specific code is the runner:
- `runner/server.build_vllm_command` — the server launch argv.
- `runner/telemetry` — the `/metrics` names (`vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`).

**Adding an engine = two small adapters** (a command builder + a metric-name map). The diagnosis,
advisory, quality, and comparison stack is unchanged. The fp8/preemption **law itself is a property of
paged-KV continuous batching**, not of vLLM: when KV is full and the scheduler preempts (recompute
waste), halving KV bytes reclaims throughput — this mechanism exists identically in SGLang and TRT-LLM.

**Prediction (to verify):** on SGLang, a KV-pressured (preempting) fp8-KV run should show the same
+~40–50% throughput vs bf16 KV, and the SAME diagnosis (`kv_capacity_bound` from SGLang's telemetry).
A minimal SGLang throughput probe (`experiments/sglang-probe/`) tests this; the runner adapters are the
follow-on for full campaign support.

**Status:** validated on vLLM (3 models × 2 GPUs). SGLang/TRT-LLM adapters + a confirming run are the
next breadth step. The claim here is *design-level* engine-agnosticism (true by construction) plus a
*mechanistic prediction* for SGLang — stated honestly, not yet fully measured on a second engine.

## SGLang result (2026-09-19): fp8 KV throughput win generalizes (+71%)

Minimal SGLang probe (Qwen2.5-3B, ~5k-token prompts × 48, KV-pressured batch, A10):

| engine | bf16 KV | fp8 KV | gain |
|---|---|---|---|
| vLLM (arrival-rate, 3 models) | baseline | fp8 | +40–52% |
| **SGLang (batch throughput, 3B)** | ~0.74 req/s | **1.27 req/s** | **+71%** |

fp8 KV materially improves throughput on SGLang too, confirming the mechanism is a property of
paged-KV continuous batching, not vLLM. **Honest caveat:** this probe measured throughput only — it
did NOT capture SGLang's KV/preemption telemetry, so the `kv_capacity_bound` *diagnosis* is not yet
confirmed on SGLang (only the win is). Wiring a SGLang telemetry adapter (metric-name map) to run the
full diagnose→recommend→verify loop is the follow-on; the reasoning stack itself is unchanged.
