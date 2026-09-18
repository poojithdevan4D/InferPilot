# InferPilot regime→lever map (grounded in vLLM mechanics)

Encoded in `inferpilot.diagnosis.BottleneckDiagnosis` (v0.2.0). Synthesis of our three
campaigns' negatives + an external review of vLLM's actual scheduling behavior. The point:
a config win is only *physically possible* in specific regimes; elsewhere InferPilot must
abstain, not fish.

## The map

| Diagnosis | Signals | Lever | Preconditions to verify | Predicted effect |
|---|---|---|---|---|
| `compute_bound` | saturated, GPU mean ≥90% | **none** | — | abstain; add GPUs / quantize / smaller model |
| `kv_capacity_bound_decode` | saturated, GPU <90%, KV ≥95%, output≥prompt | **`kv_cache_dtype=fp8`** | attention not sliding-window; head_dim≠256; decode-dominated | ~30–50% more concurrent seqs at similar ITL |
| `kv_capacity_bound_prefill` | saturated, GPU <90%, KV ≥95%, prompt>output | **lower `max_num_batched_tokens`** | chunked prefill on; prefill bursts disrupting decode | better ITL, slightly worse TTFT |
| `low_utilization_latency` | not saturated, GPU <60%, KV <50% | **speculative decoding** | latency SLO present; ITL-sensitive; low QPS | 1.5–3× ITL at low concurrency (never a throughput lever) |
| `underutilized` | not saturated (otherwise) | none | — | default already adequate |
| `other_bottleneck` | saturated, spare GPU+KV | none | — | scheduler/client/CPU — no weights-level lever |

## Why our campaigns all hit `compute_bound`/`underutilized`

7B/A10 (short + decode) and 14B/A100 (long-context) were all GPU-pinned at ~100% mean.
`max_num_seqs` is largely a dead lever: vLLM's scheduler already fits concurrency to KV
blocks, so raising the ceiling only invites preemption; lowering it just drops concurrency.
The winnable regime needs the model *cheap enough to decode* that GPU has headroom while KV
is the binding constraint — i.e. a small model + long context + high concurrency.

## Key principles carried in

- **Levers are conditional, not blanket.** fp8 KV helps only compatible architectures on
  decode-dominated work (hybrid/sliding-window models gain little; head_dim=256 can regress
  prefill). Encoded as `lever_preconditions`.
- **Regimes are mutually exclusive by intent.** Capacity levers (fp8, batched-tokens) and the
  latency lever (spec-decode) live in different regimes and must not share a search space.
- **Goodput under SLO, not raw throughput.** `compare_configs` already enforces no-worse on
  every latency metric; the missing signal is a preemption counter (see gaps).
- **Abstention is the correct policy** outside a winnable regime. Online re-tuning should fire
  on a *regime change*, not continuously (dynamic controllers underperform static baselines on
  noisy signals).

## Data-collection gaps to close (make the map fully actionable)

1. **Preemption counter** — vLLM exposes preemption metrics; capture them in the runner and
   add as a first-class feasibility signal (rising preemptions ⇒ latency being traded away).
2. **Prefix cache hit rate** — required before recommending `enable_prefix_caching`; low hit
   rate ⇒ prompts aren't prefix-sticky (or chunked-prefill defeats affinity beyond chunk 1).
3. **Model architecture** — head_dim, sliding-window/hybrid attention (from HF config) to
   discharge the fp8 preconditions automatically instead of leaving them to verify.

## Next aimed experiment (the one place a win is predicted)

Small model (Qwen2.5-3B) on A10, ~8k context, 64+ concurrent decode-heavy requests, measuring
GPU util and KV usage separately. Cheap probe → `diagnose()`; proceed to fp8 vs default ONLY if
it reports `kv_capacity_bound_decode`. If it still says `compute_bound`, that too is a useful
answer: defaults are near-optimal there and the operator should scale hardware, not tune.
