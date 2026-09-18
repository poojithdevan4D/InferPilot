# InferPilot — a senior ML-systems-engineer for LLM serving, in a box

**What it is.** Given (Model + Hardware + Workload + SLO), InferPilot **diagnoses the serving
bottleneck from measured telemetry, predicts which lever (if any) can help, verifies with a
fail-closed canary, and abstains when physics says no config can win** — recommending the right
hardware instead. It answers the operator's real questions: *am I on the right GPU? should I tune or
scale? can I turn on fp8 without breaking my SLO? what's my cost-per-token?*

**Why it's different.** It is not a knob-sweeper hoping to beat defaults (a marginal, mostly-dead
game — vLLM's scheduler already fits concurrency to KV). It reasons from the physics of the bottleneck,
carries mechanistic predictions, and every recommendation is a self-validating, tamper-evident,
human-readable artifact. **Abstention is a first-class output**, not a failure.

## What it does, in one call

`recommend_plan(measured_baseline, slo, economics, model)` →

```
DIAGNOSIS: kv_capacity_bound_decode (gpu_mean 100%, kv_peak 1.00, saturated, preemptions>0).
PREDICTION: KV full + server preempting = recompute waste burning GPU; fp8 KV frees blocks,
            cuts preemption, converts waste into throughput (~+40-50%).
RECOMMENDATION: TUNE kv_cache_dtype=fp8.  VERIFY FIRST: not sliding-window; head_dim!=256;
            long_context_accuracy_verified.
VERIFICATION: apply only after a fail-closed Pareto canary.
```
…or, when the GPU is genuinely pinned with nothing to reclaim:
```
DIAGNOSIS: compute_bound.  RECOMMENDATION: SCALE — no config lever helps; sustains 0.51 qps
under SLO on 1 GPU; cheapest fit to hold your load = A100-80GB @ $2.50/hr; ~$2.23/1M tokens.
```

## The proof

Measured on rented cloud GPUs (Modal), single-variable, fail-closed:

| Regime | Model / GPU / workload | Result |
|---|---|---|
| KV-pressure (preempting) | 3B / A10 / 8k ctx | **fp8 KV: +52% throughput**, TTFT 80s→41s, TPOT 146→120ms |
| KV-pressure (preempting) | 14B / A100-40GB / long ctx | **fp8 KV: +41% throughput** (mechanism consistent) |
| Compute-bound | 7B & 14B, several workloads | correctly diagnosed "no lever — scale"; a false dev-set win was caught by held-out |

The system also **self-corrected**: it first mis-called the 3B case compute-bound; the preemption
signal exposed the error; the model was refined to be preemption-aware and now predicts the +52%.
That loop — diagnose → predict → verify → **refine on a measured contradiction** — is the moat.

## The value

- **Stop overprovisioning.** "Compute-bound → scale to the cheapest GPU that meets your SLO" replaces
  hours of guesswork and idle-GPU waste ($10k+/mo at fleet scale).
- **Capture the wins that exist.** fp8 KV is +40–52% in KV-pressured regimes — but only there, and only
  on compatible models; InferPilot knows the difference and guards the fp8 long-context accuracy trap.
- **Trust.** Every action is a mechanistic, replayable artifact, not a black-box RL knob.

## Honest limits

Validated across a handful of models/GPUs, throughput/latency only (accuracy gate specified but not yet
run on long-context fp8). Generality, richer levers (KV offloading, PD-disaggregation, spec-decode),
and simulation-based search (Vidur) + sequential-test canaries (SPRT) are the roadmap. The reasoning
spine is built and self-validating (420+ tests).
