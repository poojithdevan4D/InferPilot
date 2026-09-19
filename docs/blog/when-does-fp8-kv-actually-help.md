# When does fp8 KV cache actually help? We measured it — and built the tool that decides.

fp8 KV cache is sold as a near-free throughput win: halve the KV memory, serve more, done.
We ran the experiments on rented cloud GPUs, and the reality is sharper and more useful than the
folklore. fp8 is **not** a free speedup, it is **not** always a win, and it can **silently change your
outputs**. But there is a clean, mechanistic rule for exactly when it helps — and we built a system
(InferPilot) that reads that rule off telemetry and refuses to guess.

Everything below is measured, single-variable, and reproducible. Every claim comes with its caveats.

## The two-sided law

We tested `kv_cache_dtype=fp8` vs bf16 KV on Qwen2.5 **3B, 7B, and 14B** across **A10 and A100**, plus **Mistral-7B-v0.3** (a different model family), in a
KV-pressured regime and a compute-bound regime:

| Regime (read from telemetry) | fp8 vs bf16 | Verdict |
|---|---|---|
| KV full **+ server preempting** (3B/7B/14B, **Mistral-7B**) | **+40% to +52% throughput** | real win |
| compute-bound (KV low, **no preemption**) | **+1.7%** (noise) | correctly nothing |

The counter-intuitive part: **GPU utilization was ~100% in *both* regimes.** If you gate on "is the GPU
busy," you'd wrongly conclude fp8 can never help. The actual discriminator is **preemptions**: when KV is
full and the scheduler preempts (evict a sequence's KV, recompute it later), a chunk of that "100% GPU"
is *wasted on recompute*. fp8 halves KV bytes, frees blocks, kills the preemption, and converts the
wasted compute into throughput. No preemption → nothing to reclaim → fp8 does nothing.

We know this because **our own model got it wrong first.** It initially called the winning case
"compute-bound, no lever." The preemption counter exposed the error; we made the diagnosis
preemption-aware; it then predicted the +52% correctly and generalized across all three models. A
diagnostic that self-corrects from a measured contradiction is worth more than one that happens to be
right.

## Is it a "free speedup"? No — it's a goodput-ceiling lever.

Every win above happened at **TTFT of 20–94 seconds** — servers past capacity. We tested a lower load:
even then the baseline was saturated, and fp8 gave +52% and halved TTFT but was *still* saturated. The
honest framing: **fp8 lets a KV-pressured server sustain ~50% more load before latency explodes.** It is
a capacity/goodput lever, not a "faster at low load" lever. Report it as goodput-under-SLO and
cost-per-token, never as raw latency.

## Does it hurt quality? We measured it four ways.

fp8 KV quantization can silently degrade outputs. We checked (Qwen2.5-3B, fp8 vs bf16):

| Measurement | Result | Meaning |
|---|---|---|
| Greedy token agreement | **12%** | outputs *reworded* — a cascade artifact, **not** a quality metric |
| Factual QA (canonical answers) | **0/16 regressions** | task-lossless on short tasks |
| Teacher-forced KL | mean >0.01, **p99 0.39** | a real, mostly-benign distributional shift |
| Needle-in-haystack @14k, 5 depths | **fp8 5/5 = bf16 5/5** | long-context retrieval preserved |

So the +52% is **quality-safe through 14k context** — correct answers, intact retrieval — with a
measurable-but-benign shift. The one documented danger (retrieval collapse beyond ~100k context) we did
*not* test, so we gate it explicitly rather than pretend. (Exact-token agreement over-flags open-ended
generation; teacher-forced KL + needle are the defensible metrics.)

## It's not vLLM-specific

A minimal **SGLang** probe (same 3B, KV-pressured batch) showed fp8 KV **+71% throughput** — the
mechanism is a property of paged-KV continuous batching, not one engine. (Throughput-only probe;
telemetry-confirmed diagnosis on SGLang is a follow-on.)

## The tool: InferPilot

Instead of sweeping knobs and hoping, InferPilot **diagnoses the bottleneck, then acts (or abstains):**

```
detect_saturation → BottleneckDiagnosis → plan_optimization (abstain unless winnable)
  → recommend_scale (which GPU/TP/precision) → CapacityAdvisory (tune|scale|accept + $/token)
  → quality gate (KL + needle) → compare_configs (fail-closed Pareto) → apply/rollback
```

On a drowning 3B/A10 deployment it emits, in one call:

> **SCALE/TUNE:** kv_capacity_bound (GPU 100%, KV full, preempting) → `kv_cache_dtype=fp8` →
> +52% goodput, TTFT −56%, $1.08→$0.71 per 1M tokens (−34%), quality-verified. One flag, zero new
> hardware.

And when the GPU is genuinely compute-bound, it says **"no config lever helps — here's the cheapest GPU
that meets your SLO."** Abstention is a first-class answer.

## Honest limits

Validated on Qwen 3B/7B/14B **and Mistral-7B** (+42%) + vLLM (and SGLang, throughput-only) + a handful of GPUs. All wins are in
overloaded regimes (goodput-ceiling framing). >100k-context fp8, free-running long-horizon drift, and a
telemetry-confirmed SGLang run are untested. This is rigorous open research + a working reasoning
library — not a finished product.

## Reproduce

```bash
uv run python scripts/cost_rescue_demo.py    # the rescue story from real evidence
uv run python scripts/fp8_law_demo.py        # diagnosis vs measured, 4/4, GPU-free
uv run --extra dev pytest -q                 # 430+ self-validating tests
```

If you self-host LLMs and have ever wondered *"should I turn on fp8, or add a GPU?"* — that's the
question InferPilot answers, with the mechanism and the dollar delta, and a refusal to fool you.
