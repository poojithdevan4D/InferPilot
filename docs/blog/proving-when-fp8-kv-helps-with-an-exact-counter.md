# I instrumented vLLM to prove exactly when fp8 KV cache helps — and failed my own GO test honestly

Most inference "optimizations" are sold as free speedups. fp8 KV cache is the classic example: "halve
your KV memory, get more throughput." Sometimes true, sometimes does nothing — and almost nobody tells
you *which*, or *why*, with evidence you can check.

I spent a few weeks building [InferPilot](https://github.com/poojithdevan4D/InferPilot) to answer that
question the un-hand-wavy way. Here is the honest result, including the part where the data refused to
give me the headline I wanted.

## The hypothesis

fp8 KV cache should help **only when the server is preempting** — i.e. when the KV cache fills, vLLM
evicts running requests and later **recomputes** their tokens from scratch. That recomputation is pure
waste. fp8 halves KV footprint, so fewer preemptions, less recompute, more real throughput. When the
server is *not* KV-bound (compute-bound, low load), fp8 changes nothing.

The discriminator, then, is **not GPU utilization** (which sits near 100% in both cases) — it's
**recomputed tokens**. But vLLM didn't expose an exact count of them. So the claim wasn't checkable.

## Making it checkable: a new vLLM metric

I added an exact recomputed-token counter to vLLM's scheduler and exposed it through the existing
Prometheus path. It counts only the token positions that are scheduled *again* after a preemption —
prefix-cache hits, which are never re-scheduled, are excluded.

That patch is now an open PR to vLLM core:
[vllm-project/vllm#57698](https://github.com/vllm-project/vllm/pull/57698).

A measurement-path canary confirmed it means what it claims: under forced KV pressure the counter read
**23,324** recomputed-token executions; a low-pressure control read **0**.

## The measured result

Six paired cells on an A10G (Qwen2.5-3B, vLLM 0.29.0, pinned reproducible image), fp8 KV vs bf16 KV,
same workload each pair:

| Block | bf16 load state | bf16 recomputed tokens | fp8 recomputed tokens | fp8/bf16 throughput |
|---|---|---:|---:|---:|
| 1 | overloaded | 29,826 | 0 | 1.360× |
| 2 | near_capacity | 7,709 | 0 | 1.118× |
| 3 | overloaded | 30,954 | 0 | 1.472× |

Geometric-mean throughput ratio: **1.308×**. Recomputed tokens went to **zero** in every fp8 cell.
TTFT, TPOT and end-to-end latency all improved; 100% request success throughout. Total GPU spend for
the whole thing, including a stopped v1 and the canary: **$0.81**.

The mechanism is exactly as hypothesized: fp8 removed the KV pressure, which removed the preemptions,
which removed the recomputation, which is where the throughput came from.

## Why I still called it NOT a GO

Before running, I registered a decision rule: to authorize a larger held-out campaign, **every** bf16
cell had to be `overloaded`, **every** recompute burden ≥10%, and **every** throughput ratio ≥1.15.

Block 2 was only `near_capacity`, its recompute burden was under 10%, and its ratio was 1.118. So the
preregistered verdict is **`NOT_TARGET_REGIME`** — promising, directionally clean, but not a pass. I
didn't move the goalposts to the numbers I got. A separate 7B study I attempted even self-invalidated
because its load window ended at 27.7s, below the 30s minimum I'd registered — so I threw it out rather
than report it.

## What this is, and isn't

**Is:** a mechanistically demonstrated, reproducible account of *why* fp8 KV helps when it helps, with
an exact counter (now proposed upstream) as the discriminator, graded against a rule I didn't bend.

**Isn't:** a causal law, cross-model generality, a quality-safety claim, or a capacity guarantee. One
model family, one GPU, and a verdict that says "not yet." The KL quality preflight actually *failed*
(real distributional shift), so I don't call fp8 KV "lossless" either.

That's the whole point of InferPilot: it diagnoses only when the evidence supports it, and abstains —
loudly — when it doesn't. If you run vLLM and want to know whether a config change will actually help
*before* you ship it, that's the tool. Traces and critique welcome.

Repo: https://github.com/poojithdevan4D/InferPilot · vLLM PR: https://github.com/vllm-project/vllm/pull/57698
