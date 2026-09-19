# When does fp8 KV cache actually help? A measured pattern—and a stricter test

fp8 KV cache is often described as a near-free throughput win: halve the KV memory, serve more,
done. Our measurements were more interesting. fp8 was a large win in some overloaded deployments
and effectively noise in another. The difference tracked scheduler preemption, not GPU utilization.

That result is useful, but it is not yet a law. It came from a small, partly post-hoc experiment
matrix with single runs and no preregistered held-out diagnostic test. This article separates what we
measured, the mechanism we hypothesize, and what remains to be proved.

## What we measured

We compared `kv_cache_dtype=fp8` with bf16 KV on Qwen2.5 3B, 7B, and 14B across A10 and A100 GPUs,
plus a throughput-only Mistral-7B-v0.3 probe:

| Legacy observed condition | fp8 vs bf16 |
|---|---|
| KV full and server preempting | **+40% to +53% throughput** |
| no preemptions, shorter decode workload | **+1.7% throughput** |

GPU utilization was near 100% in both conditions. The useful association was preemption: when the
scheduler evicts sequence state and later recomputes work, reducing KV bytes can free blocks and
recover some wasted compute. With no observed preemption, there may be little to recover.

This is a mechanism hypothesis, not causal proof. In some large-model runs preemptions fell without
reaching zero, and the experiments do not isolate every competing effect. A preregistered matrix must
measure aligned load, recomputed work, and effect size across the saturation boundary before the
heuristic can be treated as predictive.

## It is a capacity lever, not a free latency win

The large gains occurred under severe overload, with TTFT between roughly 20 and 94 seconds. In the
strongest measured 3B/A10 case, fp8 increased goodput by 52%, reduced TTFT by 56%, and reduced measured
cost per output token by 34%. The server was still overloaded afterward.

The defensible interpretation is therefore: fp8 KV may raise the goodput ceiling of a KV-pressured,
preempting server. It should be evaluated as goodput under an explicit SLO—not advertised as a
universal latency speedup.

## Output quality is a separate gate

KV quantization can change model outputs. We ran four small checks on Qwen2.5-3B:

| Measurement | Result | Interpretation |
|---|---|---|
| Greedy token agreement | **12%** | outputs diverged; not itself a quality score |
| Factual QA | **0/16 regressions** | no regression detected on these short tasks |
| Teacher-forced KL | mean >0.01, **p99 0.39** | preflight failed; distribution shifted |
| Needle retrieval at 14k, five depths | **5/5 fp8 = 5/5 bf16** | no regression detected in this smoke test |

These checks do not establish that fp8 is lossless or generally safe. The KL gate failed, the task
set is small, and very-long-context behavior remains untested. Performance and quality decisions must
remain separate and fail closed.

## The expert critique changed the system

An early InferPilot prototype tried to infer saturation and bottlenecks from aggregate latency plus
GPU/KV snapshots. An external review correctly pointed out that those signals do not establish the
load state: a busy GPU can be healthy, an overloaded queue can predate the observation window, and a
finite closed-loop benchmark can hide backlog growth.

We did not relabel the old evidence to make the advisor look right. We changed the contract:

1. New benchmark runs align arrivals, successful and failed exits, useful token demand/delivery,
   queue samples, and preemption deltas to one measured window.
2. Request conservation must hold across that window.
3. “Overloaded” requires persistent backlog or useful-work deficit, not a utilization threshold.
4. Missing, incomplete, or contradictory evidence produces `indeterminate` and no recommendation.
5. Legacy bundles still reproduce the measured fp8 association, but the redesigned advisor returns
   `unknown` because they predate the required aligned evidence.

That negative result is part of the artifact: the new implementation invalidates its predecessor's
confidence instead of laundering it through a new schema.

## What InferPilot is now

InferPilot is an evidence-first benchmarking and decision library for LLM serving. Its current path is:

```text
aligned LoadEvidence
  -> conservative LoadAssessment
  -> bottleneck diagnosis or abstention
  -> candidate plan with explicit preconditions
  -> controlled comparison under an SLO
  -> quality gate and rollback-capable decision artifact
```

The contracts, runner integration, evidence store, comparison logic, and abstention behavior exist.
The next research result must come from redesign-native GPU runs and a preregistered held-out test.
Until then, the fp8 result is a measured hypothesis—not a validated advisor win.

## Scope and limits

- The fp8 matrix covers a few models, two model families, and a handful of GPU/workload combinations.
- vLLM is fully wired; the SGLang result is throughput-only and does not validate diagnosis portability.
- Traffic is synthetic, and all large wins occurred in overloaded regimes.
- Existing GPU bundles lack the aligned evidence required by the redesigned diagnosis path.
- The test suite establishes contract and implementation behavior, not accuracy on arbitrary deployments.
- InferPilot is an offline research/canary library, not a production control plane.

## Reproduce without a GPU

```bash
uv sync --extra dev --locked
uv run python scripts/aligned_load_demo.py   # healthy / overloaded / abstain contract
uv run python scripts/fp8_law_demo.py        # legacy measurements + current evidence status
uv run python scripts/cost_rescue_demo.py    # measured economics, no retroactive diagnosis
uv run --extra dev pytest -q                 # 482 tests
```

The interesting claim is not that a script can print “turn on fp8.” It is that the project preserves
the surprising measurement, records why the first explanation was insufficient, and now has an
instrumented, falsifiable path to test the hypothesis properly.
