# FP8 KV improved throughput by 33.6%. Our hypothesis still failed.

Inference optimization produces persuasive stories very quickly. A server is overloaded, KV cache
is full, preemptions are rising, and switching the KV cache to FP8 increases throughput. It is
tempting to declare the mechanism solved.

We preregistered a stricter test with InferPilot and learned why that is dangerous.

## The result

We tested Qwen2.5-7B-Instruct on one NVIDIA A10G with vLLM 0.29.0. The experiment contained twelve
accepted cells: three independent arrival blocks, two workloads, and two KV-cache configurations.
Everything except `kv_cache_dtype` was held fixed within each comparison.

On the long-context, KV-pressured workload:

- FP8 KV produced a **1.336× geometric-mean throughput ratio** over BF16 KV.
- TTFT p95 improved in every block.
- The server remained overloaded and continued preempting.

On the short-context control workload:

- FP8 KV produced a **1.0017× throughput ratio**: effectively no change.
- Neither configuration preempted or executed recomputed tokens.

That contrast is useful. FP8 helped under KV pressure and did essentially nothing where KV pressure
was absent.

## Why the verdict was still `INCONCLUSIVE`

Before measuring the outcome, we registered a more specific mechanism claim: FP8 should reduce the
fraction of token executions spent recomputing evicted sequences on the pressured workload.

It did not. Median recompute-burden reduction was **-0.19**; in two of three blocks recomputation
increased. FP8 appears to have admitted more concurrent KV state and raised delivered throughput
without clearing overload. The performance association survived, but our registered explanation did
not.

InferPilot therefore preserved both facts:

1. FP8 KV materially improved throughput in this pressured configuration.
2. The preregistered recomputation mechanism was not confirmed.

The study verdict is `INCONCLUSIVE`, not `GO`.

## Why this is the product

InferPilot is not a collection of universal tuning rules. It separates three stages that are often
collapsed into one:

1. **Screen:** read live vLLM metrics and identify a controlled experiment worth running.
2. **Measure:** compare compatible configurations under identical offered traffic and explicit SLOs.
3. **Gate:** preserve negative evidence and abstain when the registered claim is not supported.

The live command is deliberately non-invasive:

```bash
uvx inferpilot doctor --url http://localhost:8000
```

It can say that FP8 is worth testing. It does not promise the effect size, the mechanism, or safe
deployment. Those require the controlled evidence that follows.

## Reproduce the software path without a GPU

```bash
uvx inferpilot demo
```

This runs the decision contracts on clearly labelled synthetic metadata. The complete measured
protocol, accepted cells, numerical table, integrity digest, limitations, and estimated **$1.30**
incremental GPU cost are in the
[7B repair-study report](../experiments/2026-09-22-fp8-mechanism-7b-repair-results.md).

## Scope

This is not production-traffic evidence, a cross-engine law, a quality-safety result, or deployment
authorization. It is one model, one GPU type, one engine version, and two synthetic workload shapes.
The value is not that FP8 always wins. The value is that the measurement system kept a useful win
and a failed explanation separate.

If you operate vLLM, try the read-only screen and share an anonymized
[field report](https://github.com/poojithdevan4D/InferPilot/issues/new?template=field-report.yml).
Negative results and abstentions are especially useful.
