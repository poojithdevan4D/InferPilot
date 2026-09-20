# InferPilot offline optimizer: product boundary v0.1

InferPilot's first operator-facing product loop accepts an immutable benchmark bundle, an explicit
SLO, and local GPU economics. It emits an `OptimizationEvidenceCard`: a portable record of what was
observed, what InferPilot concludes, one experiment worth running next, its approximate cost, and the
conditions that would accept it.

This is deliberately an **offline decision-support tool**, not an autonomous controller.

## Input

`inferpilot analyze` consumes a runner bundle directory containing `result.json`. If present, it also
uses `phases.json` for full server-occupancy cost and `mechanism-evidence.json` for evidence strength.
The result contains configuration, token counts, monotonic timings, aggregate latency/throughput,
GPU/KV telemetry, and aligned request/queue/token evidence. It contains no prompt or completion text.

The operator must supply at least one SLO and the actual GPU-hour price. InferPilot does not invent
either business constraint.

```bash
inferpilot analyze runs/my-run \
  --ttft-p95-ms 500 \
  --tpot-p95-ms 40 \
  --min-throughput-tokens-per-s 150 \
  --gpu-cost-per-hour 1.10 \
  --target-qps 4 \
  --output evidence-card.json
```

## Output semantics

The Evidence Card has three possible statuses:

- `keep_current`: the measured cohort met the supplied constraints. This applies only to that
  workload and observed window; it is not a capacity or headroom claim.
- `experiment_recommended`: aligned evidence supports one known candidate. The card estimates the
  cost of a fresh-server baseline/candidate pair and freezes SLO, fidelity, load-evidence, quality,
  long-context, and Pareto acceptance gates. It does not apply the candidate.
- `abstain`: evidence is missing, inconclusive, unhealthy without a validated lever, or maps to no
  executable candidate. The reasons are preserved instead of converting uncertainty into advice.

The current executable mapping is intentionally narrow: aligned overload plus high KV occupancy and
observed preemptions may nominate `kv_cache_dtype=fp8`. The recommendation remains a hypothesis to
test on the exact model, hardware, engine, and workload. No claim transfers automatically.

## Cost meaning

`observed_gpu_cost_per_million_output_tokens_usd` is raw measured GPU cost divided by successful
output tokens. It is not quality-adjusted and is not a capacity estimate. Experiment cost uses two
observed full server occupancies when `phases.json` exists; otherwise it is explicitly labelled a
measured-window lower bound.

## Privacy and deployment boundary

The card embeds metadata-only evidence and can be reviewed or shared without request text. That does
not by itself establish a formal privacy guarantee: operators must still inspect any custom fields
and environment metadata before exporting a bundle.

Version 0.1 does not capture live production traffic, deploy a config, monitor a canary, or roll back
a service. Those integrations should be added only after the offline workflow proves that engineers
can repeatedly get a useful, trustworthy next experiment from their own evidence.
