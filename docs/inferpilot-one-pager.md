# InferPilot — evidence-first diagnosis for LLM inference

## The problem

LLM-serving changes are easy to benchmark badly. A faster run may have used different prompts,
arrivals, engine behavior, or failed requests. GPU utilization alone does not identify a bottleneck,
and an overloaded result cannot certify capacity or cost under an SLO.

InferPilot turns benchmark evidence into auditable decisions. It verifies configuration and workload
identity, preserves raw measurements, binds derived reports to their inputs, and abstains whenever the
evidence cannot support a diagnosis.

## What is implemented

- A reproducible vLLM runner with deterministic workloads, open- and closed-loop arrivals, warm-up,
  monotonic timings, telemetry, cleanup, and immutable run bundles.
- Aligned `LoadEvidence`: arrivals, exits, failures, useful token demand/delivery, queue state, GPU/KV
  samples, and preemption deltas share one measured window and obey request conservation.
- Conservative load assessment with four outcomes: `healthy`, `near_capacity`, `overloaded`, and
  `indeterminate`.
- Strict comparison, SLO, blocked-study, search-replay, cost, quality, and controller contracts whose
  persisted conclusions are recomputed and checked when loaded.
- Honest failure artifacts, invalid-study preservation, exact schema gates, provenance fingerprints,
  and a locked GPU-free test/build path.

## The research finding

Across a small legacy matrix of Qwen2.5 3B/7B/14B and Mistral-7B runs on A10/A100 GPUs, fp8 KV was
associated with **+40–53% throughput** in KV-full, preempting cases and **+1.7%** in a no-preemption
case. The strongest 3B/A10 result measured **+52% goodput, TTFT −56%, and cost/output-token −34%**.

This is an empirical, post-hoc heuristic—not a causal law or a validated prediction. The quality
smoke test also found a real distributional shift: factual QA and 14k needle checks showed no detected
regression, but the teacher-forced-KL gate failed.

## The engineering result that matters

Expert review exposed that the first diagnosis path inferred too much from aggregate latency and
GPU/KV snapshots. The response was not to defend the headline. The project was redesigned so new runs
collect aligned request, queue, token, and preemption evidence. Legacy runs remain valid measurements,
but the current advisor returns `unknown` on them because the necessary evidence does not exist.

That gives the project a falsifiable boundary:

```text
complete aligned evidence + conserved request flow
  -> load assessment
  -> diagnosis when supported

missing or contradictory evidence
  -> indeterminate
  -> no recommendation
```

Held-out diagnostic validation on redesign-native GPU evidence is the next proof point.

## Why this is a useful engineering artifact

- **Experimental discipline:** deterministic arrivals and prompts, declared varied fields, blocked
  seeds, preregistered execution/acceptance rules, and preserved negative results.
- **Systems correctness:** monotonic clocks, always-cleanup lifecycle, structured failures, engine
  configuration verification, and measured-window provenance.
- **Data-contract rigor:** exact version gates, backward compatibility tests, tamper detection, and
  recomputation of persisted decisions.
- **Research honesty:** measured observations, mechanism hypotheses, and advisor claims are kept
  separate; insufficient evidence becomes abstention.
- **Operational framing:** comparisons optimize goodput/cost under an explicit SLO and keep model
  quality as a mandatory independent gate.

## Five-minute reviewer path

```bash
uv sync --extra dev --locked
uv run python scripts/instrumentation_dry_run.py
uv run python scripts/aligned_load_demo.py
uv run python scripts/fp8_law_demo.py
uv run python scripts/cost_rescue_demo.py
uv run --extra dev pytest -q
```

Then read:

1. `docs/blog/when-does-fp8-kv-actually-help.md` — result, critique, and claim boundary.
2. `docs/CRITIQUE-RESPONSE.md` — independent objections and the resulting redesign.
3. `src/inferpilot/saturation.py` — fail-closed load-state contract.
4. `src/inferpilot/runner/load_evidence.py` — aligned evidence construction.
5. `docs/experiments/` — preregistrations, studies, invalid runs, and honest negatives.

## Current boundary

InferPilot is a research-grade offline benchmarking and decision library. It is not a production
control plane, its fp8 hypothesis has not passed a preregistered held-out diagnostic campaign, and its
test suite proves implementation/contract behavior rather than deployment-wide empirical accuracy.
