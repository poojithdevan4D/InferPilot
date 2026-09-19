# fp8 held-out v1 — execution-readiness audit

**Status:** BLOCKED before GPU execution  
**Protocol:** `2026-09-19-fp8-heldout-preregistration.yaml`  
**Audit date:** 2026-09-19  
**Outcome data inspected:** none

This audit asks one question: can the current InferPilot runner collect every input needed to answer
the frozen protocol's primary, mechanism, and quality claims? The answer is **no**. Running the matrix
now would create performance measurements but could not test the registered recomputation mechanism.
No GPU run should start from this protocol until a versioned executable protocol closes the blockers
below.

## Requirement-to-implementation matrix

| Registered requirement | Current support | Verdict |
|---|---|---|
| deterministic paired prompt/arrival seeds | split `prompt_seed` / `arrival_seed`, replay digest | ready |
| open-loop arrivals and dispatch drift | scheduled + actual offsets in `arrivals.json` | ready |
| ≥90 s, ≥100-request aligned window | `LoadEvidence`, 5 s bins, request conservation | ready with qualifying configs |
| waiting/running queue, KV, preemption count | sampled from vLLM `/metrics` | ready, capability must be checked per pinned release |
| fixed-length prompt/output workloads | deterministic generator | ready |
| lognormal prompt/output distributions | `WorkloadSpec` accepts only fixed lengths | **blocked** |
| per-request queue-entry and first-scheduled timestamps | client sees send/first-token/end only | **blocked** |
| true per-token ITL p95 | only TTFT, E2E, and request-level mean TPOT are retained | **blocked** |
| scheduled prefill/decode/recompute tokens | not collected | **blocked** |
| preempted/recomputed token executions | only preemption event-count delta is collected | **blocked** |
| effective batch-size time series | not collected | **blocked** |
| prefix hits and KV block events | not collected | **blocked** |
| monotone isotonic capacity estimator + confidence rules | no registered-analysis implementation | **blocked** |
| leave-one-family-out classifier evaluation | no frozen analysis implementation | **blocked** |
| SGLang transportability subset | throughput-only probe; no semantic telemetry adapter | **blocked; optional first budget cut** |

## Why preemption count is not recomputation burden

The existing `vllm:num_preemptions*` counter records events, not the amount of discarded and repeated
work. One preemption after a long partial sequence can be more expensive than several early ones. The
registered threshold `recompute_burden >= 0.10` therefore cannot be reconstructed honestly from the
current result schema.

vLLM's current metrics design documents scheduler events and aggregate prompt-token sources, but the
public Prometheus surface does not guarantee the exact per-iteration recompute accounting registered
here. The upstream per-iteration telemetry request explicitly describes this gap:

- [vLLM metrics design](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md)
- [vLLM per-iteration scheduler telemetry request](https://github.com/vllm-project/vllm/issues/38760)
- [vLLM metric logger implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/loggers.py)

Metric names and semantics must be verified against the **pinned vLLM release/revision**, not inferred
from `main`. A prompt-token counter may support a clearly named excess-prefill-work proxy, but that is
not interchangeable with exact recomputed-token executions and cannot be substituted after outcomes
are observed.

## Required resolution before execution

1. **Choose the mechanism measurement before collecting outcomes.** Either:
   - instrument the pinned vLLM scheduler to emit exact computed/recomputed token counters, or
   - preregister a v2 protocol around a precisely defined observable proxy, with different field and
     claim names. Do not label a proxy `recompute_burden` without validation.
2. **Run a metrics-capability preflight only.** Start each pinned engine version, capture the exposed
   metric families, and verify counter monotonicity/reset behavior. This is instrumentation QA, not an
   outcome run. The name-level, fail-closed preflight is implemented as
   `python -m inferpilot.runner.metrics_capabilities http://127.0.0.1:8000`; semantic validation of
   any newly instrumented counters remains required.
3. **Make the workload executable.** Add deterministic per-request length traces and bind their digest,
   or replace the lognormal language in a versioned protocol with fixed lengths already supported.
4. **Choose ITL or TPOT.** If ITL p95 remains an SLO, retain monotonic timestamps for every output token
   and derive ITL from raw intervals. Otherwise preregister request-level TPOT and change the SLO name;
   they are not equivalent.
5. **Implement the frozen analysis before opening results.** Capacity bracketing, SLO confidence bounds,
   paired cluster bootstrap, TOST, dose-response, exclusions, and counterexample tables must be code,
   tested on synthetic data, and committed before the first outcome cell.
6. **Generate an exact run manifest.** It must bind configs, randomized arm order, repetitions, seeds,
   retry/acceptance/stopping rules, engine/model revisions, quality tasks, and the hard dollar stop.

## Minimum pilot that is allowed

A short, explicitly labeled **instrumentation pilot** may verify only:

- required metric families exist on the pinned release;
- counters reset at fresh-server boundaries and remain monotonic;
- aligned evidence covers the intended measurement interval;
- per-token timestamps and length-trace replay round-trip;
- the runner can stop at the configured cost limit.

Its performance outcomes must not be used to change thresholds, load multipliers, SLOs, classifier
rules, or exclusions in the held-out study. If pilot outcomes inform those choices, the subsequent
study must declare a fresh held-out partition.

## Exit criteria for READY

The study may move to `READY` only when all core-matrix rows in the table above are implemented and
covered by synthetic/fault-injection tests, the exact analysis command reproduces a synthetic golden
report, and a manifest dry run proves the registered cell count and budget stop. Optional
transportability cells may remain excluded only under the already registered budget rule.
