# Engine-neutral evidence contracts; vLLM implementation today

InferPilot's persisted evidence and decision contracts are intentionally independent of a serving
engine. Request timings, workload identity, SLO checks, quality gates, and aligned load accounting do
not encode vLLM metric names or launch flags.

That does **not** make a new engine a two-line adapter. Scheduler, admission, cache, preemption,
streaming, and failure semantics differ across engines. A trustworthy integration must map and test
those meanings, not merely rename Prometheus series.

## Current boundary

- **vLLM:** full runner path—launch, readiness, request measurement, resolved-config verification,
  telemetry, aligned load evidence, lifecycle cleanup, and immutable result storage.
- **SGLang:** exploratory throughput-only probe. No aligned queue/preemption evidence and therefore no
  InferPilot diagnosis validation.
- **TensorRT-LLM:** no adapter or measured evidence.

The engine-specific surface currently includes command construction, readiness/config discovery,
streaming response parsing, telemetry semantics, and cleanup. The comparison and report layers can be
reused only after an adapter proves that it supplies equivalent evidence.

## Exploratory SGLang result

A minimal Qwen2.5-3B/A10 batch probe measured approximately 0.74 req/s with bf16 KV and 1.27 req/s
with fp8 KV (**+71%**). This shows that an fp8 throughput gain is not unique to the vLLM executable.
It does **not** confirm the proposed preemption/recompute mechanism because the probe did not collect
the required scheduler evidence.

## Requirements for a second full engine

1. Define the engine's queue, running-request, cache-pressure, eviction/preemption, and useful-token
   semantics in writing.
2. Implement launch, readiness, effective-config verification, streaming measurement, telemetry, and
   guaranteed cleanup.
3. Produce the same aligned request/token conservation evidence used by `LoadAssessment`.
4. Add fault-injection and semantic-conformance tests, not just a successful throughput run.
5. Run a preregistered cross-engine study and report disagreements rather than forcing identical labels.

The claim is therefore **engine-neutral contracts with one complete engine integration**, not a
validated engine-agnostic diagnosis system.
