# M8 measured canary contract

## Closed trust gap

Controller v0.1 accepted a caller-supplied boolean canary outcome. That format remains readable, but
it cannot establish why a candidate passed. Controller v0.2 instead requires a complete
`CanaryEvaluation`; passing or rollback is derived from measured `ExperimentResult` evidence.

The controller and every event require an explicit version, preventing a caller from accidentally
falling back to the boolean format.

## Acceptance gates

A v0.2 canary passes only when all of the following hold:

- the originating profile decision recommends a candidate;
- the run is baseline-eligible (complete, successful, verified effective config, complete telemetry);
- model revision, GPU, token lengths, arrival algorithm, and requested rate match the decision;
- every recommended engine override matches both requested and effective engine configuration;
- raw request measurements recompute to the stored aggregate metrics;
- TTFT p95 and TPOT p95 satisfy the declared canary SLO.

The full decision and result are embedded. Loading a persisted evaluation or controller replay
recomputes all gates and rejects changed metrics, reasons, outcome, candidate, policy, or state.

## Current boundary

This contract evaluates evidence; it does not launch a canary, route production traffic, restart
vLLM, or verify a store bundle manifest. An executor must preserve the run bundle and pass its loaded
`ExperimentResult` into the evaluator. The current absolute SLO gate does not compare against a
concurrent control and therefore cannot distinguish candidate regressions that remain under the SLO.

The next experimental step is to preregister the canary workload duration/repetitions and determine
whether a short canary predicts the longer M6 measurement. Until that validation exists, v0.2 is an
auditable safety mechanism, not evidence that short canaries are sufficient for deployment.
