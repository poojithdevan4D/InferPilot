# M2D held-out corpus — frozen preregistration

Development evidence is complete. Before any held-out run, this commit freezes the 2×2 candidate
grid (`max_num_seqs={1,4}`, `max_num_batched_tokens={512,2048}`, explicit chunked prefill), three
new prompt samples (2001/2002/2003), and arrival seeds 50/51/52.

Pilot-informed policies are: prefill TTFT p95 ≤400 ms and TPOT p95 ≤20 ms; decode TTFT p95 ≤1500
ms and TPOT p95 ≤8 ms; burst TTFT p95 ≤1500 ms and TPOT p95 ≤8.5 ms. Objective is TPOT p95. These
thresholds are experimental policies derived from development data, not product SLOs. Candidate
budgets are 1, 2, and 4; `budget_exhausted_no_feasible` is not exhaustive infeasibility.

The exact self-validating corpus is `experiments/m2d-heldout/corpus.json`; its generator binds all
development and held-out config positions, contexts, explicit seeds, and partitions. Held-out files
must not be executed or inspected for outcomes until an independent read-only review confirms the
freeze. Total held-out measurement is 36 cells. No production, significance, safety, or generality
claim is authorized.

The policy thresholds are frozen from the accepted development evidence in
`2026-09-15-m2d-binding-followup-results.md` (prefill) and
`2026-09-15-m2d-development-complete-results.md` (decode and burst). Each block has exactly one
run per candidate (`min_runs=1`).

## Sealed execution protocol

The 36 cells must run in this exact deterministic order. For each arrival seed in `50, 51, 52`, in
that order, visit candidates in corpus order: `(seq1,tok512)`, `(seq1,tok2048)`,
`(seq4,tok512)`, `(seq4,tok2048)`. For each candidate, run shapes in the order `prefill`, `decode`,
`burst`. Thus the first three cells are the three shapes at seed 50 with `(seq1,tok512)`, the next
three use `(seq1,tok2048)`, and so on; seed 51 begins only after all 12 seed-50 cells, followed by
seed 52. No adaptive reordering is allowed.

A cell is accepted only when it is baseline-eligible, has the requested effective configuration
verified, complete telemetry, no pre-teardown lifecycle error, all 128 measured requests successful,
and `arrivals.json` dispatch-drift p95 is at most **10 ms**. Raw bundles and every attempted run must
be preserved.

One retry of the identical config is allowed (two attempts maximum) only when the sole failed
acceptance gate is `dispatch_drift_exceeded`. Do not inspect latency, throughput, or other outcome
metrics before making that retry. No seed substitution, config change, extra repetition, or retry for
any other reason is allowed.

A non-retryable rejection, or a second dispatch-drift rejection, invalidates that shape's held-out
task. Stop all later cells for that shape, preserve the evidence, and do not inspect its outcome
metrics or collect replacement data. The other independently preregistered shape tasks may continue
in their fixed relative order. Selection, ranking, and hypothesis evaluation occur only after data
collection ends; partial outcomes must not change execution or stopping decisions.
