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
