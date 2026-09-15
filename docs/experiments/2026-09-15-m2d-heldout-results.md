# M2D held-out binding-grid results

**Status:** sealed collection complete; 36/36 cells accepted on first attempt. The held-out seal was
opened only after collection ended and bundle integrity was checked. Total server-process occupancy
was 4009.36 s (1.1137 h; mean 111.37 s/cell). Maximum dispatch-drift p95 was 4.591 ms against the
preregistered 10 ms gate.

Candidate indices follow the frozen corpus: 0=`seq1/tok512`, 1=`seq1/tok2048`,
2=`seq4/tok512`, 3=`seq4/tok2048`. Values below are the range across held-out arrival seeds 50–52;
feasibility uses the preregistered worst-observed rule in every block.

| Shape | Candidate | TTFT p95 range (ms) | TPOT p95 range (ms) | Feasible blocks | Robust |
|---|---:|---:|---:|---:|---:|
| prefill | 0 | 347.94–914.82 | 6.265–6.296 | 1/3 | no |
| prefill | 1 | 331.26–857.64 | 6.274–6.283 | 1/3 | no |
| prefill | **2** | **159.40–241.90** | **15.616–19.648** | **3/3** | **yes** |
| prefill | 3 | 147.95–234.97 | 16.879–20.834 | 1/3 | no |
| decode | 0 | 39681.22–77314.11 | 6.275–6.293 | 0/3 | no |
| decode | 1 | 39834.74–77538.92 | 6.286–6.291 | 0/3 | no |
| decode | **2** | **31.85–780.19** | **7.028–7.080** | **3/3** | **yes, rank 1** |
| decode | 3 | 30.82–783.48 | 7.020–7.095 | 3/3 | yes, rank 2 |
| burst | 0 | 3190.76–7238.93 | 6.276–6.279 | 0/3 | no |
| burst | 1 | 3220.64–7249.84 | 6.286–6.289 | 0/3 | no |
| burst | **2** | **281.62–895.16** | **7.812–7.862** | **3/3** | **yes, rank 1** |
| burst | 3 | 286.66–893.57 | 7.815–8.175 | 3/3 | yes, rank 2 |

## Result

Candidate 2 (`max_num_seqs=4`, `max_num_batched_tokens=512`) is the unique held-out selection for
all three shapes. The strongest confirmation is prefill: candidate 2 alone remains feasible across
all seeds; candidate 3 breaches the frozen 20 ms TPOT ceiling in two blocks. For decode and burst,
both width-4 candidates are robust-feasible, but candidate 2 has the lower mean TPOT objective. The
decode difference between candidates 2 and 3 is only 0.00145 ms, so it is a deterministic ranking
under the contract, not evidence of a practically meaningful token-budget effect.

The held-out evidence confirms that width 1 is unsafe for these arrival regimes and that the binding
512-token budget protects prefill TPOT without harming robust feasibility. It does **not** support a
need for a workload-specific candidate on this grid and hardware: the same static candidate wins all
three tasks.

## Preregistration limitation

The corpus froze candidate budgets and the hypothesis that a workload-aware policy would outperform
a static candidate, but it did not freeze the algorithms that learn those two policies or the exact
cross-task scoring rule. That hypothesis therefore cannot be tested confirmatorily from this held-out
corpus without a post-hoc policy definition. Any replay comparison designed now must be labelled
exploratory. The per-shape SLO evaluations above remain valid because their candidates, blocks, SLOs,
objective, acceptance rules, and stopping rules were all fixed before measurement.

No statistical significance, production SLO, deployment safety, other-model, other-GPU, or
sustainable-capacity claim is made.
