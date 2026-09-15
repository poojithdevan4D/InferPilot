# M2D binding-token-budget development follow-up — results

All **12/12** cells passed every preregistered validity gate. No OOM, timeout, fidelity drift, or
lifecycle error occurred. Total server-process occupancy was **0.213 hours**. This remains
development evidence; no held-out outcomes were collected.

| seq | token budget | TTFT p95 mean [range] ms | TPOT p95 mean [range] ms | Throughput mean tok/s |
|---:|---:|---:|---:|---:|
| 1 | 512 | 994.12 [468.09, 1987.60] | 6.254 [6.247, 6.258] | 61.93 |
| 1 | 2048 | 930.38 [452.92, 1850.03] | 6.257 [6.250, 6.266] | 62.03 |
| 4 | 512 | 228.11 [174.92, 311.23] | 19.139 [18.794, 19.557] | 62.03 |
| 4 | 2048 | 225.42 [171.51, 327.73] | 21.076 [20.790, 21.382] | 62.04 |

At width 4, 512 reduced TPOT p95 by **9.19% on average**, with reductions of 8.54%, 8.29%, and
10.75% in the three blocks. Throughput was unchanged. At width 1, TPOT was unchanged. TTFT varied
substantially with the finite arrival schedules and showed no consistent token-budget direction.

The preregistered interaction hypothesis is therefore supported descriptively: a binding token
budget changes the width-4 latency trade-off but not the width-1 TPOT. This establishes a meaningful
second search dimension on this workload. It is not a significance or deployment claim.

Before held-out evaluation, collect the same explicit-chunked 512/2048 grid for the development
decode and burst shapes. This is necessary because their earlier development evidence did not test
512 and cannot validate a common held-out candidate grid. Only then freeze per-shape SLOs, search
budgets, and policy before generating/opening held-out outcomes.
