# M6 advisor rate-band confirmation

## Result

The frozen M3 actions passed the preregistered SLO at every independently held-out boundary cell:

| Validated closed interval | Frozen action | Held-out seeds | Worst TTFT p95 | Worst TPOT p95 |
|---|---:|---:|---:|---:|
| 1.5–2.5 QPS | `max_num_seqs=2` | 93, 94, 95 | 154.21 ms | 7.174 ms |
| 5.5–6.5 QPS | `max_num_seqs=4` | 93, 94, 95 | 134.31 ms | 7.926 ms |

The fixed SLO was TTFT p95 ≤ 250 ms and TPOT p95 ≤ 8.5 ms. All 12 held-out cells were accepted on
their first attempt. The full self-validating evidence is in `evidence/m6-rate-bands.json`; every cell
names its immutable store run ID.

## Advisor consequence

Policy v0.2 may recommend width 2 anywhere inside 1.5–2.5 QPS and width 4 anywhere inside 5.5–6.5
QPS. Both endpoints are inclusive. It must abstain in the unmeasured gap and outside the outer
boundaries. The original M3 report remains the evidence that selected the actions; M6 is separate
evidence that bounds where those actions remained feasible.

## Claim boundary

This confirms SLO feasibility at the sampled endpoints and supports a deliberately bounded advisor
rule. It does not prove that each action is optimal at every interior rate, that all possible arrival
realizations are safe, or that the result generalizes beyond the pinned model, revision, GPU, fixed
128/32-token Poisson workload, and engine configuration.
