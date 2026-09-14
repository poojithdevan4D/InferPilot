# C7 preregistration — scheduler width across workload-rate regimes

**Preregistered:** 2026-09-14, before any C7 measurement

**Status:** protocol fixed; results pending

## Research question

Does the SLO-feasible, TPOT-minimizing `max_num_seqs` setting change as nominal
open-loop load moves from 4 to 6 to 8 QPS? A change would establish the minimal
empirical premise for a workload-aware controller; a constant answer would
falsify that premise for this workload and search space.

## Fixed design

Three independent blocked studies share the same experimental policy:

| Item | Preregistered value |
|---|---|
| Nominal rates | 4, 6, and 8 QPS |
| Unseen workload seeds | Consecutive seeds `6,7,8` at every rate |
| Candidate | `max_num_seqs ∈ {1,2,3,4}` |
| Workload | 256 measured + 4 warm-ups, ~128 input / exactly 32 output tokens |
| Runs | One per rate/seed/candidate cell; 36 total |
| SLO | TTFT p95 ≤ 250 ms and TPOT p95 ≤ 8.5 ms in every block |
| Objective | Minimize mean block-level TPOT p95 among robust-feasible candidates |
| Other conditions | Identical to the pinned C5/C6 model, engine, and RTX 3050 setup |

The SLO is the same fixed pilot-informed research policy used for C6. It is not a
customer SLA or deployment guarantee. C7 does not use C6 seeds as evidence.

## Hypotheses fixed before measurement

1. Increasing arrival rate will weakly increase the minimum scheduler width
   needed to satisfy the TTFT constraint.
2. At least two rate regimes will produce different selected widths or decision
   statuses, establishing an adaptation opportunity.
3. The 8-QPS study may again return `no_feasible_candidate` because C6 placed
   seq3 and seq4 on opposite sides of the TTFT/TPOT constraints.

Failure to observe hypotheses 1 or 2 is a valid negative result; thresholds and
candidate space will not be changed after measurement.

## Schedules known before inference

| Nominal QPS | Seed | Scheduled span (s) | Realized finite-schedule QPS |
|---:|---:|---:|---:|
| 4 | 6 | 64.285056 | 3.966707 |
| 4 | 7 | 61.625672 | 4.137886 |
| 4 | 8 | 66.383381 | 3.841323 |
| 6 | 6 | 42.856704 | 5.950061 |
| 6 | 7 | 41.083781 | 6.206829 |
| 6 | 8 | 44.255587 | 5.761984 |
| 8 | 6 | 32.142528 | 7.933415 |
| 8 | 7 | 30.812836 | 8.275772 |
| 8 | 8 | 33.191691 | 7.682646 |

For a given seed, `poisson-v1` uses the same exponential random variates scaled
by rate. The workload seed also changes prompt content, so blocks represent
whole synthetic workload samples rather than arrival-only samples.

## Execution and interpretation

The committed driver interleaves rates and candidate order. Every accepted cell
must be COMPLETED, baseline-eligible, 256/256 successful, configuration-verified,
telemetry-complete, lifecycle-clean, and have complete arrival evidence. Dispatch
drift gates remain p95 ≤ 10 ms and individual maximum ≤ 250 ms.

Each rate is evaluated independently with the exact-gated blocked-study report
0.1.1. Cross-rate conclusions compare decision outcomes, not raw cohorts; rate is
a real workload change and is never removed from a compatibility fingerprint.
There is one observation per cell and only three blocks, so results remain an
observed-sample characterization without significance, confidence, long-horizon,
or deployment-safety claims.
