# C6 preregistration — blocked open-loop `max_num_seqs` confirmation

**Preregistered:** 2026-09-14, before any C6 measurement

**Status:** protocol fixed; results pending

## Research question

Does any `max_num_seqs` setting remain feasible across multiple independent
synthetic workload samples at a nominal 8 QPS, and, if more than one does, which
feasible setting has the lowest TPOT p95?

C5 was exploratory and used one seed repeatedly. It established measurement
repeatability conditional on that exact workload but did not measure robustness
to workload sampling. C6 is the confirmatory follow-up; C5 observations are not
included in its decision blocks.

## Fixed design

| Item | Preregistered value |
|---|---|
| Model/runtime/hardware target | Same pinned Qwen2.5-0.5B / vLLM 0.29.0 / RTX 3050 setup as C5 |
| Workload | `poisson-v1`, nominal 8 QPS, 256 measured + 4 warm-ups, ~128 input / exactly 32 output tokens |
| Independent blocks | Consecutive unseen workload seeds `1,2,3,4,5` |
| Candidate | `max_num_seqs ∈ {1,2,3,4}` |
| Repetitions | One run per seed/candidate cell (20 runs total) |
| SLO policy | TTFT p95 ≤ 250 ms **and** TPOT p95 ≤ 8.5 ms |
| Robust feasibility | Candidate must pass both limits in every seed block |
| Objective | Among robust-feasible candidates, minimize mean block-level TPOT p95 |
| Possible outcomes | selected, tie, or `no_feasible_candidate` |

The SLO is a **pilot-informed experimental policy**, fixed before C6 measurements.
It is not a customer SLA or a deployment guarantee. The 250 ms TTFT limit places
the decision near the scheduling knee observed in C5; the 8.5 ms TPOT ceiling
prevents winning by accepting an unbounded decode-latency regression. No threshold,
seed, candidate, or objective may be changed after observing C6 results.

## Deterministic schedule expectations

The arrival schedules can be computed before inference and are not outcomes.
For 256 requests at nominal 8 QPS, the fixed `poisson-v1` schedules are:

| Workload seed | Scheduled span (s) | Finite-schedule realized rate (QPS) |
|---:|---:|---:|
| 1 | 31.034193 | 8.216743 |
| 2 | 31.848433 | 8.006673 |
| 3 | 33.011951 | 7.724475 |
| 4 | 33.021636 | 7.722210 |
| 5 | 34.049859 | 7.489018 |

These were not selected for favorable outcomes: `1..5` is the preregistered
consecutive sequence after C5's exploratory seed 0.

## What a block represents

`WorkloadSpec.seed` currently seeds both prompt construction and the Poisson
schedule. A C6 block therefore represents an independent sample of the complete
synthetic workload (prompt contents plus arrivals), **not an isolated change in
arrival timing**. Length targets, vocabulary distribution, generation parameters,
and every engine condition remain fixed. This limitation must be retained in the
results report; causal claims about arrival timing alone are not allowed.

## Execution and acceptance

Execution uses a committed fixed interleaving: a Latin-square order for seeds
1–4 and a fixed reverse order for seed 5. The driver is resume-safe and accepts a
cell only when it is COMPLETED, baseline-eligible, 256/256 successful, effective
configuration verified, telemetry complete, lifecycle clean, and its arrival
artifact complete. Dispatch drift must have p95 ≤ 10 ms and maximum ≤ 250 ms.

An invalid cell aborts the study and remains preserved; it is not silently
excluded or replaced. Raw run bundles, the content-addressed store, manifest, and
machine-readable decision report remain under `runs/c6-openloop-blocked/` and are
gitignored. The human-readable outcome will cite this preregistration commit.

## Interpretation limits

Five blocks with one observation per cell provide a conservative observed-sample
robustness gate. They do not provide a confidence bound, significance test,
long-horizon queue-stability proof, or deployment-safety guarantee. In particular,
C5 estimated capacity near 7.5 requests/s, below the nominal 8 QPS mean; a
`no_feasible_candidate` result is anticipated and is scientifically valid.
