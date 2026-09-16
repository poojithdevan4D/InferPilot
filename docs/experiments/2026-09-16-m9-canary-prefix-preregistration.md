# M9 canary-prefix predictive validity — preregistration

## Motivation and exploratory basis

Post-hoc analysis of 36 existing M3/M6 runs compared early prefixes with their complete 256-request
SLO outcome (TTFT p95 ≤250 ms and TPOT p95 ≤8.5 ms). A 32-request prefix produced 3 false passes;
64 produced 2; 128 produced 0. The 128-request result was 27 true passes and 9 true failures, but it
is exploratory because the prefix length was selected after inspecting these outcomes.

## Frozen confirmatory design

- Model, revision, GPU, workload shape, arrival algorithm, SLO, and engine settings are identical to
  M3: pinned Qwen2.5-0.5B, RTX 3050 Laptop GPU, fixed 128/32 tokens, `poisson-v1`, and
  `max_num_batched_tokens=512`.
- Candidate grid: rates {2, 6} QPS × `max_num_seqs` {1,2,3,4} × arrival seeds {103,104,105} = 24
  complete cells. Prompt seed is the new fixed seed 7003.
- Outcome-blind execution order is seed → width → rate, inherited from the established M3 executor.
- Acceptance/retry/stopping rules are inherited unchanged: 256/256 successful, baseline eligible,
  effective configuration verified, complete telemetry/lifecycle/phases/arrivals, dispatch-drift p95
  ≤10 ms; retry only when the sole failure is dispatch drift, at most once; any other rejection stops.
- Predictor: sort raw measurements by dispatch start and evaluate exactly the first 128. Its duration
  is the maximum prefix completion offset from the original measured-window origin. Recompute
  aggregates from those raw measurements. Prefix and full outcomes use the same frozen SLO.

## Frozen evaluation

Report the four confusion counts with **false pass** meaning prefix passes but the full run fails.
The predictor passes this bounded validation only if:

1. there are at least 3 full-run passes and 3 full-run failures (the corpus is discriminative);
2. false passes equal zero; and
3. false-fail rate among full-run passes is at most 10%.

No threshold, prefix size, seed, failed cell, or SLO may be changed after execution. If the corpus is
not discriminative, the result is inconclusive rather than successful.

## Claim boundary

This tests whether a prefix of an uninterrupted synthetic benchmark predicts that benchmark's final
binary SLO result. It does not validate a separate production canary, dynamic traffic transitions,
restart safety, deployment rollback, other workloads, or other hardware/models.
