# Milestone 2 architecture — search before autonomy

## Research claim to test

InferPilot should find and safely apply an SLO-feasible inference configuration
under a changing workload using fewer measured trials and fewer SLO violations
than conventional search and a static configuration.

This is deliberately narrower than “an autonomous performance engineer.” It is
falsifiable under a fixed candidate space, workload trace, measurement budget,
and policy. An LLM is not a baseline and is not introduced until conventional
methods are measured.

## Evidence entering Milestone 2

- C5 established the open-loop scheduling trade-off under one repeated workload.
- C6 showed that one seed is insufficient and that `no_feasible_candidate` is a
  necessary decision outcome.
- C7 showed a rate-dependent decision under one fixed policy:
  - nominal 4 QPS: select `max_num_seqs=3`;
  - nominal 6 QPS: select `max_num_seqs=4`;
  - nominal 8 QPS: no robust-feasible candidate among seq1–4.
- The 4-QPS TPOT advantage of seq3 over also-feasible seq4 is only about 0.064
  ms at p95. C7 establishes that the observed optimum can change, but not that a
  disruptive live switch is worthwhile.
- `inferpilot.search` now supplies outcome-blind declared-order and seeded-random
  offline replay. It records budget exhaustion, first-feasible trial, oracle hit,
  and simple regret against a complete blocked-study report.

## Controller boundary

The controller must not directly turn an LLM suggestion into deployment state.

```text
observed request stream
        |
        v
workload feature window ----> regime descriptor + uncertainty
        |                              |
        |                              v
        |                       candidate generator
        |                              |
        v                              v
current config + SLO ----------> search policy
                                       |
                                       v
                              isolated measured trial
                                       |
                                       v
                        evidence store + compatibility gates
                                       |
                                       v
                           SLO decision / no-feasible
                                       |
                         +-------------+-------------+
                         |                           |
                    canary + rollback          escalation action
                                                (admit/scale/refuse)
```

An eventual LLM component may propose candidate families or explain evidence. A
deterministic search policy chooses measurements, and a deterministic decision
gate controls acceptance.

## Required controller actions

The action type is not simply an engine configuration. It must eventually be a
tagged decision with at least:

1. `keep_current` — expected improvement does not justify change cost;
2. `test_candidate` — run an isolated experiment, not a production mutation;
3. `apply_candidate` — only after evidence and canary gates;
4. `no_feasible_candidate` — measured space cannot satisfy the policy;
5. `escalate_capacity_or_admission` — scale, shed/admit load, or request an
   explicit product-policy change;
6. `rollback` — canary violates the declared guardrail.

C7 makes actions 1, 4, and 5 mandatory. Always returning the least-bad measured
configuration would violate the current fail-closed design.

## Workload state

Mean requested QPS is necessary but not sufficient. At minimum, a feature window
must include:

- arrival count and window duration;
- realized QPS;
- inter-arrival mean and coefficient of variation;
- maximum arrivals in fixed 250-ms, 500-ms, 1-s, and 2-s windows;
- prompt-token and requested-output-token distributions;
- shared-prefix fraction/reuse signal when prefix caching is studied;
- current in-flight/queue depth if the serving engine exposes it;
- feature-window age and sample count.

Features must be based only on observations available before a decision. Future
requests and post-completion latency cannot leak into the policy input.

The current `WorkloadSpec.seed` jointly controls prompt construction and arrivals.
Do not silently describe it as an arrival-only seed. Separating prompt corpus and
arrival schedule is desirable, but it requires an explicit contract/version plan
that preserves access to schema-0.3.0 C5–C7 evidence.

## Search protocol

Every search comparison fixes:

- candidate space and encoding;
- workload blocks and SLO;
- maximum candidate trials and measured GPU time;
- candidate measurement cost (blocks × repetitions plus startup/warm-up time);
- random-policy seeds;
- stopping rule;
- treatment of failed/OOM/timeout trials;
- oracle construction from a fully measured held-out corpus.

Primary search metrics are:

| Metric | Meaning |
|---|---|
| Feasible-found rate | Fraction of tasks where the policy finds an SLO-feasible candidate within budget |
| Trials to first feasible | Candidate evaluations consumed before any feasible action exists |
| Oracle-hit rate | Fraction where the final observed choice belongs to the complete-space optimum set |
| Simple regret | Direction-aware objective gap to the complete-space optimum |
| GPU seconds to decision | Actual experiment cost, including startup and warm-up |
| Trial SLO exposure | Violations observed during isolated/canary measurements |
| No-feasible correctness | Whether the policy distinguishes unknown-after-budget from exhaustively infeasible |
| Change cost | Restart time and disruption incurred by applying a different configuration |

`budget_exhausted_no_feasible` and `search_space_exhausted_no_feasible` are not
equivalent and must remain separate. Only exhaustive evidence can prove that a
finite measured space contains no feasible point.

## Baselines, in required order

1. Static vLLM/default configuration.
2. Declared grid order.
3. Seeded random search across enough policy seeds to report a distribution.
4. TPE/Bayesian optimization using only prior revealed observations.
5. A workload-feature heuristic derived without held-out outcome leakage.
6. Only then, an LLM-assisted hypothesis policy using the same candidate and GPU
   budgets.

The complete blocked report is available to the replay evaluator only for
post-hoc oracle scoring. A search policy must never receive unrevealed candidate
outcomes.

## Implementation sequence

### M2A — conventional replay baseline (complete)

- Exact-gated replay report `0.1.0`.
- Outcome-blind `declared_order-v1` and stable `seeded_random-v1` ordering.
- Fixed candidate budget, first-feasible trial, oracle hit, and simple regret.
- Full source report embedded so persisted results are self-validating.

### M2B — workload-feature contract (complete)

The derived, exact-gated arrival feature report `0.1.0` is computed from validated
arrival evidence. Its online-safe prefix mode uses only actual dispatch events at
or before a declared observation cutoff and includes declared token-length targets.
It preserves no completion or latency outcome. `ExperimentResult` remains 0.3.0.
Tests cover monotonicity, closed-window boundaries, zero/one/simultaneous-arrival
semantics, mismatched provenance, derived-field tampering, and future-event
exclusion. Actual per-request token distributions await an arrival-time request
metadata contract; completion-reported token counts are intentionally not used
because they would make workload state depend on serving outcomes.

### M2C — cost-aware replay

Record actual source run duration and startup/warm-up cost provenance per revealed
candidate. Add GPU-seconds budget evaluation without replacing candidate-count
budgets. A policy may not see candidate cost until that candidate is revealed
unless the cost is declared a priori.

### M2D — larger held-out corpus

Introduce at least one meaningful second engine dimension and multiple workload
shapes (prefill-heavy, decode-heavy, and burst/shared-prefix when supported).
Preregister training/search tasks separately from held-out evaluation tasks. The
current four-point `max_num_seqs` space is sufficient to test replay mechanics,
not to claim superior search.

### M2E — adaptive policy and canary state machine

Implement explicit controller actions, restart/change cost, canary acceptance,
and rollback. Compare a feature heuristic and TPE before adding LLM proposals.

The M7 controller boundary implements the first, deliberately offline part of this
step: evidence-bound decisions require repeated agreement before an isolated
`test_candidate`, and only an explicit canary outcome can produce
`apply_candidate` or `rollback`. Its complete transition replay is self-validating.
It does not yet execute restarts, define canary metrics, or claim online adaptation.

## Stop conditions and anti-claims

- If a simple conventional method matches or beats an LLM policy at the same
  budget, report that result; do not reinterpret the benchmark.
- Do not claim online adaptation from offline replay.
- Do not claim queue stability from a 30–60-second bounded workload.
- Do not infer a service-capacity ceiling from throughput when completions merely
  track offered load.
- Do not relax an SLO after seeing evidence and reuse the same evidence as
  confirmation.
- Do not claim generality from the RTX 3050 / Qwen2.5-0.5B corpus.
