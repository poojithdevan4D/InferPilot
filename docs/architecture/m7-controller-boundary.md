# M7 controller boundary

## Purpose

M7 adds a deterministic decision boundary between the passive workload advisor and any future
serving-system executor. It does **not** mutate vLLM. This separation is required because changing
`max_num_seqs` may require a disruptive server restart, and a noisy rate estimate must never trigger
that restart directly.

The controller consumes self-validating `ProfileAdvisorDecision` objects and explicit canary results.
It emits only four actions:

- `keep_current` — no external action;
- `test_candidate` — ask an external executor to test the candidate; do not apply it yet;
- `apply_candidate` — emitted only after an explicit passing canary result;
- `rollback` — emitted after an explicit failing canary result.

## State and invariants

The exact advisor policy is embedded in `ControllerSpec`; matching a human-readable policy ID is not
enough. `ControllerState` records the current overrides, a repeated pending recommendation, an
outstanding canary, and cooldown windows. A pending recommendation and an outstanding canary cannot
coexist.

The default mechanics require three consecutive profile windows to recommend the same non-current
candidate before `test_candidate`. An abstention, a recommendation for the current configuration, or
a different candidate resets this count. While a canary is outstanding, later observations cannot
apply or replace it. Passing and failing canaries both start a two-window cooldown.

The values three and two are conservative **mechanism defaults, not empirically validated control
parameters**. They remain explicit in every `ControllerSpec` and must be preregistered before a
controller evaluation.

## Replay and tamper resistance

`ControllerReplay` embeds the initial state, exact spec, ordered input events, every transition, and
the final state. On load it recomputes the entire trace. Changing an action, reason, candidate, state,
policy, decision, or canary outcome invalidates the artifact.

This proves deterministic state-machine behavior only. It does not prove that profile windows are
independent, that a canary is representative, that restart is safe, or that the M6 steady-state rate
bands remain valid during transitions.

## Next experiment

Before connecting an executor, preregister an offline transition-trace study with alternating,
ramping, gap, and sustained-low/high workloads. Fix window length, minimum observations, controller
parameters, canary acceptance criteria, and restart cost. Compare the controller with immediate
switching and a static configuration on false-switch count, time to stable action, abstention rate,
SLO exposure, and estimated restart cost.

M8 subsequently introduced controller/event v0.2, which replaces the unaudited canary boolean with
measured `CanaryEvaluation` evidence. The v0.1 replay format remains readable but should not be used
for new controller studies.
