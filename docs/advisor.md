# Evidence-bound advisor

InferPilot's first advisor is intentionally fail-closed. It recommends an engine override only when
model revision, GPU name, prompt/output lengths, arrival algorithm, and measured rate match a
validated policy regime. Policy v0.2 accepts rates only inside independently confirmed closed
intervals; it does not extrapolate or claim generality beyond the evidence.

```bash
python -m inferpilot.advisor policies/m3-rtx3050-qwen05b.json \
  runs/m3-heldout/policy-report.json examples/advisor_request.json \
  --applicability-evidence evidence/m6-rate-bands.json
```

The included policy returns width 2 from 1.5–2.5 QPS and width 4 from 5.5–6.5 QPS, inclusive, with
the fixed 512-token batching budget. Rates in the gap or outside both bands abstain. The CLI verifies
both the original optimization report and the separate M6 applicability report against SHA-256
digests before recommending. This is a research demonstrator, not an automatic production controller.

## Automatic workload profiling (M5)

Passive request observations can be characterized into a profile and fed to the same fail-closed
advisor. The end-to-end workflow is:

```
observations (JSON/JSONL)                 # factual: arrival offset, prompt/output tokens, success
  → python -m inferpilot.profiling obs.jsonl profile.json    # offline; refuses to overwrite
  → WorkloadProfile                        # self-validating, provenance-digested descriptive stats
  → advise_from_profile(policy, profile, context)            # policy verified vs its evidence report
  → ProfileAdvisorDecision                 # recommended override, or a reasoned abstention
```

Explicitly:

- **Profiling is passive characterization, not workload classification.** The profile reports counts,
  realized rate, prompt/output p50/p95, inter-arrival p50/p95, CV, and simultaneous-arrival fraction.
  It never labels traffic "Poisson", "burst", or any other class.
- **Only evidence-supported contexts receive a recommendation.** The observed realized rate must be
  inside a validated closed interval. The adapter never rounds, maps to the nearest regime, bridges
  the untested 2.5–5.5 QPS gap, or extrapolates past a boundary.
- **Unsupported or mixed workloads abstain.** A rate outside the two validated bands abstains
  (`unsupported_request_rate_qps`); mixed prompt/output lengths abstain
  (`mixed_prompt_lengths` / `mixed_output_lengths`); a workload not declared fixed-length abstains.
  Token summaries are never silently treated as fixed request lengths.
- **Provenance-bound and tamper-evident.** The decision binds to the profile's SHA-256 provenance
  digest; the profile recomputes every derived field on load, and both the profile and the decision
  reject tampering.

### Validity scope

The current policy is validated **only** for the pinned `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, the **RTX 3050 Laptop GPU**, **fixed 128/32-token
`poisson-v1`** workloads, and the **1.5–2.5 QPS and 5.5–6.5 QPS** closed intervals. Anything else
abstains. Endpoint evidence demonstrates SLO feasibility of the frozen action, not optimality at every
interior rate. This is
**not** a production autoscaler and provides **no deployment-safety guarantee**.

## Controller boundary (M7)

`inferpilot.advisor.controller` adds a pure state machine for consuming profile decisions. By default,
three consecutive windows must agree before it emits `test_candidate`; the current configuration does
not change until an explicit passing canary result produces `apply_candidate`. Failure produces
`rollback`, and both outcomes start a declared cooldown. Complete event/transition replays recompute
themselves on load and reject tampering.

This is an offline control contract, not a live vLLM integration. Window length, canary measurement,
restart orchestration, and production rollback remain external and unvalidated. See
`docs/architecture/m7-controller-boundary.md`.
