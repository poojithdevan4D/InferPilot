# fp8 instrumentation pilot — GO / NO-GO checklist

This checklist gates the six-run pilot in
`2026-09-19-fp8-instrumentation-pilot-preregistration.yaml`. It is intentionally stricter than
“the server started.” A green metric-name report is necessary but not sufficient: counter semantics
must also survive the checks below.

## Gate 0 — zero-cost repository proof

Run:

```bash
uv sync --extra dev --locked
uv run python scripts/instrumentation_dry_run.py
uv run --extra dev pytest -q
```

Required dry-run output:

- `pipeline = PASS`
- `load_evidence_present = true`
- `load_state = overloaded`
- `diagnosis_regime = kv_pressure`
- `real_pilot_gate = NO_GO_EXPECTED`

`NO_GO_EXPECTED` is correct for the fake server: it deliberately lacks exact scheduler/recompute
counters. Any other result means the local evidence chain is broken. **Do not rent a GPU.**

## Mechanism-metrics path decision — 2026-09-19

**Decision: use a small, pinned vLLM 0.29.0 instrumentation patch (path b).** Stock vLLM's
Prometheus surface is insufficient for the registered mechanism claim. Stock iteration logging is a
useful cross-check, but it is not the source of truth. Path (c), “not currently obtainable,” is also
rejected: the scheduler has the required state at the point where tokens are scheduled, so a narrow
patch can count it without changing scheduling decisions.

The audit was performed against vLLM tag `v0.29.0`, upstream commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`, not against `main`:

| Required semantic | Stock Prometheus | Stock `--enable-logging-iteration-details` | Decision |
|---|---|---|---|
| scheduled prefill/context tokens per iteration | no separate family; only combined iteration-token and request-level aggregates | yes: `context tokens` | patch counter; log cross-check |
| scheduled decode/generation tokens per iteration | no separate family | yes: `generation tokens` | patch counter; log cross-check |
| effective batch size per iteration | no exact per-iteration request-count family | yes: `context requests + generation requests` | patch histogram; log cross-check |
| exact recomputed token executions | no | no | patch counter; positive/negative canary |

The stock log fields come from [`compute_iteration_details`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/utils.py)
and are emitted by the opt-in iteration logger in
[`vllm/v1/metrics/loggers.py`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/loggers.py).
The scheduler resets a preempted request's computed-token frontier in
[`scheduler.py`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/sched/scheduler.py),
but stock metrics do not retain the old frontier or count its later re-execution. An upstream
[metrics RFC](https://github.com/vllm-project/vllm/issues/38760) likewise treats per-forward-pass
scheduler measurements as missing from the public Prometheus interface.

### Required patch contract

The patch must be based on the pinned upstream commit and do only the following:

1. Maintain a request-local high-water mark for token positions that have actually completed a
   scheduler step. For each later scheduled interval, count the intersection below that high-water
   mark as recomputed executions. Re-executing the same position twice counts twice. Merely restoring
   a prefix-cache block does not count because it was not scheduled for execution.
2. Increment monotonic counters for vLLM-native context and generation tokens on every completed
   scheduler iteration. In this protocol, `scheduled_prefill_tokens_total` means vLLM's
   `context tokens`; it can include replay while a preempted request is in context phase and must not
   be described as “new prompt tokens only.”
3. Observe `context requests + generation requests` once per completed scheduler iteration in an
   effective-batch-size histogram.
4. Export exactly the four `inferpilot:*` families already listed in Gate 1, without changing policy,
   allocation, preemption, batching, or model execution.
5. Keep `--enable-logging-iteration-details` enabled during the canary. Across the same complete
   server lifetime, the two scheduled-token counter deltas and the batch histogram count/sum must
   reconcile exactly with the stock log rows. The recomputation counter is validated by the controlled
   canaries in Gate 2, because stock vLLM has no equivalent field.

The fork commit, patch diff digest, container-image digest, upstream commit, and launch arguments must
be frozen in the preregistration before the first GPU canary. Until those values replace the explicit
placeholder in the preregistration, the study status remains **INSTRUMENTATION_NO_GO**. A log parser,
`vllm:prompt_tokens`, preemption count, or subtraction from nominal prompt tokens is not an acceptable
substitute for the patched recomputation counter.

## Gate 1 — pinned real-server metric surface

Start only the preregistered Qwen2.5-3B server on one A10G using the frozen patch revision and
`--enable-logging-iteration-details`. Before sending the measured workload, run:

```bash
uv run python -m inferpilot.runner.metrics_capabilities http://127.0.0.1:8000 \
  > metrics-capabilities.json
```

The command must exit 0, `ready` must be `true`, and `missing_required` must be empty. The following
semantics must each have one accepted metric name:

| Semantic | Accepted metric family |
|---|---|
| waiting requests | `vllm:num_requests_waiting` |
| running requests | `vllm:num_requests_running` |
| KV occupancy | `vllm:kv_cache_usage_perc` |
| preemption events | `vllm:num_preemptions` or `vllm:num_preemptions_total` |
| exact recomputed token executions | `inferpilot:recomputed_token_executions_total` or `vllm:recomputed_token_executions_total` |
| scheduled prefill tokens | `inferpilot:scheduled_prefill_tokens_total` or `vllm:scheduled_prefill_tokens_total` |
| scheduled decode tokens | `inferpilot:scheduled_decode_tokens_total` or `vllm:scheduled_decode_tokens_total` |
| effective batch size | `inferpilot:effective_batch_size` or `vllm:effective_batch_size` |

### Immediate kill criteria

Stop the server and record `INSTRUMENTATION_NO_GO`—without running a performance cell—if:

- the command exits nonzero or any required semantic is missing;
- a required family is present under an unregistered name (amend a future protocol; do not alias it
  during this study);
- the server/version/model revision differs from the preregistration;
- metric collection requires disabling normal scheduler behavior;
- enabling metrics changes request handling in a way that cannot be applied equally to both arms.

Name presence does not establish meaning. Passing Gate 1 only authorizes Gate 2.
The programmatic runner gate is `require_metric_capabilities=True` with
`FP8_MECHANISM_REQUIREMENTS`; when a family is absent it emits a structured
`MetricCapabilityMismatch` before sending any measured request.

## Gate 2 — semantic counter canary

Use a fresh server and a short, discarded instrumentation workload. Preserve its bundle, but never
include it in pilot outcomes.

Required checks:

1. Counter families are monotonic during one server lifetime and reset only on a fresh server.
2. Waiting/running gauges are nonnegative and never contradict the complete request census at aligned
   sample boundaries.
3. `scheduled_prefill_tokens_total` and `scheduled_decode_tokens_total` increase under their respective
   phases; neither decreases or produces non-integer deltas. Their complete-lifetime deltas exactly
   equal the sums of stock iteration-log context and generation tokens.
4. `effective_batch_size` is a per-iteration histogram. Its observation count equals the number of
   complete stock iteration-log rows, its sum equals the logged sum of context plus generation
   requests, and every logged observation is within `[0, max_num_seqs]`.
5. A deliberately preempting canary makes both the preemption-event counter and
   `recomputed_token_executions_total` increase.
6. A low-pressure control produces zero recomputed-token delta. If it does not, explain and validate
   the counter's baseline work before proceeding.
7. Recomputed tokens are a subset of actual scheduled compute work; they must not exceed total
   scheduled prefill plus decode token executions.
8. Sampling covers the same monotonic window as `LoadEvidence`, with both boundary samples present.

### Semantic kill criteria

Record `INSTRUMENTATION_INVALID` and stop if a counter resets, decreases, is ambiguous about cached vs
recomputed work, cannot be aligned to the measurement window, or fails the positive/negative canary.
Also stop on any patch/log reconciliation mismatch. Do not replace exact recomputation with
preemption event count, log-derived subtraction, or an undocumented prompt-token proxy.

## Gate 3 — each of the six registered cells

Before accepting a cell, require all of the following:

- fresh server and preregistered arm order;
- exact model and vLLM revisions recorded;
- `COMPLETED`, effective configuration verified, and 100/100 successful measured requests;
- all warm-ups successful and excluded;
- `LoadEvidence` present with `coverage_complete=true`, `steady_state=true`, ≥100 arrivals, and ≥30 s;
- metric capability report present and `ready=true`;
- complete queue/KV/preemption/recompute/prefill/decode/batch coverage;
- `arrivals.json`, telemetry, phase timing, lifecycle, and bundle-integrity checks pass;
- dispatch-drift p95 ≤10 ms;
- no pre-teardown lifecycle errors;
- cumulative spend remains below the preregistered hard stop before starting the next cell.

### Retry and stopping rule

- Retry once only when the sole rejection is dispatch-drift p95 >10 ms or an external provider launch
  failure before measurement. Preserve both attempts.
- No seed substitution, threshold change, workload change, or metric alias is allowed.
- Any other rejection stops the pilot before outcome comparison.
- If accrued cost plus one worst-case cell could exceed $12, do not start that cell.
- Never exceed the user's $30 absolute limit; the protocol's lower $12 stop governs this pilot.

## Final interpretation gate

The six-run pilot may authorize the larger study only under the preregistered decision rule. It cannot
establish cross-model generality, an SLO capacity curve, quality safety, or production readiness. A
mixed/negative result is publishable evidence, not permission to tune thresholds after the fact.
