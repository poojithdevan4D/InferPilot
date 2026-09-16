# Evidence-bound advisor

InferPilot's first advisor is intentionally fail-closed. It recommends an engine override only when
model revision, GPU name, prompt/output lengths, arrival algorithm, and measured rate exactly match a
validated policy regime. It abstains on every unsupported input; it does not interpolate or claim
generality beyond the evidence.

```bash
python -m inferpilot.advisor policies/m3-rtx3050-qwen05b.json \
  runs/m3-heldout/policy-report.json examples/advisor_request.json
```

The included policy returns width 2 at 2 QPS and width 4 at 6 QPS, with the fixed 512-token batching
budget. The CLI validates the self-checking report and its SHA-256 digest before recommending. This is a
research demonstrator, not an automatic production deployment controller.

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
- **Only exact evidence-supported contexts receive a recommendation.** The observed realized rate must
  exactly match a validated regime; the adapter never rounds, buckets, interpolates, or maps a measured
  rate to 2 or 6 QPS.
- **Nearby or mixed workloads abstain.** A rate that is not exactly a regime abstains
  (`unsupported_request_rate_qps`); mixed prompt/output lengths abstain
  (`mixed_prompt_lengths` / `mixed_output_lengths`); a workload not declared fixed-length abstains.
  Token summaries are never silently treated as fixed request lengths.
- **Provenance-bound and tamper-evident.** The decision binds to the profile's SHA-256 provenance
  digest; the profile recomputes every derived field on load, and both the profile and the decision
  reject tampering.

### Validity scope

The current policy is validated **only** for the pinned `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, the **RTX 3050 Laptop GPU**, **fixed 128/32-token
`poisson-v1`** workloads, and the **exact 2 QPS and 6 QPS** regimes. Anything else abstains. This is
**not** a production autoscaler and provides **no deployment-safety guarantee**.
