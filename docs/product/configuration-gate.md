# Configuration acceptance gate v0.1

`inferpilot gate` combines an operator-reviewed policy, two measured run bundles, and quality
evidence into one deterministic `PASS`, `FAIL`, or `ABSTAIN` report.

First bind each measured quality-gate artifact to the exact run pair and corpus:

```bash
inferpilot bind-quality teacher-forced-kl evidence/kl-gate.json \
  runs/baseline runs/candidate \
  --corpus-id operator-eval-v1 \
  --corpus-sha256 <sha256> \
  --output evidence/teacher-forced-kl.json
```

Use `needle-retrieval` for a `NeedleQualityGate`. The binder computes experiment IDs and
configuration fingerprints from the run bundles; operators do not hand-author those identities.
It does not receive or persist prompt text.

```bash
inferpilot gate examples/configuration_gate_spec.json \
  runs/baseline runs/candidate \
  --quality-evidence evidence/teacher-forced-kl.json \
  --quality-evidence evidence/needle-retrieval.json \
  --output configuration-gate-report.json
```

Exit status is `0` for `PASS`, `1` for `FAIL`, and `2` for `ABSTAIN` or invalid input. The JSON
report preserves the distinction.

## What `PASS` means

All of the following were true for this observed pair:

- both runs were baseline-eligible and differed only in every declared engine field;
- their intended replay and aligned load windows matched;
- the candidate was healthy and improved at least one of TTFT p95 or TPOT p95 without regressing
  the other beyond the declared tolerance;
- every candidate SLO constraint passed; and
- every preregistered quality gate was present, bound to the exact experiment IDs, configuration
  fingerprints, and corpus digest, and passed.

Precision-changing candidates (`dtype` or `kv_cache_dtype`) cannot be registered without both
teacher-forced KL and long-context needle-retrieval gates. Greedy token agreement remains a drift
prefilter and cannot certify quality.

## Failure semantics

- `FAIL` means observed evidence rejected the candidate: overload, an SLO violation, a Pareto
  regression/no improvement, or a sufficiently sampled quality regression.
- `ABSTAIN` means evidence was missing or inconclusive: an ineligible run, incomplete aligned-load
  evidence, or absent/undersampled quality evidence.
- Structurally incomparable or misbound inputs are rejected rather than converted into a verdict.

The report is self-validating on reload. Altering its verdict, checks, comparison, or reasons without
changing the bound evidence causes validation to fail.

## Safety boundary

`PASS` is an observed-pair acceptance result, not statistical significance and not deployment
authorization. Every report fixes `deployment_authorized=false`; operator review and a controlled
canary remain required. InferPilot still does not modify a deployment or collect task-quality data
automatically.
