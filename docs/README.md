# InferPilot documentation map

The repository keeps its complete research audit trail. You do not need to read it in chronological
order. Choose the path that matches your goal.

## I want to understand the product

1. Run `uv run inferpilot demo` from the repository root.
2. Read [the guided assessment](product/guided-assessment.md).
3. Read [the configuration acceptance gate](product/configuration-gate.md).
4. Read [the offline optimizer boundary](product/offline-optimizer.md).
5. Inspect [`evidence_card.py`](../src/inferpilot/evidence_card.py) and its
   [tests](../tests/test_evidence_card.py).

## I want to evaluate the scientific claim

1. Start with [FP8 KV improved throughput by 33.6%. Our hypothesis still failed.](blog/fp8-throughput-improved-hypothesis-failed.md).
2. Read [When does FP8 KV actually help?](blog/when-does-fp8-kv-actually-help.md).
3. Read the [response to expert critique](CRITIQUE-RESPONSE.md).
4. Inspect the preregistrations and full positive, invalid, and negative record in
   [`experiments/`](experiments/).

## I want to review measurement correctness

1. [`saturation.py`](../src/inferpilot/saturation.py) defines the conserved load assessment.
2. [`load_evidence.py`](../src/inferpilot/runner/load_evidence.py) constructs aligned window evidence.
3. [`orchestrator.py`](../src/inferpilot/runner/orchestrator.py) owns lifecycle, timing, and cleanup.
4. [`comparison/`](../src/inferpilot/comparison/) contains compatibility and decision gates.
5. [`tests/`](../tests/) is the executable contract suite.

## I want architecture and roadmap context

- [Engine-agnostic boundary](design/2026-09-19-engine-agnostic.md)
- [Quality-gate methodology](design/2026-09-19-quality-gate-methodology.md)
- [Measured-canary boundary](architecture/m8-measured-canary.md)
- [Search and adaptation](architecture/milestone-2-search-and-adaptation.md)

## Evidence labels

- **Synthetic demo:** proves software behavior only; never empirical evidence.
- **Exploratory:** useful for hypotheses, not a confirmatory claim.
- **Preregistered:** design frozen before the measured outcome.
- **Invalid:** preserved evidence that failed a registered gate.
- **Held out:** measured only after the corresponding protocol was frozen.

These labels are part of the project, not decoration: InferPilot is designed to preserve honest
negative results and prevent unsupported evidence from silently becoming a recommendation.
