# InferPilot

Autonomous LLM inference-optimization system. Given **Model + Hardware + Workload + SLO**,
InferPilot should determine, test, and adapt the best way to run that model — treating
inference tuning as the system's problem rather than the engineer's.

Full vision: [`InferPilot — Shared Project Context.md`](./InferPilot%20%E2%80%94%20Shared%20Project%20Context.md).

## Status: Milestone 1 — data contracts only

This repository currently contains **only the validated, serializable data contracts** that
flow through the optimization loop. There is deliberately **no** engine/server management,
GPU profiling, search, database, web UI, or LLM-driven decision logic yet. Those come later,
on top of trustworthy data objects.

## Milestone 1 execution flow

The target loop (only the schemas exist today; the runner is future work):

```
ExperimentConfig  (model + engine knobs + workload + optional SLO)
      │
      ▼
capture EnvironmentMetadata  (host + hardware + software stack)
      │
      ▼
run workload → collect RequestMeasurement per request      # future runner
      │
      ▼
derive AggregateMetrics  (TTFT/TPOT/e2e percentiles, throughput)   # future
      │
      ▼
ExperimentResult  (config + environment + measurements + aggregates + status/failure)
      │
      ▼
serialize to JSON, store, compare against baseline         # future
```

Every result is losslessly JSON-serializable so it can be stored, reloaded, and compared as
ground truth. Failure outcomes (including **OOM** on the 4 GB laptop GPU) are first-class:
a result with no aggregates but a populated `FailureRecord` is valid and round-trips.

## Package layout

```
src/inferpilot/
  _base.py         # SchemaModel base (strict, lossless JSON) + SCHEMA_VERSION
  config.py        # ExperimentConfig, EngineConfig (MVP knobs), SLO
  environment.py   # EnvironmentMetadata, HardwareInfo
  workload.py      # WorkloadSpec
  measurements.py  # RequestMeasurement (per-request ground truth)
  results.py       # AggregateMetrics, ExperimentResult
  status.py        # ExperimentStatus, FailureRecord
examples/
  example_experiment.json
tests/
  test_example_config.py   # example parses into ExperimentConfig
  test_roundtrip.py        # JSON round-trip incl. OOM/failed result
  test_validation.py       # negative tests for every invalid contract state
```

## Quickstart

Dependencies are locked with [uv](https://docs.astral.sh/uv/) (`uv.lock`, hashed,
reproducible). Exact setup and test commands:

```bash
# 1. Install uv (once): https://docs.astral.sh/uv/getting-started/installation/
# 2. Create/refresh the locked environment (base + dev tools):
uv sync --extra dev

# 3. Run the full test suite:
uv run --extra dev pytest

# Build the package (sdist + wheel):
uv build

# Regenerate the lock after changing dependencies in pyproject.toml:
uv lock
```

`uv.lock` pins exact versions + hashes for every transitive dependency; `uv sync`
reproduces that environment exactly. The Python floor is pinned in
`pyproject.toml` (`requires-python = ">=3.10,<3.15"`).

<details>
<summary>Plain pip/venv (no uv)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
</details>

Load the example config:

```python
import json
from inferpilot import ExperimentConfig

cfg = ExperimentConfig.model_validate(json.load(open("examples/example_experiment.json")))
print(cfg.engine.model)  # operator-chosen model id
```

## Unresolved decisions (intentionally not fixed yet)

These are open and must be decided before/with the runner. The schemas keep them
configurable so we are **not guessing**:

- **Model** — the example uses `facebook/opt-125m` purely as a placeholder small model.
  The real target model (and whether it fits the 4 GB RTX 3050 vs. a rented GPU) is undecided.
  `EngineConfig.model` is a free-form, operator-chosen identifier.
- **vLLM version** — which serving-engine version to pin is undecided. Recorded as
  `EnvironmentMetadata.vllm_version` but not constrained.
- **Workload shape** — `WorkloadSpec` currently models a fixed prompt/output length with a
  simple open- or closed-loop arrival pattern. Realistic distributions (variable lengths,
  chat vs. batch, traces) are undecided.
- **SLO** — concrete latency/throughput targets are undecided. `SLO` fields are all optional;
  experiments may run with no SLO for pure characterization.

## Contract decisions for review

The hardening pass encoded several semantic rules directly into the schemas. These
are defensible defaults but warrant an architectural sign-off before the runner
depends on them:

- **Successful request must produce ≥ 1 token.** A `RequestMeasurement` with
  `success=True` requires `output_tokens >= 1`, `ttft_ms`, and `e2e_latency_ms`.
  A genuinely empty-but-successful completion (0 tokens) would be rejected — flag
  if that case is real.
- **TPOT is defined only for ≥ 2 output tokens.** One-token outputs have no
  inter-token interval, so `tpot_ms` must be `None` there and present for
  successful multi-token requests. Confirm this matches how the runner will
  compute TPOT (mean of inter-token gaps, excluding TTFT).
- **Failed requests keep best-effort timings.** Failures must carry an `error`
  but may retain partial `ttft_ms`/`tpot_ms`/`e2e_latency_ms` (not forced to
  `None`), so partial-progress data survives.
- **Zero-duration aggregates.** A completed/aggregated run over ≥ 1 request must
  have `duration_s > 0`; zero duration is valid only for an empty
  (`num_requests == 0`) aggregate. Revisit if we ever want to store a
  placeholder aggregate for a not-yet-run experiment.
- **`FAILED`/`OOM`/`TIMEOUT` may still carry aggregates.** These require a
  `failure` but do not forbid partial aggregates. `COMPLETED` requires aggregates
  and forbids a `failure`.
- **`SKIPPED` is under-specified.** It is treated as terminal but is only
  constrained by "if a `failure` is present, its status must match". Whether a
  skipped experiment should be allowed aggregates, or should record a skip
  reason, is undecided.
- **Schema versioning is exact-match.** `SUPPORTED_SCHEMA_VERSIONS` is currently
  `{"0.1.0"}`; loading any other version fails loudly. There is no migration /
  upgrade path yet — one will be needed before the contract changes.

## Hardware note

Primary local machine is an **RTX 3050 Laptop, 4 GB VRAM** — too small for meaningful serving,
so the architecture anticipates dispatching experiments to other machines (Kaggle/Colab,
RunPod/Vast, larger GPUs). `EnvironmentMetadata` exists so results from different hardware are
never silently compared.

## Engineering principles (from the vision)

Measure before optimizing · every optimization needs benchmark evidence · reproducible
experiments · one meaningful variable at a time · **real benchmark results are ground truth**.
