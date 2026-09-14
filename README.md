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
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

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

## Hardware note

Primary local machine is an **RTX 3050 Laptop, 4 GB VRAM** — too small for meaningful serving,
so the architecture anticipates dispatching experiments to other machines (Kaggle/Colab,
RunPod/Vast, larger GPUs). `EnvironmentMetadata` exists so results from different hardware are
never silently compared.

## Engineering principles (from the vision)

Measure before optimizing · every optimization needs benchmark evidence · reproducible
experiments · one meaningful variable at a time · **real benchmark results are ground truth**.
