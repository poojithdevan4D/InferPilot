# InferPilot

Autonomous LLM inference-optimization system. Given **Model + Hardware + Workload + SLO**,
InferPilot should determine, test, and adapt the best way to run that model — treating
inference tuning as the system's problem rather than the engineer's.

Full vision: [`InferPilot — Shared Project Context.md`](./InferPilot%20%E2%80%94%20Shared%20Project%20Context.md).

## Status: Milestone 1 — contracts + first runner slice

This repository contains the validated, serializable data contracts **and** the first
vertical slice of the benchmark runner (`inferpilot.runner`): start a vLLM server, run a
fixed characterization workload, measure raw per-request timings, aggregate, and store one
immutable result. There is deliberately still **no** optimization/search, SLO evaluation,
GPU profiling, distributed execution, database, web UI, or LLM-driven decision logic.

The runner does **not** wrap or parse `vllm bench` — InferPilot owns its raw request
measurements end to end.

## Milestone 1 execution flow

```
ExperimentConfig  (model + engine knobs + workload)
      │
      ▼
capture EnvironmentMetadata  (host + software stack; no GPU polling yet)
      │
      ▼
start vLLM server (shell-free subprocess, /health readiness poll)   # runner.server
      │
      ▼
warm-up requests (excluded)  →  measured workload                   # runner.client
      │  (async streaming /v1/completions, monotonic TTFT/e2e/TPOT)
      ▼
AggregateMetrics  (TTFT/TPOT/e2e percentiles, throughput)           # runner.aggregate
      │
      ▼
ExperimentResult  (config + environment + measurements + aggregates + status/failure)
      │
      ▼
write immutable result.json + server logs into a unique run dir     # runner.artifacts
```

Every result is losslessly JSON-serializable so it can be stored, reloaded, and compared as
ground truth. Failure outcomes are first-class: `FAILED`, `OOM`, and `TIMEOUT` produce a
structured result with a populated `FailureRecord`. Only a `COMPLETED` result is eligible as
a baseline / optimization signal (`ExperimentResult.is_baseline_eligible`); partial
aggregates from a failed run may be stored but must not be compared.

**Timing:** all durations and latencies use a **monotonic** clock. Wall-clock timestamps are
recorded only for provenance (`started_at`/`finished_at`) and never used for any duration.

**TPOT** is defined precisely as:

```
tpot_ms = (e2e_latency_ms - ttft_ms) / (output_tokens - 1)
```

so it is undefined (absent) for one-token outputs.

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
  runner/
    defaults.py      # pinned runtime decisions (vLLM 0.29.0, model + revision, py3.12)
    server.py        # vLLM subprocess adapter (shell-free, health poll, always-cleanup)
    workload_gen.py  # deterministic prompt construction
    client.py        # async streaming measurement client (/v1/completions)
    aggregate.py     # pure percentile + aggregate computation
    artifacts.py     # unique run dir + immutable result JSON
    orchestrator.py  # glue: COMPLETED / FAILED / OOM / TIMEOUT
    __main__.py      # CLI: python -m inferpilot.runner <config.json>
examples/
  example_experiment.json
tests/
  fake_vllm_server.py      # stdlib fake vLLM server (in-thread + subprocess)
  test_example_config.py   # example parses into ExperimentConfig
  test_roundtrip.py        # JSON round-trip incl. OOM/failed result
  test_validation.py       # negative tests for every invalid contract state
  test_aggregate.py        # unit: percentiles, TPOT formula, aggregation
  test_client.py           # integration: streaming client vs fake server
  test_server_lifecycle.py # subprocess lifecycle + guaranteed cleanup
  test_orchestrator.py     # end-to-end runner vs fake server + artifacts
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

## Running a real benchmark (GPU + vLLM)

The automated tests need **no** GPU, model, or vLLM (they use a stdlib fake server). To run a
real benchmark you need a separate serving environment, because vLLM is deliberately **not**
a project dependency (it is launched as a subprocess):

```bash
# Dedicated Python 3.12 environment with vLLM pinned exactly.
uv venv --python 3.12 .venv-bench
uv pip install --python .venv-bench "vllm==0.29.0"
uv pip install --python .venv-bench -e .   # inferpilot + httpx

# Run the pinned characterization experiment (starts a real vLLM server):
.venv-bench/bin/python -m inferpilot.runner examples/example_experiment.json --output-dir runs
```

Pinned runtime decisions (see `inferpilot/runner/defaults.py`):

| Decision | Value |
|---|---|
| Benchmark Python | 3.12 |
| vLLM | `0.29.0` (exact) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| HF revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Workload | 4 warm-up + 32 measured, ~128 in / 32 out tokens, closed-loop c=1 |
| Generation | greedy (`temperature=0`), `ignore_eos` + `min_tokens=32` (controlled length) |

> **Real-GPU smoke test: intentionally deferred.** vLLM 0.29.0 is not installed on the dev
> box, and the local RTX 3050 has only ~4 GB VRAM (≈3.6 GB free) — below vLLM's practical
> footprint even for a 0.5B model once the CUDA context and KV cache are allocated. The slice
> is validated end-to-end against a fake server; a real run should target a ≥16 GB GPU. No
> claim of a passing GPU run is made.

## Unresolved decisions

Resolved for Milestone 1: **model**, **vLLM version**, and the **characterization workload
shape** (table above). Still open:

- **Workload shape (beyond characterization)** — realistic distributions (variable lengths,
  chat vs. batch, real traces) and open-loop arrivals are not implemented; the current slice
  is closed-loop, fixed-length only.
- **SLO** — concrete latency/throughput targets are undecided. `SLO` fields are all optional;
  experiments currently run with no SLO for pure characterization (no SLO evaluation yet).
- **Prompt tokenization** — prompts are ~N tokens (word-approximate), not exact; the runner
  records the server's *actual* counted tokens, so aggregates use real counts regardless.

## Contract decisions for review

The hardening pass encoded several semantic rules directly into the schemas. These
are defensible defaults but warrant an architectural sign-off before the runner
depends on them:

- **Successful request must produce ≥ 1 token.** A `RequestMeasurement` with
  `success=True` requires `output_tokens >= 1`, `ttft_ms`, and `e2e_latency_ms`.
  A genuinely empty-but-successful completion (0 tokens) would be rejected — flag
  if that case is real.
- **TPOT formula (now implemented).** `tpot_ms = (e2e_latency_ms - ttft_ms) /
  (output_tokens - 1)` — time to generate every token after the first, per
  inter-token step. Defined only for ≥ 2 output tokens (`None` for one-token
  outputs). The client computes it exactly this way.
- **Failed requests keep best-effort timings.** Failures must carry an `error`
  but may retain partial `ttft_ms`/`tpot_ms`/`e2e_latency_ms` (not forced to
  `None`), so partial-progress data survives.
- **Eligibility.** Only `COMPLETED` results are baseline/optimization-eligible
  (`ExperimentResult.is_baseline_eligible`). `FAILED`/`OOM`/`TIMEOUT` may store
  partial aggregates but they must never drive a comparison. `COMPLETED` requires
  aggregates and forbids a `failure`.
- **Zero-duration aggregates.** A completed/aggregated run over ≥ 1 request must
  have `duration_s > 0`; zero duration is valid only for an empty
  (`num_requests == 0`) aggregate.
- **`SKIPPED` removed.** The status enum is now `PENDING / RUNNING / COMPLETED /
  FAILED / OOM / TIMEOUT` only.
- **OOM/TIMEOUT classification is heuristic.** OOM is detected by scanning server
  stderr for CUDA OOM signatures; a never-ready server is `TIMEOUT` (or `OOM` if
  the log shows an OOM). Revisit if signatures prove unreliable across vLLM versions.
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
