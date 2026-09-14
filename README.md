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
distributed execution, database, web UI, or LLM-driven decision logic.

The runner does **not** wrap or parse `vllm bench` — InferPilot owns its raw request
measurements end to end.

The minimal comparison layer catalogs complete run bundles by content hash, builds
exact-repeat cohorts, and refuses comparisons that differ outside explicitly allowlisted
engine parameters. It reports run-level descriptive statistics and observed changes; it
does not make significance or automatic accept/reject claims.

## Milestone 1 execution flow

```
ExperimentConfig  (model + engine knobs + workload)
      │
      ▼
capture EnvironmentMetadata  (host + software/toolchain snapshot)
      │
      ▼
start vLLM server (shell-free subprocess, /health readiness poll)   # runner.server
      │
      ▼
verify resolved vLLM configuration against requested configuration
      │
      ▼
warm-up requests (excluded)  →  measured workload                   # runner.client
      │  (async streaming /v1/completions, monotonic TTFT/e2e/TPOT)
      ▼
AggregateMetrics + measured-window GPU/KV telemetry
      │
      ▼
ExperimentResult  (config + environment + measurements + aggregates + status/failure)
      │
      ▼
write immutable result/warmup/telemetry/lifecycle artifacts + logs
```

Every result is losslessly JSON-serializable so it can be stored, reloaded, and compared as
ground truth. Failure outcomes are first-class: `FAILED`, `OOM`, and `TIMEOUT` produce a
structured result with a populated `FailureRecord`. Baseline eligibility additionally
requires every request to succeed, a fully verified effective configuration, all comparison
metrics, and complete GPU/KV telemetry. Partial results may be stored but must not be compared.

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
  effective.py     # resolved vLLM configuration contract
  telemetry.py     # raw resource sample + telemetry summary contracts
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
    effective_config.py # startup-log parsing + requested/resolved fidelity checks
    telemetry.py     # measured-window NVML + vLLM KV-cache sampling
    orchestrator.py  # glue: COMPLETED / FAILED / OOM / TIMEOUT
    __main__.py      # CLI: python -m inferpilot.runner <config.json>
  comparison/
    store.py         # content-addressed, integrity-checked run-bundle catalog
    fingerprint.py   # exact-repeat and controlled-comparison identities
    compare.py       # cohort statistics + direction-aware observed deltas
    __main__.py      # ingest, summary, and compare CLI
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
  test_effective_config.py # effective-config parsing + fidelity checks
  test_telemetry.py        # resource sampling + graceful degradation
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

## Cataloging and comparing repeated runs

Only baseline-eligible results can be ingested. The catalog copies the complete run bundle
and verifies every file against an integrity manifest when loading it.

```bash
# Ingest three equivalent repetitions.
python -m inferpilot.comparison ingest result-store \
  runs/<run-1> runs/<run-2> runs/<run-3>

# Summarize one exact-repeat cohort (minimum three runs by default).
python -m inferpilot.comparison summary result-store <experiment-id>

# Compare two cohorts that differ only in one declared engine field.
python -m inferpilot.comparison compare result-store \
  --baseline <baseline-experiment-id> \
  --candidate <candidate-experiment-id> \
  --vary max_num_seqs \
  --output runs/comparison.json
```

Human labels, hostnames, and capture timestamps do not define compatibility. Model/revision,
workload, hardware, software/toolchain, runtime overrides, and requested/resolved engine
settings do. A comparison is rejected if any non-allowlisted condition changes. Resource
telemetry is reported as context; KV-cache utilization is not assumed to be intrinsically
better when lower or higher.

### First controlled scheduling experiment

`examples/experiment_c4_seq1.json` and `examples/experiment_c4_seq4.json` hold workload,
model, runtime, and every engine setting constant except `max_num_seqs`. Four clients issue
requests concurrently. The baseline restricts vLLM to one active sequence; the candidate
allows four, directly exercising continuous batching. Runs should be interleaved and repeated
at least three times per cohort before comparison with `--vary max_num_seqs`.

Pinned runtime decisions (see `inferpilot/runner/defaults.py`):

| Decision | Value |
|---|---|
| Benchmark Python | 3.12 |
| vLLM | `0.29.0` (exact) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| HF revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Workload | 4 warm-up + 32 measured, ~128 in / 32 out tokens, closed-loop c=1 |
| Generation | greedy (`temperature=0`), `ignore_eos` + `min_tokens=32` (controlled length) |
| Sampler | `sampler_backend=pytorch` (→ `VLLM_USE_FLASHINFER_SAMPLER=0`) |
| Generation defaults | `generation_config=vllm` (ignore model-provided sampling defaults) |

> **Sampler backend & the FlashInfer/nvcc issue.** On this box, FlashInfer 0.6.18's sampler
> JIT invokes the local CUDA **12.4** `nvcc` with `--compress-mode=size` (an nvcc ≥12.6/13
> flag), which fails and crashes vLLM startup. The baseline therefore pins
> `sampler_backend: "pytorch"` (native sampler, no FlashInfer JIT). A result produced this way
> **must not** be compared against one that used the FlashInfer sampler — the sampler is
> recorded in `environment.effective_sampler_backend` / `runtime_overrides`. Set
> `sampler_backend: "flashinfer"` (or fix the toolkit) to measure the FlashInfer path.

### Schema version

The result schema is **`0.3.0`**. Version `0.2.0` added split CUDA provenance and the sampler
backend; `0.3.0` adds resolved-configuration evidence and measured-window resource telemetry.
There is **no in-place migration**: older artifacts fail loudly rather than being
reinterpreted. Re-run to produce a `0.3.0` artifact.

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
- **`COMPLETED` ≠ valid benchmark.** `COMPLETED` means *orchestration* finished
  (server started, measured workload ran to the end) — even if every request
  failed. Benchmark validity is the separate, stricter
  `ExperimentResult.is_baseline_eligible`, which requires: status `COMPLETED`,
  aggregates present, `num_requests == config.workload.num_requests`, all
  requests succeeded (`num_failed == 0`), and every comparison metric
  (ttft/tpot/e2e p50·p95·p99 + both throughputs) populated, plus a verified effective
  configuration and complete measured-window GPU/KV telemetry. `FAILED`/`OOM`/
  `TIMEOUT` and partial/all-failed runs are ineligible but may still store partial
  aggregates.
- **Warm-up failure aborts before measuring.** If any warm-up request fails, the
  runner stops and emits a `FAILED` result with error type `WarmupFailure`; warm-up
  measurements are preserved in a separate `warmup.json` artifact and never mixed
  into measured aggregates.
- **Hardware snapshot + measured telemetry.** GPU/CPU/RAM + driver/CUDA and
  Python/PyTorch/vLLM versions are captured once at run start. During the measured
  window only, a background sampler records GPU memory/utilization through NVML and
  `vllm:kv_cache_usage_perc` every 250 ms. Raw samples are retained in `telemetry.json`.
- **Zero-duration aggregates.** A completed/aggregated run over ≥ 1 request must
  have `duration_s > 0`; zero duration is valid only for an empty
  (`num_requests == 0`) aggregate.
- **`SKIPPED` removed.** The status enum is now `PENDING / RUNNING / COMPLETED /
  FAILED / OOM / TIMEOUT` only.
- **OOM/TIMEOUT classification is heuristic.** OOM is detected by scanning server
  stderr for CUDA OOM signatures; a never-ready server is `TIMEOUT` (or `OOM` if
  the log shows an OOM). Revisit if signatures prove unreliable across vLLM versions.
- **Schema versioning is exact-match.** `SUPPORTED_SCHEMA_VERSIONS` is currently
  `{"0.3.0"}`; loading any other version fails loudly. There is no migration /
  upgrade path yet — one will be needed before the contract changes.

## Hardware note

Primary local machine is an **RTX 3050 Laptop, 4 GB VRAM** — too small for meaningful serving,
so the architecture anticipates dispatching experiments to other machines (Kaggle/Colab,
RunPod/Vast, larger GPUs). `EnvironmentMetadata` exists so results from different hardware are
never silently compared.

## Engineering principles (from the vision)

Measure before optimizing · every optimization needs benchmark evidence · reproducible
experiments · one meaningful variable at a time · **real benchmark results are ground truth**.
