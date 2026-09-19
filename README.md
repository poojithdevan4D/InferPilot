# InferPilot

**Evidence-first diagnosis and optimization for LLM inference.** InferPilot is a research-grade
benchmarking and decision library that measures serving behavior, checks provenance and configuration
fidelity, diagnoses only when aligned evidence supports a claim, and otherwise abstains. Persisted
reports are strict, self-validating, and tamper-evident.

## The headline result (all measured on rented cloud GPUs, reproducible)

**An empirical fp8-KV heuristic**—measured but not yet confirmed by the preregistered held-out
protocol—was observed across **4 models × 2 families × 2 GPUs**:

| Legacy observed condition | Measured outcome |
|---|---|
| KV full **+ preempting** (3B/7B/14B) | fp8 KV: **+40–53% throughput** |
| no preemptions, shorter decode workload | fp8 KV: **+1.7%** |

The association follows **preemptions**, not GPU utilization (both cases show GPU near 100%). It is a
post-hoc heuristic, not a causal law. Quality smoke tests found 0/16 factual-QA regressions and 5/5
needle retrieval at 14k, while the teacher-forced-KL preflight **failed** (mean >0.01, p99 0.39),
showing real distributional shift. InferPilot therefore does not call fp8 KV “lossless.”

The strongest measured case was a 3B/A10 run: fp8 KV delivered **+52% goodput, TTFT −56%, and
−34% measured cost per output token**. Those legacy bundles predate aligned load evidence, so the
redesigned advisor now returns `unknown` on them instead of reverse-engineering a confident diagnosis.

## Five-minute review

```bash
uv sync --extra dev --locked
uv run python scripts/aligned_load_demo.py   # healthy / overloaded / abstain, GPU-free
uv run python scripts/fp8_law_demo.py        # real legacy measurements + evidence status
uv run python scripts/cost_rescue_demo.py    # measured economics; no retroactive diagnosis
uv run --extra dev pytest -q                 # 477 tests; no GPU required
```

For a technical review, read these in order:

1. [`docs/blog/when-does-fp8-kv-actually-help.md`](docs/blog/when-does-fp8-kv-actually-help.md) — finding and mechanism hypothesis.
2. [`docs/CRITIQUE-RESPONSE.md`](docs/CRITIQUE-RESPONSE.md) — what expert review invalidated and how the design changed.
3. [`src/inferpilot/saturation.py`](src/inferpilot/saturation.py) — conservative aligned-window load contract.
4. [`src/inferpilot/runner/load_evidence.py`](src/inferpilot/runner/load_evidence.py) — instrumentation for diagnosis-capable future runs.
5. [`docs/experiments/`](docs/experiments/) — preregistrations, positive results, invalid studies, and honest negatives.

## What it does (the reasoning pipeline)

`LoadEvidence` → `LoadAssessment` → `BottleneckDiagnosis` → `plan_optimization` (abstain unless winnable) →
`analyze_fit` / `recommend_scale` (which GPU/TP/precision) → `CapacityAdvisory` (tune/scale/accept +
$/token) → quality gate (`KLQualityGate` + `NeedleQualityGate`) → `compare_configs` (fail-closed
Pareto) → `controller` 0.3.0 (apply/rollback). Ingests real traffic via `WorkloadTrace`.

**Honest limits (read `docs/CRITIQUE-RESPONSE.md`):** the fp8 result is a heuristic from a small,
partly post-hoc matrix (no preregistered held-out validation yet); all wins are in *overloaded* regimes
(a goodput-ceiling lever, not a low-load speedup); quality is smoke-tested only (KL preflight failed);
synthetic traffic; vLLM is the only fully-wired engine (SGLang throughput-only); and this is a rigorous
reasoning **library + evidence, not a running product**. Full measured evidence is in
`docs/experiments/`; methodology is in `docs/design/`. Existing committed GPU bundles predate
`LoadEvidence`: they reproduce measurements but cannot certify the redesigned diagnosis.

---

*Original vision + history below.* Full vision:
[`InferPilot — Shared Project Context.md`](./InferPilot%20%E2%80%94%20Shared%20Project%20Context.md).

## Status: evidence pipeline implemented; held-out diagnostic validation pending

This repository contains the validated, serializable data contracts **and** the first
vertical slice of the benchmark runner (`inferpilot.runner`): start a vLLM server, run a
fixed characterization workload, measure raw per-request timings, aggregate, and store one
immutable result. The repository now also contains evidence comparison, explicit SLO
decisions, and outcome-blind offline replay for conventional search baselines. There is deliberately
still **no production control-plane integration, distributed executor, database, or web UI**.
Controller and recommendation logic are offline, evidence-gated contracts.

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
AggregateMetrics + measured-window GPU/KV telemetry + aligned LoadEvidence
      │
      ▼
ExperimentResult  (config + environment + measurements + aggregates + load evidence + status/failure)
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
  saturation.py    # conserved aligned-window load assessment + legacy TTFT trend
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
    load_evidence.py # bind requests, useful token work, queue, and telemetry to one window
    orchestrator.py  # glue: COMPLETED / FAILED / OOM / TIMEOUT
    __main__.py      # CLI: python -m inferpilot.runner <config.json>
  comparison/
    store.py         # content-addressed, integrity-checked run-bundle catalog
    fingerprint.py   # exact-repeat and controlled-comparison identities
    compare.py       # cohort statistics + direction-aware observed deltas
    frontier.py      # compatibility-guarded multi-objective Pareto analysis
    decision.py      # explicit SLO feasibility + feasible-cohort ranking
    __main__.py      # ingest, summary, compare, frontier, and evaluate CLI
  search/
    policy.py        # outcome-blind declared-order + stable seeded-random baselines
    replay.py        # fixed-budget replay over complete blocked-study evidence
    models.py        # self-validating replay spec/report contracts
    __main__.py      # replay CLI
  features/
    models.py        # arrival artifact + exact-gated online-safe feature report
    arrival.py       # prefix-only rate/inter-arrival/burst feature extraction
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
uv sync --extra dev --locked

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

# Identify nondominated settings for explicitly selected objectives.
python -m inferpilot.comparison frontier result-store \
  --experiment <experiment-a> --experiment <experiment-b> \
  --vary max_num_seqs \
  --objective ttft_p95_ms --objective tpot_p95_ms \
  --objective throughput_tokens_per_s \
  --output runs/frontier.json

# Apply a persisted StudySpec containing candidate ids, SLO, and objective.
python -m inferpilot.comparison evaluate result-store /path/to/study.json \
  --output runs/decision.json
```

Human labels, hostnames, and capture timestamps do not define compatibility. Model/revision,
workload, hardware, software/toolchain, runtime overrides, and requested/resolved engine
settings do. A comparison is rejected if any non-allowlisted condition changes, and every
allowlisted field must have actually changed (a partially-unused allowlist is refused).
When an engine field is allowlisted as varied, its requested value, effective-config value,
and only its mapped environment/runtime *reflections* are excluded from the compatibility
fingerprint — every unrelated runtime override stays significant. For example, allowlisting
`sampler_backend` also excludes `environment.effective_sampler_backend` and
`runtime_overrides.VLLM_USE_FLASHINFER_SAMPLER`, so PyTorch-vs-FlashInfer cohorts compare
correctly (their exact fingerprints still differ). Resource telemetry is reported as context;
KV-cache utilization is not assumed to be intrinsically better when lower or higher. GPU
memory is likewise contextual until a study defines a memory constraint or objective. Derived
report versions: comparison `0.2.1`, frontier `0.1.1`, decision `0.1.1` (result schema stays
`0.3.0`).
The frontier requires at least two explicit non-contextual objectives and computes exact
Pareto dominance over cohort means. It deliberately does not collapse competing metrics
into an implicit score, claim statistical significance, or select a deployment setting
without an SLO.

The decision gate requires a `StudySpec` with at least one explicit SLO constraint. A
cohort is feasible only when the **worst observed run-level value** meets every constraint
(maximum for latency ceilings, minimum for throughput floors). Only feasible cohorts are
ranked, using their declared objective's run-level mean. This conservative observed-sample
policy is deterministic and auditable, but it is not a confidence bound or deployment
guarantee.

### Blocked multi-seed confirmatory studies

`StudySpec`/`evaluate_study` judge candidates under a *single* workload schedule (repetitions
measure repeatability conditional on that schedule). `BlockedStudySpec`/`evaluate_blocked_study`
(CLI: `python -m inferpilot.comparison blocked-evaluate <store> <study.json>`) add
arrival-process variability: each **block** fixes one workload seed and evaluates the same
rectangular set of engine candidates against the SLO; at least three independent blocks are
required. Within a block, candidates must be compatible under the strict comparison fingerprint
(only the declared engine fields differ, workload seed included); across blocks, only the
workload seed and human identifiers may differ (enforced by a separate cross-block fingerprint
that does **not** weaken the within-block fingerprints). A candidate is **robust-feasible** only
if every block's worst observed run meets the SLO; robust-feasible candidates are ranked by the
mean of their per-block objective means, and all per-block results are preserved. Reports are
immutable and make **no** statistical-significance, confidence-bound, or deployment-safety claim.
Report versions: comparison `0.2.1`, frontier `0.1.1`, decision `0.1.1`, blocked study `0.1.1`
(result schema `0.3.0` unchanged).

### Fixed-budget search replay baselines

`inferpilot.search` evaluates conventional candidate-ordering baselines against a complete
blocked-study report without rerunning the GPU. The baseline order is fixed without accepting
outcome data: `declared_order-v1` follows candidate order, while `seeded_random-v1` assigns
each candidate a stable SHA-256 priority using only the policy seed and candidate index.

```bash
python -m inferpilot.search \
  runs/blocked-report.json replay-spec.json \
  --output runs/replay-report.json
```

Each revealed candidate represents one complete blocked evaluation, so its measurement cost is
the number of blocks in the source study. The report distinguishes a budget that ended without
finding feasibility from exhaustive proof that the measured candidate set has no feasible point.
After replay, it scores the observed choice against the complete-study oracle (`oracle_hit` and
simple objective regret). The complete source report is embedded and every derived field is
recomputed on load. Replay report version is exact-gated at `0.1.0`.

This is an **offline evaluation baseline**, not a live optimizer: the unobserved source outcomes
exist in the replay corpus but are unavailable to baseline ordering. It makes no statistical,
generalization, or deployment claim. Adaptive policies, Bayesian/TPE search, experiment-cost
models, and LLM hypothesis generation remain future work.

### Online-safe workload features

`inferpilot.features.extract_arrival_features` converts a validated open-loop `arrivals.json`
artifact into a feature prefix at an explicit observation cutoff. It retains only actual
dispatch offsets at or before the cutoff plus declared prompt/output token targets. It never
accepts or emits request latency, completion, SLO, or future-arrival outcomes.

The exact-gated feature report (`0.1.0`) contains two deliberately distinct rate estimates:
arrival count divided by observation-window duration, and inverse mean observed inter-arrival
time. It also reports population inter-arrival CV and maximum arrivals in closed 250-ms,
500-ms, 1-s, and 2-s windows. The observed prefix is embedded and every derived feature is
recomputed on load. Zero/one-arrival prefixes and simultaneous arrivals have explicit
undefined-field semantics instead of fabricated rates.

### Workload profiling → evidence-bound advisor (M5)

Passive request observations flow through one offline, fail-closed pipeline:

```
observations (JSON/JSONL) → python -m inferpilot.profiling obs.jsonl profile.json
  → WorkloadProfile (self-validating, provenance-digested) → advise_from_profile(policy, profile, context)
  → ProfileAdvisorDecision (recommended override, or a reasoned abstention)
```

Profiling is **passive characterization, not workload classification** (no Poisson/burst labels).
Only an evidence-supported context is recommended: policy v0.2 requires the observed realized rate
to fall inside one of two independently confirmed closed intervals (with no rounding, nearest-regime
mapping, gap filling, or extrapolation), and token summaries are treated as
fixed request lengths only for an explicitly-declared fixed-length workload whose every observation
matches — otherwise the advisor abstains. The decision binds to the profile's provenance digest and is
tamper-evident. The current policy is validated **only** for the pinned Qwen2.5-0.5B revision, the RTX
3050 Laptop GPU, fixed 128/32-token `poisson-v1` workloads, and the 1.5–2.5 / 5.5–6.5 QPS bands; it is **not**
a production autoscaler or a deployment-safety guarantee. See `docs/advisor.md`.

M7 adds a deterministic, replayable controller boundary around these decisions. Repeated agreement
may emit `test_candidate`, but only an explicit canary outcome can emit `apply_candidate` or
`rollback`; the library still performs no live server mutation. See
`docs/architecture/m7-controller-boundary.md`.

Controller v0.2 requires measured canary evidence rather than trusting a pass/fail boolean. It binds
the candidate and execution context, recomputes aggregates from raw requests, and applies the declared
SLO before a transition can report `apply_candidate`. Canary execution itself remains external.

### First controlled scheduling experiment

`examples/experiment_c4_seq1.json` through `experiment_c4_seq4.json` hold workload, model,
runtime, and every engine setting constant except `max_num_seqs`. Four clients issue
requests concurrently while vLLM is allowed one to four active sequences, directly mapping
the continuous-batching trade-off. Runs
should be interleaved and repeated at least three times per cohort before comparison with
`--vary max_num_seqs`.

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

The current schema is **`0.5.0`**; **`0.3.0` and `0.4.0` are still read-supported**
(`SUPPORTED_SCHEMA_VERSIONS = {0.3.0, 0.4.0, 0.5.0}`). Version history: `0.2.0` added split CUDA
provenance and the sampler backend; `0.3.0` added resolved-configuration evidence and
measured-window resource telemetry; **`0.4.0`** adds optional `WorkloadSpec.prompt_seed` and
`arrival_seed`; **`0.5.0`** adds an explicit arrival pattern and bounded batched-Poisson bursts.
Unknown versions (`0.1.0`, `0.2.0`, anything else) fail loudly.

**Prompt/arrival seed separation (0.4.0).** Prompt-generation randomness and arrival-schedule
randomness can now be seeded independently:

- `prompt_seed` seeds prompt-text generation; `arrival_seed` seeds the Poisson schedule.
- Each defaults to `None` and **falls back to the legacy `seed`** when omitted, so a 0.4.0
  workload that sets only `seed` reproduces 0.3.0 behavior exactly.
- Under schema `0.3.0` the split seeds must remain unset (the single `seed` drives both);
  `ExperimentConfig` rejects a 0.3.0 config that sets them.
- **Backward compatibility is byte-exact:** loaded 0.3.0 configs/results keep identical
  exact/comparison/cross-block fingerprints — the identity payload strips the split-seed keys
  when they are `None`. A frozen-hash regression test guards this. When set, either seed is
  compatibility-significant; a blocked study's `cross_block_fingerprint` strips the arrival seed
  (the block variable) but keeps the prompt seed significant (content fixed across blocks).
- `arrivals.json` records an explicit `arrival_seed`; arrival-feature provenance binds to the
  effective arrival seed without reinterpreting older seed-only artifacts.

## Unresolved decisions

Resolved for Milestone 1: **model**, **vLLM version**, and the **characterization workload
shape** (table above). Still open:

- **Workload shape (beyond characterization)** — realistic distributions (variable lengths,
  chat vs. batch and real traces) are not implemented. Arrivals support **closed-loop**
  (`request_rate_qps=None`, bounded by `max_concurrency`) and open-loop `poisson-v1` or
  `batched-poisson-v1` (`request_rate_qps>0`), described below.

### Open-loop arrivals (`poisson-v1` and `batched-poisson-v1`)

When `WorkloadSpec.request_rate_qps > 0`, measured requests are dispatched on a seeded Poisson
schedule instead of the closed-loop semaphore:

- Offsets are seconds relative to the measured **arrival origin** (t0). The first measured
  request arrives at offset `0.0`; each subsequent inter-arrival time is drawn from an
  exponential distribution with mean `1/request_rate_qps`
  (`random.Random(effective_arrival_seed).expovariate(rate)`), and offsets are the running sum.
  The effective arrival seed is `arrival_seed` if set, else the legacy `seed`.
- The schedule is fully determined by `(num_requests, request_rate_qps, effective_arrival_seed)`
  — identical inputs give identical schedules; different seeds give different ones.
- With `arrival_pattern="batched-poisson-v1"`, `burst_size >= 2` requests share each batch
  offset. Batch starts use exponential inter-arrivals with mean
  `burst_size/request_rate_qps`; the final batch is truncated to exactly `num_requests`.
  Thus `request_rate_qps` is the long-run request rate rather than the batch rate, and the
  schedule is determined by `(num_requests, request_rate_qps, effective_arrival_seed,
  burst_size)`. Legacy schemas accept only `poisson-v1`.
- **Arrivals are independent of completion**: every request is scheduled up front and fires at
  its offset regardless of whether earlier requests have finished (no semaphore, no
  back-pressure). Warm-ups run sequentially and finish *before* the arrival clock and telemetry
  window start. Benchmark `duration_s` spans the arrival origin to the final completion.
- Generated and actual dispatch offsets, algorithm, seed, and burst size are stored in an
  immutable `arrivals.json` diagnostic artifact.
- **Limitations:** this is a bounded MVP — the full schedule is materialized up front and all
  requests may become in-flight at once; it does not throttle to a sustainable rate, cap
  in-flight requests, or drop/delay arrivals. Batched Poisson is a synthetic burst model, not
  a production-traffic trace or evidence of queue stability. Use only for small bounded studies.
- **SLO** — concrete latency/throughput *targets* are still undecided and characterization runs
  set `slo: null`. SLO **evaluation** itself exists: a `StudySpec` (single-schedule) or
  `BlockedStudySpec` (multi-seed, robust-feasible) applies an explicit SLO conservatively over
  observed runs. No default/global SLO is invented.
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
