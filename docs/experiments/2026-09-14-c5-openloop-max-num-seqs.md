# C5 — Open-loop scheduling study: `max_num_seqs` under `poisson-v1` arrivals

**Date:** 2026-09-14 · **Status:** characterization (no SLO, no winner declared)

## Setup (pinned)

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` |
| Engine | vLLM 0.29.0, PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), `dtype=bfloat16`, `max_model_len=2048`, `max_num_batched_tokens=2048`, `gpu_memory_utilization=0.85`, `kv_cache_dtype=auto`, prefix caching **off**, `generation_config=vllm` |
| Hardware | NVIDIA RTX 3050 Laptop, 4096 MiB, driver 595.84 (driver-CUDA 13.2 / torch-CUDA 13.0 / nvcc 12.4.131) |
| Workload | `poisson-v1`, **256 measured** + **4 sequential warm-ups**, ~128 input / **exactly 32** output tokens, `request_rate_qps=8.0`, `seed=0`, `max_concurrency=null`, SLO **null** |
| Variable | **only** `engine.max_num_seqs ∈ {1,2,3,4}** (verified programmatically; only `experiment_id`/`name`/`max_num_seqs` differ) |
| Configs | `experiments/c5-openloop/seq{1..4}.json` (committed `54757f1`) |

## Arrival rate: nominal vs. realized

- **Nominal** rate: 8.0 QPS.
- **Realized** rate of the *finite* seed-0 schedule: **7.5218 QPS** (255 inter-arrivals over a
  33.901 s scheduled span). This is a property of the specific 256-draw Poisson sample, not a
  measurement artifact, and is identical across all four settings (same seed ⇒ same schedule).
- **Dispatch fidelity:** actual dispatch offsets tracked the schedule with p95 drift ≤ 1.36 ms
  and a single worst-case of **105 ms** across all 12 runs (realized *dispatch* rate 7.5217 QPS).
  Arrivals did not wait on completions. Drift is **not material**.

## Method

Three interleaved repetitions (to spread any drift/thermal ordering effects):
`[seq1,seq2,seq3,seq4]`, `[seq4,seq3,seq2,seq1]`, `[seq2,seq4,seq1,seq3]` → 12 runs total.
Every run was validated before acceptance: status COMPLETED, baseline-eligible, 256/256
successful, effective config verified, telemetry complete, no pre-teardown lifecycle errors,
`arrivals.json` present. **12/12 passed.** Statistics below are computed **per run** (each run's
own percentiles), then summarized across the 3 runs of a setting — raw request percentiles are
**not** pooled across runs.

## Per-setting run-level summary (mean | stddev | CV% | min | max, n=3)

**TTFT p50 (ms)** — queueing delay
| seqs | mean | stddev | CV% | min | max |
|---|---|---|---|---|---|
| 1 | 8711.80 | 162.03 | 1.86 | 8588.37 | 8895.29 |
| 2 | 463.63 | 12.81 | 2.76 | 448.89 | 472.00 |
| 3 | 35.57 | 0.64 | 1.81 | 34.95 | 36.24 |
| 4 | 33.06 | 0.65 | 1.96 | 32.63 | 33.80 |

**TTFT p95 (ms)**
| seqs | mean | stddev | CV% | min | max |
|---|---|---|---|---|---|
| 1 | 19340.68 | 220.56 | 1.14 | 19191.12 | 19593.99 |
| 2 | 1851.94 | 95.51 | 5.16 | 1746.79 | 1933.31 |
| 3 | 226.51 | 5.05 | 2.23 | 221.18 | 231.22 |
| 4 | 92.93 | 0.68 | 0.74 | 92.50 | 93.71 |

**TPOT p50 (ms)** / **TPOT p95 (ms)**
| seqs | p50 mean | p50 CV% | p95 mean | p95 CV% |
|---|---|---|---|---|
| 1 | 6.28 | 0.07 | 6.45 | 2.36 |
| 2 | 7.25 | 0.49 | 7.40 | 1.77 |
| 3 | 7.55 | 0.52 | 7.84 | 1.65 |
| 4 | 7.49 | 1.14 | 8.00 | 0.45 |

**Throughput (output tok/s)** · **Peak GPU mem (MiB)** · **Peak KV-cache usage (frac)**
| seqs | thr mean | thr CV% | peak_gpu_mem | peak_kv |
|---|---|---|---|---|
| 1 | 150.95 | 0.43 | 3584 | ~0.001 |
| 2 | 235.42 | 0.47 | 3584 | ~0.002 |
| 3 | 240.09 | 0.01 | 3606 | ~0.003 |
| 4 | 240.10 | 0.00 | 3606 | ~0.004 |

## Four-setting Pareto frontier

Objectives (explicit, non-contextual): `ttft_p95_ms` ↓, `tpot_p95_ms` ↓, `throughput_tokens_per_s` ↑
(computed over cohort means; context fingerprint `c470eb30…`, frontier report v0.1.1).

**All four settings are non-dominated.** Each is a distinct trade-off point:

| seqs | ttft_p95 (ms) | tpot_p95 (ms) | throughput (tok/s) | note |
|---|---|---|---|---|
| 1 | 19340.7 | **6.45** | 150.95 | best per-token latency, worst queueing, lowest throughput |
| 2 | 1851.9 | 7.40 | 235.42 | |
| 3 | 226.5 | 7.84 | 240.09 | |
| 4 | **92.9** | 8.00 | **240.10** | best TTFT & throughput, worst per-token latency |

## Observations (descriptive only)

- At `max_num_seqs=1` the single-sequence service rate is below the ~7.52 QPS arrival rate, so a
  queue builds across the window: TTFT p50 ≈ 8.7 s, p95 ≈ 19.3 s, throughput saturates ≈ 151 tok/s.
- Raising `max_num_seqs` lets batching absorb the offered load: TTFT collapses (p95 19.3 s → 93 ms
  from seq1→seq4) and throughput rises to ≈ 240 tok/s (seq3 ≈ seq4).
- The classic counter-movement appears: TPOT rises modestly with batch width (p95 6.45 → 8.00 ms).
- This is a **trade-off surface, not a ranking** — TTFT/queueing and TPOT move in opposite
  directions, and no objective weighting or SLO is applied.

## Stability / variance

Run-to-run CVs are within the reproducibility thresholds (TPOT & throughput CV < 5%, TTFT CV
< 10%) for **every** setting. The largest latency CV is seq2 TTFT p95 at 5.16% (below the 10%
TTFT flag; noted as the most variable metric — consistent with seq2 sitting on the steep part of
the load/queueing curve). No metric was flagged for investigation.

## Limitations

- Realized 7.52 QPS ≠ nominal 8 QPS (finite-sample effect); a different seed yields a different
  realized rate. Do not treat 8 QPS as the achieved offered load.
- `poisson-v1` is a bounded MVP: the whole schedule is materialized and all arrivals may become
  in-flight; it does not throttle to a sustainable rate. Appropriate here (256 bounded requests).
- Peak KV-cache usage is tiny (≤0.4%) — the 0.5 B model barely fills the cache; KV pressure is not
  exercised by this workload.
- PyTorch-sampler runs; **must not** be compared against FlashInfer-sampler results.

## Reproducibility / artifacts

- Configs: `experiments/c5-openloop/seq{1..4}.json`. Driver: `experiments/c5-openloop/run_study.py`.
- Raw run bundles, ingest store, `manifest.jsonl`, and `analysis.json` live under
  `runs/c5-openloop/` (gitignored — generated artifacts are not committed).
- No SLO was defined and **no winning configuration is declared**; this report characterizes the
  observed scheduling trade-offs only.
