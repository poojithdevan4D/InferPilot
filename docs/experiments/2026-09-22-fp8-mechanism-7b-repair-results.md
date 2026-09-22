# FP8 mechanism 7B repair study — results

**Preregistered verdict: `INCONCLUSIVE`.** All twelve cells completed and passed every acceptance
gate. On 7B, fp8 KV produced a real throughput gain on the pressured workload (+33.6% geomean) and
correctly produced ~none on the unpressured workload (+0.2%) — but it did **not** reduce recomputation,
which is what the registered positive-effect rule required. The clean 3B mechanism ("fp8 drives
recomputed tokens to zero") **does not replicate at 7B on a single A10G**, so the study cannot confirm
the mechanism, and it does not refute the throughput association either.

## Integrity and execution

- Study id: `inferpilot-fp8-mechanism-7b-repair-v1` (repairs the `INVALID_STOP` held-out v1)
- Model: `Qwen/Qwen2.5-7B-Instruct@a09a3545…` · GPU: A10G · vLLM 0.29.0
- Pinned image: `ghcr.io/poojithdevan4d/vllm-inferpilot@sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1`
- 12 cells (3 blocks × {long-context positive, short-context negative} × {bf16, fp8}), all accepted,
  100/128 successful requests per cell, dispatch drift within the 10 ms p95 limit, every window ≥30 s
- Decision report content digest: `dd5a101063f434b3607a6380cdfb8ccbe780baa2e63a5aa2cdaf20c752a11120`
- Estimated incremental A10G cost: **$1.30** (cumulative program spend ≈ $2.11)

## Positive workload (long-context, KV-pressured)

| Block | bf16 tok/s | fp8 tok/s | throughput | bf16 preempt | fp8 preempt | bf16 recompute | fp8 recompute | bf16 burden | fp8 burden |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 184.4 | 249.7 | **1.354×** | 10 | 12 | 42,908 | 51,090 | 0.093 | 0.111 |
| 2 | 191.4 | 304.0 | **1.589×** | 10 | 9 | 42,535 | 37,873 | 0.092 | 0.082 |
| 3 | 236.5 | 262.0 | **1.108×** | 9 | 13 | 38,903 | 54,830 | 0.084 | 0.119 |

Geometric-mean throughput ratio: **1.336×**. Median recompute-burden *reduction*: **−0.19** (i.e. fp8
did not reduce recompute; in two of three blocks it rose). Every fp8 positive cell remained
`overloaded` and kept preempting. TTFT p95 improved substantially under fp8 in every block.

## Negative workload (short-context, unpressured)

Geometric-mean throughput ratio: **1.0017×** (~no effect), 0 preemptions and 0 recomputed tokens in
every bf16 and fp8 cell, load state `near_capacity`/`indeterminate`. The null behaved as a null.

## Interpretation

- **Discrimination holds.** Pressured workload gains (+33.6%), unpressured does not (+0.2%);
  `separation = true`, and the exact counter tracks preemption (`counter_tracks_preemption = true`).
- **The 3B mechanism does not generalize as-is.** At 7B on one A10G the KV cache is pressured enough
  that halving it with fp8 does **not** clear preemption — the server stays overloaded and recompute
  persists (or rises). The throughput win therefore comes from fitting more concurrent KV per step,
  **not** from eliminating recomputation. Because the registered positive effect was defined as a
  recompute reduction, the verdict is `INCONCLUSIVE`, not GO.
- This is a model-size-dependent nuance, not a failure of the tool: InferPilot measured a real gain,
  measured that its hypothesized cause was absent, and refused to call the mechanism confirmed.

## Excluded claims (unchanged from preregistration)

Not a pristine first-look confirmation; not cross-model or cross-engine generality; not production
traffic behavior; not quality safety; not deployment readiness. Held-out support is limited to this
model, GPU, engine, and two synthetic fixed-length workloads.
