# GPU campaign — preregistration (rented ≥24 GB)

Frozen before any spend. Purpose: move InferPilot from a single toy envelope (Qwen2.5-0.5B,
4 GB RTX 3050) to the **first end-to-end proof that the pipeline generalizes and beats defaults at a
non-trivial scale**, and to produce the **first measured** prompt-length envelope. Same discipline as
M3/M6/M9: fixed grid, seed blocks, SLOs, acceptance/stopping rules, held-out confirmation.

## Fixed choices (the constants)

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, revision pinned at campaign start (record the exact commit
  sha; do not float). Chosen for a clean same-family scale-up from the 0.5B results.
- **GPU:** one rented **L4 24 GB** or **A10G 24 GB**, image with **CUDA/nvcc ≥ 12.6** (unblocks the
  FlashInfer sampler JIT that the 3050 could not build; falls back to PyTorch sampler if needed).
- **Serving:** vLLM, same subprocess runner and contracts as the 3050 work. fp16 weights (~15–16 GB)
  leave ~7–8 GB for KV cache — enough for the batch/rate grid below.
- **Workload:** fixed 128-token nominal prompt / 32-token output (matches the existing generator so
  results are comparable), `poisson-v1` open-loop arrivals, 256-request measured window per cell,
  warmup as in M3. Prompt content from the committed generator.
- **Reused machinery:** the M3 executor's per-cell validation (request counts, effective-config
  verified, telemetry, lifecycle, phases, arrival-drift provenance), manifest, and content-addressed
  store — verbatim, only IDs/output dir change.

## The variable grid

- **max_num_seqs (width):** {1, 2, 4, 8} — wider than the 3050's {1,2,3,4}; the 7B + 24 GB headroom
  makes 8 feasible and is where scale-up behavior should diverge.
- **request_rate_qps:** {2, 6} bands, same as M3/M6 for continuity (min/max bands as in M6).
- **arrival seeds:** development block {60,61,62}; **held-out** block {73,74,75} (disjoint, never seen
  during search) — mirrors M3-dev vs M3-heldout.

## Phases (each gates the next)

**Phase 0 — reproduce-before-trust (de-risk).** Run one known cell and confirm the runner, SLOs, and
telemetry behave on the new hardware before any search. Abort the campaign if the pipeline does not
produce a clean, validated result. (~2–4 cells.)

**Phase 1 — default characterization.** Measure the vLLM **default** config across the rate grid ×
seeds. This is the "beats defaults" baseline; freeze it before searching.

**Phase 2 — crossover search (find best).** Full width × rate × dev-seed grid → per-rate optimal width
via the existing crossover analysis. Emit a candidate policy (rate regimes → width).

**Phase 3 — held-out validation + beats-defaults.** Re-run the discovered policy's configs on the
held-out seed block; quantify SLO attainment and latency/throughput vs the Phase-1 default. Success =
policy meets SLO on held-out AND strictly improves on the default on the target metric, with zero
held-out SLO violations the default avoided.

**Phase 4 — measured prompt-length envelope.** Record the actual tokenized prompt-length distribution
for the 7B tokenizer over the generator's prompts; write it as the applicability report's
`prompt_tokens_min/max` span (mechanism already landed, `e0dd160`), and issue the policy's
evidence-backed envelope. No hand-set spans.

## Acceptance / stopping rules (frozen)

- Per-cell acceptance = the M3 validator passes with zero problems; one retry max, then the cell is
  recorded invalid and the phase stops (no adaptive reruns).
- A phase that fails its gate stops the campaign; the failure is recorded factually (M9-style), not
  worked around.
- No mid-campaign grid changes. Any new question after results is a *new* preregistration.
- Held-out seeds are never used for search or tuning.

## Cost estimate (firm-ish)

7B cell ≈ server startup (~1–2 min) + measured window (~45 s at qps6, ~130 s at qps2) + teardown ≈
**~4–5 min/cell**.

| Phase | ~cells | ~GPU-min |
|---|---|---|
| 0 reproduce | 4 | 20 |
| 1 default characterization | 2 rates × 3 seeds = 6 | 30 |
| 2 crossover search | 4 widths × 2 rates × 3 seeds = 24 | 110 |
| 3 held-out validation | ~2 widths × 2 rates × 3 seeds = 12 | 55 |
| 4 envelope measurement | tokenizer-only, ~0 GPU | 5 |
| **subtotal** | ~48 | **~3.7 GPU-hr** |

With model download, warmup, one-retry headroom, and debugging slack, budget **8–15 GPU-hours wall**.

- L4 / A10G at ~$0.75–1.50/hr ⇒ **~$6–22** realistic; **~$30 hard ceiling** with generous slack.
- A100 40 GB (only if we later add a 14B model) at ~$1.5–2/hr ⇒ ~$15–40.

**Bottom line for provisioning: a single ~$20–30 rental covers the whole campaign.** Earlier
$40–120 figures assumed multiple models; this focused single-model campaign is cheaper. I will report
actual GPU-hours and cost after Phase 0 so you can stop early if it drifts.

## What this proves (and does not)

Proves: the InferPilot pipeline (search → policy → held-out validation → beats-defaults) works on a
real 7B model at 24 GB, and yields the first measured prompt-length envelope. Does **not** prove
multi-model or multi-hardware generality — those are later campaigns, each preregistered.
