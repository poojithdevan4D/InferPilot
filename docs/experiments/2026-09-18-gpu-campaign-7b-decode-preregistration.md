# Decode-heavy GPU campaign — preregistration

Frozen before spend. Follow-up to the 128/32 run, whose honest finding was: at an
un-stressed operating point the vLLM default trivially wins `max_num_seqs`. This campaign
moves to a regime where the knob should actually matter.

## Hypothesis

At a **decode-heavy** operating point (long outputs) under load, the vLLM default
(`max_num_seqs` ~256) over-admits requests, so their KV caches contend for a fixed budget
and vLLM preempts/recomputes — inflating TTFT and TPOT. A **moderate `max_num_seqs` cap**
should reduce that contention and **beat the default** (Pareto-dominate: no worse on
ttft_p95/tpot_p95, strictly better on ≥1, while still keeping up with the offered rate).

Directional prediction: a mid cap (32 or 64) dominates the default; too-small caps (16)
overload; too-large caps (128) approach the default. If no cap dominates, the result is a
second honest negative and the lever is elsewhere.

## Fixed design

- Model/GPU/stack: Qwen2.5-7B-Instruct @ pinned rev `a09a354…`, Modal A10 24 GB, vLLM 0.29.0,
  PyTorch sampler — identical to the prior campaign.
- Workload: prompt 128, **output 512** (≈16× the KV growth of the 32-token run), poisson-v1,
  **rate 4 qps**, 96 requests/cell, 4 warmup. `max_model_len=1024`, `gpu_memory_utilization=0.90`.
- **Single variable:** search cells differ from the default in ONLY `max_num_seqs`.
- Grid: `max_num_seqs ∈ {16, 32, 64, 128}` × dev seeds {60,61,62}; held-out {73,74,75}.
  Phase 1 = vLLM-default baseline (no cap) × 3 seeds.

## Phases + decision rule

Phase 0 reproduce/validate (1 cell — confirm 7B + 512-token decode loads, does not OOM, completes
clean) → Phase 1 default baseline → Phase 2 crossover search. **Winner = a width whose
`compare_configs` verdict vs the same-seed default is `candidate_dominates` on a MAJORITY of seeds**
(fail-closed Pareto, feasibility-first). Phase 3 confirms any winners on held-out seeds.

## Rules

M3-style per-cell acceptance (counts, verified config, telemetry, phases, arrival drift ≤10 ms p95),
one retry on drift only, terminal-invalid stops the phase. No mid-campaign grid changes. Held-out
seeds never used for search. Report cost at the Phase-0 gate; hard cap unchanged.

## Cost

Decode-heavy cells are slower (512-token gens). ~16 cells (Phases 0–2) × ~1–2 min ≈ **20–35 GPU-min
⇒ ~₹80–150**. Phase 3 only if winners exist. Same ~$30 hard cap.
