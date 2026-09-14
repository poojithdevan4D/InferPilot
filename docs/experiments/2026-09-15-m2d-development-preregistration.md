# M2D development pilot — preregistration

## Purpose and evidence boundary

This pilot characterizes a two-dimensional scheduler grid under three workload shapes. It is
**development evidence**: its outcomes may be used to choose SLO thresholds and a deterministic
search policy. It is not held-out evaluation, and no result from this pilot may later be reported
as held-out evidence. No SLO or winner is declared before measurement.

## Fixed system and candidate grid

The hardware/runtime/model remain the pinned RTX 3050 Laptop GPU, Python 3.12, vLLM 0.29.0,
PyTorch sampler, and Qwen2.5-0.5B-Instruct revision used by C5–C7. All engine fields remain fixed
except:

| Candidate | `max_num_seqs` | `max_num_batched_tokens` |
|---|---:|---:|
| s1-t2k | 1 | 2048 |
| s1-t4k | 1 | 4096 |
| s4-t2k | 4 | 2048 |
| s4-t4k | 4 | 4096 |

The 2×2 factorial is the smallest grid that can expose both main effects and their interaction.
Both token-budget levels are at least `max_model_len=2048`, avoiding a forced chunked-prefill
coupling.

## Fixed workload shapes

Each cell has 4 sequential warm-ups and 128 measured requests. Generation is greedy with exact
requested output length (`ignore_eos=true`). Prompt seed is fixed within each shape; arrival seeds
10, 11, and 12 form three independent arrival blocks.

| Shape | Prompt target | Output target | Arrival process | Nominal rate | Prompt seed |
|---|---:|---:|---|---:|---:|
| prefill | ~1024 | 16 | poisson-v1 | 4 QPS | 1001 |
| decode | ~64 | 256 | poisson-v1 | 1 QPS | 1002 |
| burst | ~128 | 32 | batched-poisson-v1, burst size 4 | 6 QPS | 1003 |

Prompt targets are approximate because the current generator is word-based; server-reported input
tokens remain the measured provenance. Batched Poisson is a bounded synthetic burst model, not a
production trace.

## Hypotheses

1. `max_num_seqs` and `max_num_batched_tokens` interact for prefill-heavy or burst traffic.
2. The token budget has little effect on decode-heavy traffic at fixed scheduler width.
3. No single candidate is necessarily best across all three shapes.

These hypotheses may be falsified. A null interaction or one universally dominant candidate is a
valid result.

## Run count, ordering, and validity

Total: 3 shapes × 3 arrival blocks × 4 candidates × 1 repetition = **36 runs**. Execution is
interleaved by arrival seed and shape, with candidate order rotated/reversed across blocks to spread
thermal/time drift. A cell is accepted only if it is COMPLETED, baseline-eligible, 128/128 requests
succeed, effective configuration is verified, telemetry and `phases.json` are complete, lifecycle
has no pre-teardown errors, and scheduled/actual arrival evidence passes the fixed drift gates.

Any invalid/OOM/timeout cell is preserved and the study stops. It is not silently retried or
replaced. Resume skips only previously accepted and revalidated cells.

## Post-pilot decision protocol

Report every candidate and block without selecting a production winner. Use development results to
propose per-shape SLOs, candidate-count/GPU-time budgets, and a deterministic policy. Freeze those
choices and the held-out configurations in a later commit before any held-out outcomes are opened.
Do not claim online adaptation, queue stability, production safety, or hardware/model generality.
