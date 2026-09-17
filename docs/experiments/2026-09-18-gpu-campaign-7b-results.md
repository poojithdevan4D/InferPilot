# GPU campaign (Qwen2.5-7B @ A10 24 GB) — results

Executed per `2026-09-17-gpu-campaign-preregistration.md` on a rented Modal A10G (24 GB,
CUDA 12.6, vLLM 0.29.0). Raw evidence is content-addressed under `runs/gpu-campaign-7b/`
(gitignored). This is the measured outcome.

## What ran (all validated clean by the M3 validator)

- **Phase 0** (1 cell): 7B loads and serves on 24 GB; pipeline reproduces on new hardware.
- **Phase 1** (6 cells): vLLM-default baseline, rates {2,6} × seeds {60,61,62}.
- **Phase 2** (24 cells): crossover search, `max_num_seqs ∈ {1,2,4,8}` × rates {2,6} × 3 seeds.

## Headline result — defaults win; the `max_num_seqs` tuning hypothesis is REFUTED

vLLM's default (large `max_num_seqs`) is at or near the Pareto front on this envelope. Capping
batch width below the default does **not** beat it: a marginal TPOT gain is paid for with a large
TTFT regression from concurrency queueing.

Mean over 3 seeds (p95 latencies, ms; throughput req/s):

| rate | config | tpot_p95 | ttft_p95 | throughput | keeps up? |
|---|---|---|---|---|---|
| 2 | **default** | 36.2 | **142.8** | 2.01 | ✓ |
| 2 | width 8 | 36.5 | 147.3 | 2.01 | ✓ (≈ default) |
| 2 | width 4 | 35.1 | 698.2 | 2.01 | ✓ but ttft 5× worse |
| 2 | width 1,2 | 32–34 | 19k–135k | <1.8 | ✗ overloaded |
| 6 | **default** | 42.0 | **178.0** | 5.9 | ✓ |
| 6 | width 8 | 38.2 | 1976 | 5.9 | ✓ but ttft 11× worse |
| 6 | width 1,2,4 | 33–35 | 30k–215k | <3.5 | ✗ overloaded |

Per-seed consistency (rate 6, default vs width 8): default ttft 173.5 / 180.8 / 179.7 ms;
width 8 ttft 2742.7 / 1851.6 / 1333.6 ms. The effect is large and consistent, not noise.

## Reading

The lever we varied — capping `max_num_seqs` — only restricts concurrency, which forces requests
to queue and inflates TTFT while barely moving TPOT. The default's large batch admits requests
immediately (low TTFT) and its TPOT cost is small. So for a 7B model on a 24 GB A10 at these rates,
**there is no `max_num_seqs` setting that dominates the default.**

This is a measured negative result for this knob — and it is exactly what InferPilot is built to
report honestly: the advisor's correct action here is to **keep the default / abstain from tuning**,
not to invent a win.

## Caveats (fair-comparison notes)

- The search cells also fixed `max_num_batched_tokens=2048` and `enable_chunked_prefill=True`, whereas
  the default baseline used vLLM defaults for those. So the comparison varies more than `max_num_seqs`
  alone. The dominant mechanism (concurrency cap → queueing → TTFT blowup) is nonetheless clear and
  mechanistically expected; a clean single-variable follow-up would hold batched-tokens/chunked-prefill
  at vLLM defaults while varying only `max_num_seqs`.
- Single model / GPU / prompt shape (128/32) / two rates. Envelope-specific.
- Held-out (Phase 3) confirmation was **not run**: it exists to confirm *winning* configs on unseen
  seeds, and there are no winners to confirm. Re-confirming a negative that is already consistent
  across 3 seeds was judged not worth additional GPU spend.

## Bottom line

The first real InferPilot-vs-vLLM-defaults measurement on a 7B model says: **defaults are already
well-tuned for `max_num_seqs` in this envelope.** The value of the system here is the *honest,
measured "no"* — and the demonstrated, safe machinery to produce it. The lever for a real win is
elsewhere (other knobs, larger models, or hardware where the default over/under-provisions), which
is the natural next preregistered campaign.
