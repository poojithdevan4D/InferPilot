# C7 results — scheduler width across workload-rate regimes

**Date:** 2026-09-14

**Preregistered protocol:** commit `7f2871b`

**Primary result:** 4 QPS selected `max_num_seqs=3`; 6 QPS selected
`max_num_seqs=4`; 8 QPS returned `no_feasible_candidate`.

## Fixed policy and validity

C7 used three independent blocked studies at nominal 4, 6, and 8 QPS. Each used
unseen consecutive workload seeds 6–8 and the same candidates (`max_num_seqs`
1–4), TTFT p95 ≤ 250 ms, TPOT p95 ≤ 8.5 ms, and TPOT-p95 minimization among
robust-feasible candidates. All choices were committed before measurement.

All 36/36 cells were COMPLETED, baseline-eligible, configuration-verified,
telemetry-complete, lifecycle-clean, and 256/256 successful. Every run had a
complete arrival artifact. Worst dispatch-drift p95 was 1.772 ms (10-ms gate);
worst individual drift was 13.061 ms (250-ms gate). No cell was excluded or
repeated.

## Per-rate decisions

| Nominal rate | Robust-feasible candidates | Decision | Why |
|---:|---|---|---|
| 4 QPS | seq3, seq4 | **select seq3** | Both pass every block; seq3 has lower mean TPOT p95 (7.835 vs 7.898 ms) |
| 6 QPS | seq4 only | **select seq4** | seq3 fails TTFT in seed 7; seq4 passes both limits in all blocks |
| 8 QPS | none | **no feasible candidate** | seq4 fails TTFT in seed 7; seq1–3 fail TTFT in all blocks |

This supports the preregistered hypotheses: the minimum robust-feasible scheduler
width increases with offered load, and at least two regimes require different
actions. At 8 QPS the correct action under the declared policy is not to force a
winner from the measured candidate set.

## Block-level SLO evidence

Each cell below is `TTFT p95 / TPOT p95` in milliseconds. A check mark means both
preregistered constraints passed.

### Nominal 4 QPS

| Seed (realized QPS) | seq1 | seq2 | seq3 | seq4 |
|---|---:|---:|---:|---:|
| 6 (3.967) | 1873.4 / 6.622 | 253.1 / 7.570 | **66.8 / 7.729 ✓** | **41.8 / 7.960 ✓** |
| 7 (4.138) | 2019.3 / 6.656 | 441.1 / 7.619 | **155.8 / 7.628 ✓** | **60.2 / 7.919 ✓** |
| 8 (3.841) | 1982.2 / 6.751 | 257.8 / 7.718 | **80.1 / 8.147 ✓** | **39.3 / 7.816 ✓** |

### Nominal 6 QPS

| Seed (realized QPS) | seq1 | seq2 | seq3 | seq4 |
|---|---:|---:|---:|---:|
| 6 (5.950) | 10588.2 / 6.673 | 816.1 / 7.556 | **196.5 / 8.060 ✓** | **80.2 / 8.355 ✓** |
| 7 (6.207) | 12846.0 / 6.702 | 850.5 / 7.613 | 362.0 / 8.177 | **171.7 / 8.369 ✓** |
| 8 (5.762) | 9057.7 / 6.306 | 595.3 / 7.308 | **171.4 / 7.666 ✓** | **63.7 / 7.957 ✓** |

### Nominal 8 QPS

| Seed (realized QPS) | seq1 | seq2 | seq3 | seq4 |
|---|---:|---:|---:|---:|
| 6 (7.933) | 20699.2 / 6.661 | 1466.0 / 7.639 | 508.9 / 8.064 | **154.4 / 8.388 ✓** |
| 7 (8.276) | 22591.0 / 6.740 | 2743.7 / 7.643 | 528.4 / 8.062 | 300.9 / 8.353 |
| 8 (7.683) | 19592.0 / 6.716 | 1873.7 / 7.658 | 396.0 / 8.117 | **135.1 / 8.351 ✓** |

## Cohort means

| Rate | seqs | TTFT p95 (ms) | TPOT p95 (ms) | E2E p95 (ms) | Output throughput (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1958.3 | 6.676 | 2160.4 | 125.9 |
| 4 | 2 | 317.3 | 7.636 | 544.1 | 127.3 |
| 4 | 3 | 100.9 | 7.835 | 329.8 | 127.4 |
| 4 | 4 | 47.1 | 7.898 | 281.4 | 127.5 |
| 6 | 1 | 10830.6 | 6.560 | 11026.2 | 150.0 |
| 6 | 2 | 754.0 | 7.492 | 976.1 | 189.5 |
| 6 | 3 | 243.3 | 7.968 | 478.8 | 190.6 |
| 6 | 4 | 105.2 | 8.227 | 338.3 | 190.8 |
| 8 | 1 | 20960.7 | 6.706 | 21165.6 | 149.4 |
| 8 | 2 | 2027.8 | 7.647 | 2255.5 | 245.7 |
| 8 | 3 | 477.8 | 8.081 | 715.2 | 252.4 |
| 8 | 4 | 196.8 | 8.364 | 437.6 | 253.5 |

## Architectural implications

1. **A workload-rate-aware decision is necessary.** The lowest robust-feasible
   width moves from 3 to 4 between 4 and 6 QPS, then the measured action space
   becomes infeasible at 8 QPS.
2. **`no_feasible_candidate` must remain a first-class controller output.** A
   future controller needs an escalation action—admission control, scale-up/out,
   relaxed product policy, or explicit refusal—rather than always returning an
   engine configuration.
3. **The switching benefit is not established yet.** At 4 QPS, seq3 improves
   mean TPOT p95 over the also-feasible seq4 by only 0.064 ms (~0.8%). With one
   run per cell, this is an observed policy result, not evidence that restarting
   from seq4 to seq3 is worth its disruption cost.
4. **Mean QPS is not a sufficient state representation.** Within a nominal rate,
   identical engine settings can cross the TTFT boundary under different finite
   workload samples. Burst features and uncertainty must be inputs to any future
   policy.
5. **Throughput mostly tracks offered work once the scheduler is wide enough.**
   It is not useful as the sole optimization objective in these under/near-load
   regimes; SLO feasibility plus TPOT exposes the meaningful trade-off.

## Resource context and limitations

Peak GPU memory was about 3584–3586 MiB for seq1/2 and 3604–3606 MiB for seq3/4;
KV pressure remained negligible. These are scheduling results, not KV-management
results.

The experiment uses one observation per cell, three workload blocks per rate,
short synthetic fixed-shape traffic, one 0.5B model, and one laptop GPU. The seed
jointly changes prompt content and arrival timing. The blocked decisions are
deterministic observed-sample outputs, not significance tests, confidence bounds,
long-horizon stability proofs, or deployment guarantees.

## Reproducibility

- Preregistered configs, studies, and driver: `experiments/c7-rate-regimes/` at
  commit `7f2871b`.
- Raw run bundles, integrity-checked store, manifest, and three machine-readable
  blocked reports: `runs/c7-rate-regimes/` (gitignored).
- Result schema `0.3.0`; blocked-study report `0.1.1`.
