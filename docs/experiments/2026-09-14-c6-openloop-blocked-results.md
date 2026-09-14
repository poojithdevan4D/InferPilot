# C6 results — blocked open-loop `max_num_seqs` confirmation

**Date:** 2026-09-14

**Preregistered protocol:** commit `a89817d`

**Decision:** `no_feasible_candidate` under the fixed experimental policy

## Fixed decision policy

C6 was committed before measurement with five consecutive unseen workload seeds
(`1..5`), four candidates (`max_num_seqs=1..4`), one run per cell, and no use of
C5 as an evaluation block. A candidate had to satisfy both TTFT p95 ≤ 250 ms and
TPOT p95 ≤ 8.5 ms in **every** block. Robust-feasible candidates would then be
ranked by mean block-level TPOT p95 (lower is better). No seed, threshold,
candidate, or objective was changed after results were observed.

## Execution validity

All 20 preregistered cells were accepted and ingested:

- 20/20 `COMPLETED`, baseline-eligible, and 256/256 successful;
- resolved engine configuration verified in every run;
- complete measured-window telemetry and no pre-teardown lifecycle errors;
- `arrivals.json` present and complete in every run;
- worst dispatch-drift p95: **1.367 ms** (gate: 10 ms);
- worst individual dispatch drift: **119.615 ms** (gate: 250 ms).

Two isolated dispatch outliers above 100 ms were retained because their run-level
p95 drift was about 1.2 ms and both were inside the preregistered gates. Excluding
them after observing results would violate the protocol.

## Arrival blocks

| Seed | Scheduled span (s) | Realized schedule QPS | Maximum arrivals in 250 ms / 1 s |
|---:|---:|---:|---:|
| 1 | 31.034 | 8.217 | 7 / 19 |
| 2 | 31.848 | 8.007 | 9 / 19 |
| 3 | 33.012 | 7.724 | 7 / 16 |
| 4 | 33.022 | 7.722 | 8 / 14 |
| 5 | 34.050 | 7.489 | 10 / 19 |

The realized dispatch rate matched each scheduled rate to three decimal places.
Seed 5 had the lowest average rate but the largest 250-ms burst, illustrating why
a finite schedule's average QPS alone does not describe its queueing pressure.

`WorkloadSpec.seed` controls both prompt sampling and the Poisson schedule. Blocks
therefore represent whole synthetic workload samples, not arrival-only changes.
Mean counted prompt length remained tightly controlled (128.238–128.273 tokens).

## Primary metrics

### TTFT p95 (ms; limit 250)

| Seed | seq1 | seq2 | seq3 | seq4 |
|---:|---:|---:|---:|---:|
| 1 | 22018.5 | 2250.1 | 733.5 | **247.7 ✓** |
| 2 | 25054.1 | 3846.4 | 450.7 | **240.4 ✓** |
| 3 | 20163.2 | 2051.9 | 259.2 | **105.3 ✓** |
| 4 | 21079.8 | 981.7 | 255.2 | **170.1 ✓** |
| 5 | 19222.7 | 1758.1 | 586.3 | **210.8 ✓** |

### TPOT p95 (ms; limit 8.5)

| Seed | seq1 | seq2 | seq3 | seq4 |
|---:|---:|---:|---:|---:|
| 1 | **6.652 ✓** | **7.628 ✓** | **8.330 ✓** | 8.548 |
| 2 | **6.891 ✓** | **7.884 ✓** | **8.269 ✓** | 8.584 |
| 3 | **6.680 ✓** | **7.804 ✓** | **8.203 ✓** | **8.421 ✓** |
| 4 | **6.671 ✓** | **7.751 ✓** | **8.170 ✓** | **8.340 ✓** |
| 5 | **6.669 ✓** | **7.637 ✓** | **8.038 ✓** | **8.324 ✓** |

### Output throughput (tokens/s)

| Seed | seq1 | seq2 | seq3 | seq4 |
|---:|---:|---:|---:|---:|
| 1 | 150.2 | 253.4 | 261.7 | 261.7 |
| 2 | 140.7 | 228.0 | 254.4 | 254.6 |
| 3 | 148.9 | 232.3 | 245.2 | 246.0 |
| 4 | 148.2 | 241.5 | 246.2 | 246.2 |
| 5 | 149.6 | 227.7 | 238.1 | 238.9 |

## Blocked decision

| `max_num_seqs` | TTFT blocks passed | TPOT blocks passed | Fully feasible blocks | Robust-feasible | Mean TPOT p95 (ms) |
|---:|---:|---:|---:|:---:|---:|
| 1 | 0/5 | 5/5 | 0/5 | no | 6.713 |
| 2 | 0/5 | 5/5 | 0/5 | no | 7.741 |
| 3 | 0/5 | 5/5 | 0/5 | no | 8.202 |
| 4 | 5/5 | 3/5 | 3/5 | no | 8.443 |

No candidate satisfies the intersection of the two constraints in all five
blocks, so the evaluator correctly returns `no_feasible_candidate`; it does not
rank or recommend a setting. `seq4` is the only candidate that always meets the
TTFT constraint, but it exceeds the TPOT limit by 0.048 ms and 0.084 ms in seeds
1 and 2. Those small misses are still failures under the fixed fail-closed rule.

## What C6 changed in our understanding

1. **The scheduling trade-off is robust.** Wider scheduling sharply reduces
   queueing while modestly increasing decode-step latency; no single setting is
   intrinsically best without policy.
2. **C5 did not establish a 7.5 requests/s capacity ceiling.** Its ~240 output
   tokens/s matched its realized offered load. In C6, `seq4` completed the 8.217
   QPS block at 261.7 output tokens/s while keeping TTFT p95 below 250 ms. The
   preregistered capacity-based expectation of universal overload is therefore
   falsified.
3. **Average arrival rate is insufficient.** Seed 5's lower average rate still
   produced seq3 TTFT p95 of 586 ms, plausibly because of its stronger short
   burst. A future controller needs workload features beyond mean QPS.
4. **The current policy sits on a narrow trade-off boundary.** A 0.1-ms TPOT
   policy change could alter feasibility, but changing it after C6 would be
   invalid. Threshold sensitivity should be reported separately, never used to
   rewrite this result.

## Resource context

Peak GPU memory was 3584 MiB for seq1/2 and 3606 MiB for seq3/4. Peak reported
KV-cache usage remained below about 0.4%, so C6 characterizes scheduling and
queueing, not KV-pressure behavior.

## Limitations

- One observation per cell: the five blocks sample workload variability but do
  not separately estimate runtime noise inside each seed/candidate cell.
- The run window is only about 31–34 seconds of arrivals plus tail completion;
  this is not a long-horizon stability proof.
- Synthetic fixed-shape prompts on one 0.5B model and one laptop GPU do not
  establish generality across models, hardware, or real traces.
- Robust feasibility is a conservative observed-sample rule, not a confidence
  bound, statistical-significance result, or deployment guarantee.
- The workload seed jointly changes prompts and arrivals, so causal claims about
  arrival timing alone are unsupported.

## Reproducibility

- Protocol/configs/driver: `experiments/c6-openloop-blocked/` at `a89817d`.
- Raw bundles, integrity-checked store, `manifest.jsonl`, and machine-readable
  blocked report: `runs/c6-openloop-blocked/` (gitignored).
- Evaluator: blocked-study report `0.1.1`; `ExperimentResult` schema `0.3.0`.

The machine-readable result was produced with:

```bash
.venv-bench/bin/python -m inferpilot.comparison blocked-evaluate \
  runs/c6-openloop-blocked/store \
  experiments/c6-openloop-blocked/study.json \
  --output runs/c6-openloop-blocked/blocked-report.json
```
