# M2D development pilot — results

## Outcome

All **36/36 preregistered cells** completed and passed the acceptance gates: baseline eligible,
128/128 successful requests, verified effective configuration, complete telemetry and phase timing,
clean pre-teardown lifecycle, and valid arrival evidence. The study consumed **1.065 server-process
hours**. Worst dispatch-drift p95 remained below 4.32 ms (10 ms gate); no OOM or timeout occurred.

This is development evidence. No SLO was preregistered, no production winner is selected, and these
outcomes are not held-out evidence.

## Mean across three arrival seeds

Values are means of per-run metrics. `tok` is `max_num_batched_tokens`.

| Shape | seq | tok | TTFT p95 (ms) | TPOT p95 (ms) | E2E p95 (ms) | Throughput (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 2048 | 564.38 | 6.26 | 657.73 | 65.88 |
| prefill | 1 | 4096 | 568.54 | 6.26 | 662.08 | 65.88 |
| prefill | 4 | 2048 | 215.20 | 21.23 | 451.23 | 65.96 |
| prefill | 4 | 4096 | 217.67 | 21.04 | 449.15 | 65.96 |
| decode | 1 | 2048 | 76945.68 | 6.28 | 78547.20 | 158.36 |
| decode | 1 | 4096 | 76577.11 | 6.29 | 78180.30 | 158.63 |
| decode | 4 | 2048 | 506.44 | 7.08 | 2277.55 | 261.87 |
| decode | 4 | 4096 | 508.03 | 7.07 | 2277.61 | 261.86 |
| burst | 1 | 2048 | 6716.07 | 6.27 | 6909.68 | 150.57 |
| burst | 1 | 4096 | 6714.19 | 6.27 | 6908.02 | 150.54 |
| burst | 4 | 2048 | 567.70 | 7.80 | 790.35 | 194.06 |
| burst | 4 | 4096 | 566.01 | 7.80 | 789.14 | 194.06 |

Actual mean prompt-token counts were stable across candidates: 1024.26 (prefill), 64.25 (decode),
and 128.33 (burst).

## Hypothesis assessment

1. **No evidence of the preregistered token-budget interaction.** Changing 2048→4096 produced
   essentially identical throughput in every cell. Mean paired TTFT/TPOT effects were generally
   below about 1%; directions varied across seeds. At these workload sizes, 2048 does not bind the
   scheduler, so 4096 cannot expose a meaningful second dimension.
2. **Decode null-effect hypothesis supported descriptively.** At fixed sequence width, the token
   budget did not materially affect decode-heavy results.
3. **Sequence width matters, but remains a trade-off.** Width 4 drastically reduced queueing TTFT
   and increased decode/burst throughput, while raising TPOT. Prefill throughput tracked offered
   output load and was unchanged; width 4 exchanged lower TTFT for substantially higher TPOT.

## Decision

Do **not** open a held-out corpus using this 2048/4096 grid. It would spend GPU time retesting an
inactive dimension and could not support the Milestone-2 larger-search claim.

The next development pilot should test a token budget low enough to bind (for example 512 versus
2048) with `enable_chunked_prefill=true` explicitly requested and fidelity-verified. First run one
non-evaluative startup/measurement canary to establish that vLLM 0.29.0 accepts and reports the
lower budget on this hardware. If controllable, preregister a reduced follow-up development matrix.
Only after finding a meaningful second dimension should SLOs/search budgets be frozen and held-out
evidence collected.

## Anti-claims

This bounded synthetic study does not establish production traffic realism, queue stability,
statistical significance, deployment safety, or generality beyond this model/runtime/GPU. Arrival
seeds vary schedules while prompt content remains fixed; three blocks and one repetition per cell
are descriptive rather than confidence evidence.
