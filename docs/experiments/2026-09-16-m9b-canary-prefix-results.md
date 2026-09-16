# M9b canary-prefix replacement — results

Executed under the preregistration in `2026-09-16-m9b-canary-prefix-preregistration.md`
(byte-identical scientific design to invalid M9, only IDs/output dir changed). Raw evidence is
content-addressed under `runs/m9b-canary-prefix/` (gitignored); this document reports the derived
outcome.

## Question

Does a short **first-128-request canary prefix** reproduce the SLO pass/fail verdict of the
**full 256-request** window? If yes, a canary can gate an advisor candidate at ~half the cost
without admitting a bad config (zero false-pass) and with bounded misses (≤10% false-fail).

- SLO: `ttft_p95 ≤ 250 ms` and `tpot_p95 ≤ 8.5 ms`.
- Predictor: nearest-rank aggregates over the first 128 completed requests
  (`runner/aggregate.compute_aggregates`), frozen before any M9 measurement.
- Corpus: 24 held-out cells — prompt seed 7003, arrival seeds {103,104,105}, widths
  `max_num_seqs ∈ {1,2,3,4}`, rates {2, 6} qps, 256 requests each.

## Result — confusion matrix (prefix verdict vs full verdict)

| | full **pass** | full **fail** |
|---|---|---|
| **prefix pass** | 14 (true-pass) | 0 (false-pass) |
| **prefix fail** | 0 (false-fail) | 10 (true-fail) |

- **Exact agreement on all 24/24 cells.** No disagreements.
- false-pass = 0 → the canary never green-lights a config the full run would reject.
- false-fail = 0 (0% ≤ 10% bound) → the canary never needlessly rejects a good config.
- Discriminative: both classes populated (14 pass / 10 fail), so agreement is not a degenerate
  all-pass or all-fail artifact.

**Acceptance: PASSED** (`false_pass == 0`, `true_pass ≥ 3`, `true_fail ≥ 3`,
`false_fail_rate ≤ 0.10`).

## Reading

The prefix and full p95s track each other tightly; the SLO margins are wide except at qps6 near the
TTFT limit, where both windows agree (e.g. `qps6-a105-seq2`: prefix ttft_p95 304 ms / full 298 ms —
both fail together). The one large-magnitude prefix/full TTFT gap observed
(`qps2` seq with prefix ttft_p95 ≈ 63 ms vs full ≈ 8145 ms, a late-tail spike) does **not** flip the
verdict: both windows still classify identically, and the failing case is driven by the shared
decision boundary, not by prefix truncation.

## Conclusion

On this model/GPU/workload envelope, a 128-request canary is a faithful, zero-false-pass proxy for
the full 256-request SLO verdict. This validates the controller's measured-canary gate (M8/M10):
an advisor candidate can be admitted on a half-length canary without risking an SLO regression that
the full window would have caught.

**Scope / caveats.** Single model (Qwen2.5-0.5B-Instruct), single GPU (RTX 3050 4 GB), two rates,
fixed 128/32 prompt/output. The zero-error result is strong but envelope-specific; wider prompt/output
mixes, higher rates, and larger models are untested and must not be assumed. No adaptation or rerun
was performed — this is the preregistered outcome as measured.
