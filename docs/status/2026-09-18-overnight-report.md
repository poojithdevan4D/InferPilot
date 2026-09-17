# Overnight report — 2026-09-18 → morning

Ran autonomously overnight on Modal (superfloat workspace, A10 24 GB). All committed, tests green,
containers torn down, nothing left running.

## Headline

Turned the first campaign's negative into a materially better, better-evidenced InferPilot, and ran a
second (decode-heavy) campaign end-to-end. **No robust beats-defaults win emerged on 7B/A10 — but the
system now demonstrably refuses false wins**, which is the point.

## What landed (commits)

- `ee3421d` **Fail-closed Pareto config comparison** (`compare_configs`) — a candidate beats the
  incumbent only if it keeps up AND is no-worse on every latency metric AND strictly better on ≥1.
  Kills the single-metric illusory win from run 1. Evaluator + generator confound also fixed.
- `217b5c4` **Controller 0.3.0** — the online loop now gates apply/rollback on `compare_configs`
  (candidate must Pareto-dominate the current config), not an absolute SLO.
- `6d86adb` **Online-loop demo on real evidence** — width-8 vs default → rollback, keeps default.
- `d47f24d` **Drain-robust saturation signal** (`detect_saturation`) — the decode campaign exposed that
  throughput/duration mis-reads feasibility for long generations; TTFT-stability is the correct signal.
- `3d049d6`, `fd1aa9b`, decode scaffold — campaign-agnostic harness + decode-heavy campaign.
- Results docs: `2026-09-18-gpu-campaign-7b-results.md`, `...-decode-results.md`.

Test count grew to **389**; build clean throughout.

## The decode campaign (the night's experiment)

Hypothesis: decode-heavy load → default over-admits → a `max_num_seqs` cap wins. Full 4-phase run.
- The A10 is decode-compute-bound (~2.2 req/s); rate retargeted 4→2 for feasibility.
- Dev seeds showed widths 64/128 dominating the default 3/3 — a candidate win.
- **Held-out seeds refuted it**: on unseen seeds default and 64/128 are indistinguishable (<1%); the
  dev win was a ~5% high default reading on those seeds. Widths 16/32 genuinely saturate.
- Verdict: **no robust win; keep the default.** Second consistent negative — `max_num_seqs` is not a
  useful lever on this model/GPU across short- and long-output workloads.

## Honest state of the goal

InferPilot is now harder to fool (offline + online + a fixed feasibility metric) and better-evidenced.
It has **not** yet produced a confirmed beats-defaults win — because on 7B/A10 the default is
well-provisioned and the bottleneck is compute, which the knob we varied can't touch. That's a real,
defensible finding, not a failure of the system.

## Cost

Modal, per-second, superfloat only. Session total ≈ **$9–11 (₹800–950)** — under the ₹1,500 self-cap
and the $30 hard cap. Nothing idling.

## The one decision for you

A confirmed win most likely lives in a **KV-bound regime** the A10+7B+short-context setup doesn't reach:
either (a) **long context** (e.g. 4k+ tokens) so KV pressure forces the default to preempt, or (b) a
**bigger model / tighter-VRAM GPU** where the default over-provisions. Both need your green-light
(the second costs a bit more GPU). Say which and I'll preregister + run it. Meanwhile the codebase
improvements stand on their own.
