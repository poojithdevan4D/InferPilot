# Independent critique — and our response

An external principal-engineer review (2026-09-19) audited this repo hard. It was largely correct.
This document records the critique, what we fixed immediately, what we acknowledge as open, and the
resulting narrowing of scope. Keeping it public is the point.

## Status update: the load-state redesign landed

The original response below is preserved as a historical record. The brittle saturation detector
called out by the review has since been replaced on the active diagnosis path by aligned
`LoadEvidence` and a four-state assessment (`healthy`, `near_capacity`, `overloaded`,
`indeterminate`). New runner results bind request arrivals/exits, useful token demand/delivery, queue
samples, and preemption deltas to the same measured window; conservation and completeness are checked
before a load state is named. Missing evidence now causes abstention.

The existing fp8 GPU bundles predate that instrumentation. They remain valid performance
measurements, but the redesigned advisor intentionally returns `unknown` on them. A preregistered fp8
held-out protocol exists; redesign-native collection and validation remain open. The other limitations
below remain applicable unless a later experiment document explicitly closes them.

## Fixed immediately (this commit)

1. **Reproducibility was broken** (`runs/` gitignored → headline demos couldn't run on a fresh clone).
   → Committed a small evidence bundle (`docs/evidence/`, result.json with embedded telemetry); both
   `scripts/*_demo.py` now read it first. Verified the demos run with `runs/` removed. "Reproducible"
   is now actually true for the demos.
2. **`CapacityAdvisory` correctness bugs:**
   - Goodput counted SLO-violating completions when the run failed SLO. → **Goodput is now
     SLO-compliant only** (0 when SLO not met; the true ceiling needs a rate sweep).
   - Cost/token was computed even when SLO failed. → **Cost only reported for an SLO-compliant run.**
   - It recommended *tuning a deployment that already met SLO + target*. → **Adequate now wins; we do
     not tune a passing deployment.**
   - Scale text could say "sustains X qps under SLO" when `met_slo=False`. → **Honest wording**
     ("does NOT meet SLO at offered X qps"); GPU count is **not** fabricated from an overloaded run;
     the linear GPU estimate is labeled first-order/sublinear.
3. **Overstated language aligned with evidence:**
   - "law" → **empirical heuristic / hypothesis** (small, partly post-hoc matrix; not held-out).
   - "quality-verified" / "benign KL" → **quality smoke-tested**; the teacher-forced-KL preflight
     **failed** (mean >0.01, p99 0.39) — we no longer imply "lossless".
   - Test-count claims now note the suite checks **contract self-consistency, not diagnostic accuracy
     on real deployments**.

## Acknowledged as open (honestly, not yet fixed)

- **The fp8/preemption rule is a heuristic, not a causal law.** It was invented after a contradiction
  and evaluated on a tiny matrix with single runs. 14B/Mistral show preemptions *not* going to zero,
  which weakens "fp8 wins by killing preemption." **Needs** a preregistered held-out matrix (≥20 cells,
  ≥3 randomized repetitions, sweep the saturation boundary, record preempted/recomputed tokens not just
  event count, predict effect size first, hunt counterexamples).
- **Preemption is treated categorically** (>0). Should be preempted/recompute tokens per useful output
  token, normalized by load.
- **Historical saturation detector was brittle** (first-half vs second-half TTFT, hard 1.5×,
  ignored failed/timed-out requests, no "unknown" state). It remains available only as a clearly
  labeled legacy trend report; the diagnosis path now requires aligned `LoadEvidence` as described
  above. Empirical validation of the replacement is still open.
- **`decode_heavy = output ≥ prompt`** is a crude threshold, not a compute model.
- **Fit engine is a memory calculator** (fixed 2 GB overhead, no CUDA graphs/fragmentation/kernel
  workspace; ranks by memory-fit + sticker price, not measured throughput). "Fits" ≠ "right-sized".
- **Quality evidence is thin** (16 easy QA, one needle template, top-20 KL, Qwen-3B only) and not wired
  into a single mandatory deployment policy.
- **"Self-validating" means schema/formula self-consistency, not empirical truth.**
- **Engine-agnostic ≠ metric-name adapters**: scheduler/cache/preemption *semantics* differ.

## The narrowed, honest claim

Not "an autonomous inference engineer." The defensible wedge is:

> **Given a production trace and an existing vLLM deployment, run controlled canaries and produce an
> auditable capacity-vs-SLO curve: the cheapest configuration that meets this workload's SLO, with
> measured quality and a rollback command.**

Offline/canary advisor first — **not** an autonomous controller that mutates production.

## Roadmap that follows from the critique

1. Real production-trace ingestion + a **rate sweep around the SLO boundary** (not a single overloaded
   point). *(WorkloadTrace exists; the sweep is next.)*
2. Preregistered held-out validation of the fp8 heuristic (the matrix above).
3. Repair the fit engine toward measured throughput, not memory-fit.
4. Higher-value levers the review ranked above KV dtype: **prefix-cache eligibility/routing, weight
   quantization + GPU/TP selection, chunked-prefill tuning, engine/version regression testing.**
5. A one-week **concierge deployment audit** with a real design partner — the actual validation.

## Kill criterion (from the review, adopted)

Ten interviews with qualified platform owners; five free concierge audits. Kill or reposition if:
(a) <3 will share a sanitized trace / allow a non-prod benchmark; (b) across 3 audits, InferPilot finds
no recommendation worth ≥15% cost, a material SLO improvement, or a prevented bad purchase beyond engine
defaults / NVIDIA Dynamo; (c) nobody commits to pay after seeing their own result. Synthetic experiments
cannot answer whether anyone buys it.
