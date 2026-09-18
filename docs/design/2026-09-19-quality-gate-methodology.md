# Quality-gate methodology for precision-reducing levers (fp8/quantized KV)

Synthesis of an external expert review (DeepSeek, 2026-09-19) + our own measurement (fp8 on 3B:
12% greedy-token agreement but 0/16 QA regressions). A throughput win must never be reported without
a defensible quality check; this defines that check.

## Ranked methods (cost vs signal)

1. **Teacher-forced per-position KL (primary, cheapest defensible).** One forward pass per config over
   a fixed corpus; both configs consume the SAME token history, so it measures pure distributional
   shift without the autoregressive cascade. Gate: **mean KL ≤ 0.01 AND p99 KL ≤ 0.1**, but
   **noise-floor relative** — first measure KL between two bf16 seeds; a candidate KL is only meaningful
   above that floor (budget = max(abs threshold, floor × ~3)). p99 matters: low mean + high p99 =
   damage concentrated on few positions = the long-context risk. → `KLQualitySpec` / `evaluate_kl_quality`.
2. **Needle-in-haystack at max context (non-negotiable complement).** KL is structurally blind to
   long-range retrieval collapse (the documented fp8 91%→13% failure), which contributes ~nothing to
   average next-token loss. Test at multiple depths at your max supported context. → `NeedleQualitySpec`
   / `evaluate_needle_quality`.
3. **Canonical-task accuracy** (GSM8K / structured-output exact-match / multi-hop) — catches
   instruction-following and reasoning regressions a small QA set misses.
4. **LLM-judge** — tiebreaker only (model-dependent, hard to reproduce).

**Skip:** perplexity (scalar collapse of the same info KL gives, dominated by easy tokens) and
**greedy token agreement** (measures trajectory divergence, not quality — kept only as a cheap
drift *pre-filter*, never a verdict).

## Where fp8 KV breaks (encode as preconditions)

- **Long-context retrieval** (biggest risk): keys break before values (softmax exponentiates key
  error); per-tensor K+V worst, V-only best. Needle test mandatory.
- **Hybrid/sliding-window attention layers** — keep them in bf16 (cheap), fp8 only the full-attn layers.
- **head_dim=256** — prefill can regress.
- **Kernel/deployment-path silent corruption** (e.g. CUDA-graph capture attending a truncated view) —
  fluent short output, broken long recall; only needle catches it.
- **Small models** (e.g. 1B) — degrade under fp8 KV; use only when capacity is overriding.

## What even KL + needle miss (documented, not hidden)

Free-running generation drift over long horizons (compounding errors → early-stop/repetition on hard
CoT), multi-hop reasoning, and structured-output validity under interacting constraints. These need a
trajectory-level or task-specific harness — teacher-forced ≠ agent-trajectory regime.

## InferPilot gate (current)

`recommend_plan` marks an fp8 recommendation **apply-BLOCKED** until a quality gate passes. Contracts
built (GPU-free, self-validating): `KLQualityGate` (primary), `NeedleQualityGate` (long-ctx complement),
`QualityGate` (greedy drift pre-filter). **Remaining GPU work:** the measurement harness — capture
teacher-forced logits with/without fp8 (vLLM `prompt_logprobs`; full-vocab KL needs logits access, top-k
is an approximation) + a needle-in-haystack run at max context — to populate these gates on real models.
