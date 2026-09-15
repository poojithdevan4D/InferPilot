# M2D binding-token-budget development follow-up — preregistration

The first M2D development pilot found no material difference between token budgets 2048 and 4096.
This development-only follow-up tests whether a budget low enough to bind creates a real interaction
with scheduler width. A separate non-evaluative canary verified that vLLM 0.29.0 accepts and reports
`max_num_batched_tokens=512` with `enable_chunked_prefill=true`.

Fixed workload: prefill-heavy, ~1024 prompt / 16 output tokens, Poisson 4 QPS, 128 measured + 4
warm-ups, prompt seed 1001. Arrival seeds are 20, 21, 22. All candidates explicitly enable chunked
prefill. The 2×2 grid is `max_num_seqs={1,4}` × `max_num_batched_tokens={512,2048}`: 12 runs total,
one per block/cell, in the fixed interleaved order encoded in `run_binding_followup.py`.

No SLO or winner is declared. Hypothesis: 512 materially changes TTFT or TPOT at width 4 relative
to 2048, while the effect at width 1 is smaller. Every acceptance/failure/drift rule is identical to
the committed M2D development pilot. Invalid/OOM/timeout evidence is preserved and stops the study.
This is not held-out evidence and makes no significance, deployment, queue-stability, or generality claim.
