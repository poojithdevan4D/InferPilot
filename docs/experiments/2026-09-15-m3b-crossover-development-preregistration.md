# M3b development — fresh offline crossover study

M3b replaces the network-invalid M3 attempt; no M3 measurement is reused. The scientific design is
unchanged: rates `{2,6}` QPS, widths `{1,2,3,4}`, 128 prompt / 32 output tokens, 256 measured + 4
warm-ups, TTFT p95 ≤250 ms, TPOT p95 ≤8.5 ms, and TPOT-p95 minimization among candidates feasible in
all three blocks. Fixed engine context remains `max_num_batched_tokens=512` with chunked prefill on.

Fresh content uses `prompt_seed=3002`; fresh arrival blocks are `63,64,65`. There is one run per cell
and 24 total. The hypothesis remains: 2 QPS selects a narrower width than 6 QPS.

Before any server starts, the driver must set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; absence
of either value is fatal. Execution is seed→width→rate, with seeds 63→65, widths 1→4, and rates 2 then
6. Acceptance, the sole-drift one-retry rule, evidence preservation, and fail-closed stopping are
identical to the original M3 preregistration. Performance outcomes open only after 24/24 acceptance.
