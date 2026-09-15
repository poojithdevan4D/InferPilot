# M2D development shape completion — preregistration

Development-only completion of the explicit-chunked `max_num_seqs={1,4}` ×
`max_num_batched_tokens={512,2048}` grid for decode and burst workloads. Workload parameters match
the first M2D pilot; prompt seeds remain 1002/1003 and new arrival blocks are 30, 31, 32. Each cell
uses 4 warm-ups and 128 measured requests. Total: 24 runs, fixed order in `run_shape_completion.py`.

No SLO or winner is declared. Hypothesis: the token-budget effect is smaller for decode and burst
than for prefill, while width remains the dominant queueing control. Existing acceptance, drift,
failure-preservation, and stop rules apply unchanged. These are development outcomes only.
