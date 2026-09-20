# FP8 instrumentation pilot v2 — results

**Preregistered verdict: `NOT_TARGET_REGIME`.** All six cells completed and passed every acceptance
gate on their first attempt. The evidence is promising, but it does not authorize the larger held-out
campaign because the registered target-regime and GO conditions were not all satisfied.

## Integrity and execution

- Frozen v2 protocol/code commit: `2af8a10`
- Modal app: `ap-kiqOwnafMFkIjm77f8olZv`
- Pinned image: `ghcr.io/poojithdevan4d/vllm-inferpilot@sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1`
- Six distinct cells, exact registered order, one attempt each, 100/100 successful requests each
- Every bundle independently reloads and passes the frozen acceptance validator
- Stored decision report reproduces byte-for-value from raw results and mechanism evidence
- Report content digest: `8a8c24f94c53b9ddb9b95fc0c1dbb3267afbfa240edb6d6093aadd969d6013fd`
- Downloaded archive SHA-256: `a5930ad0fb391312da009b265b072d2df26e2cbdbacc4ef9261ac5c49c7420f1`
- Estimated cumulative A10G cost, including the canary and stopped v1: **$0.8084**

## Paired results

| Block | BF16 load | BF16 KV peak | BF16 recompute burden | FP8/BF16 throughput | Recompute reduction |
|---:|---|---:|---:|---:|---:|
| 1 | overloaded | 99.96% | 3.64% | 1.360× | 100% |
| 2 | near_capacity | 99.96% | 0.94% | 1.118× | 100% |
| 3 | overloaded | 99.99% | 3.78% | 1.472× | 100% |

The geometric-mean throughput ratio was **1.308×**. Exact recomputed-token executions changed from
29,826/7,709/30,954 under BF16 KV to zero in every paired FP8 cell. Registered p95 TTFT, TPOT, and
end-to-end latency all improved under FP8 in every block; all cells retained 100% request success.

## Why this is not GO

The rule required every BF16 cell to be classified `overloaded`, every BF16 recompute burden to be at
least 10%, and every paired throughput ratio to be at least 1.15. Block 2 was only `near_capacity`,
all three recompute burdens were below 10%, and block 2's throughput ratio was 1.118. KV occupancy
alone does not repair those failures. Per the frozen precedence rule, the result is
`NOT_TARGET_REGIME`, not GO.

The directional result supports a new independent study; it does not validate a causal law,
cross-model generality, quality safety, deployment readiness, or maximum SLO capacity. V2 was also
explicitly adapted after v1's timeout failure, so a future claim requires a newly preregistered
held-out design rather than threshold or workload changes applied to these cells.
