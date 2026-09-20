# FP8 instrumentation pilot v1 — protocol stop

**Status: INVALID_STOP after cell 1/6.** No arm comparison was made and the remaining five cells
were not run. The preserved baseline cell completed orchestration but failed its preregistered clean
request-census gate: 55/100 requests succeeded and 45/100 failed.

All failures were consecutive requests 55–99 and had the identical client error
`timeout: ReadTimeout('')`. The last successful request's TTFT was 298.2 seconds, immediately below
the frozen 300-second request timeout; later requests timed out before their first token. The vLLM
server remained healthy, lifecycle errors were empty, dispatch-drift p95 was 2.32 ms, mechanism
coverage was complete, GPU utilization averaged 99.4%, KV occupancy peaked at 100%, and the aligned
load assessment was `overloaded`. This identifies an undersized client timeout under the intended
queueing regime, not a server crash or an FP8 outcome.

- Modal app: `ap-kaqg1VnpkKpkGVyQh75Qlx`
- Preserved bundle: `inferpilot-fp8-pilot-results/pilot-v1/fp8-instrumentation-pilot-b1-bf16-kv-cdc5e1a0`
- Exact recomputed token executions: 15,490; preemptions: 2
- Estimated cumulative A10G cost including the discarded semantic canary: **$0.2689**

The registered stopping rule was honored. This attempt is never eligible for an FP8 comparison.
