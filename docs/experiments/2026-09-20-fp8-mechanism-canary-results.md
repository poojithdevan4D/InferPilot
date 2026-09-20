# FP8 mechanism instrumentation canary — result

**Verdict: PASS.** The discarded real-server canary satisfies Gate 2 of the
[instrumentation checklist](2026-09-19-fp8-instrumentation-pilot-checklist.md). It validates the
measurement path; it is not an fp8 performance result and is excluded from the six registered pilot
cells.

## Frozen execution

- Launcher freeze: `bc65e4e`, followed only by Modal compatibility/persistence fixes through
  `478ae0f`; all landed before a completed canary outcome existed.
- Modal app: `inferpilot-fp8-semantic-canary-v1`
- Image: `ghcr.io/poojithdevan4d/vllm-inferpilot@sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1`
- GPU: one A10G; model: `Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1`
- vLLM: `0.29.0`; Python: `3.12.1`
- Probe-imported dependencies: aiohttp `3.14.3`, protobuf `6.33.6`, idna `3.19`
- Fixed controls verified in the server's effective configuration: async scheduling off, prefix
  caching off, speculative decoding absent, iteration-detail logging on, PyTorch sampler selected.
- A prior attached invocation was canceled by the local client before producing an outcome. It is
  not an analytical attempt. The completed attempt was detached and committed artifacts after each
  arm to Modal volume `inferpilot-fp8-canary-results/semantic-canary-attempt-1`.

## Results

| Arm | Requests | Coverage | Iterations | Preemptions | Recomputed token executions | Recompute burden |
|---|---:|---|---:|---:|---:|---:|
| low-pressure control | 4/4 | complete | 256 | 0 | 0 | 0.0000 |
| forced pressure | 48/48 | complete | 1,026 | 3 | 23,324 | 0.0612 |

Recompute burden is the registered exact counter delta divided by successful prompt plus output
tokens. The pressure arm scheduled 391,980 context-token and 12,237 generation-token executions;
23,324 recomputed executions are therefore a strict subset of scheduled work. Maximum observed
effective batch size was 16, below the explicit `max_num_seqs=48` bound.

Both arms were `COMPLETED`, had zero failed measured requests, passed the five-family metric
capability gate, had no pre-teardown lifecycle errors, and produced self-validating
`mechanism-evidence.json` reports. Independent local verification recomputed the telemetry digest and
both measured-window log-slice hashes from the downloaded raw files.

The pressure workload was intentionally pathological and its latency is not a performance claim
(TTFT p95 54.8 s). It exists only to prove that real preemption makes the exact patched counter move
while the control remains zero.

## Provenance and cost

- Downloaded archive SHA-256: `20d5fcb82110f4c14256aac6700ab57aac24d2d25b00ae111e9d9297a62371ca`
- Canary report SHA-256: `c595ab72d363bb3b7f60bb305cab67d98456d11a39a3b821399fdd19505a34a3`
- Summary SHA-256: `4e2ef660920df85ee7597b9c2298b188e217a1c3b259afa26b818ca4bf8b6885`
- Combined measured runner occupancy: 255.446 seconds.
- A10G compute estimate at the observed Modal rate of $1.10/hour: **$0.0781**, excluding small
  container-control overhead. This is far below the preregistered $12 stop.

## Decision

The instrumentation path is valid: the public capability gate, exact recomputation counter,
stock iteration details, window alignment, evidence binding, and positive/negative semantics all
worked on the real pinned server. Gates 0–2 now authorize execution of the six registered pilot
cells. They do not predict that the pilot's performance GO rule will pass.
