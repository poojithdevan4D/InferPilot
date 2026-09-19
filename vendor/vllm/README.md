# Pinned vLLM instrumentation patch

The patch in `patches/` is the zero-GPU-reviewed mechanism instrumentation for the fp8 pilot.

- upstream: `vllm-project/vllm` tag `v0.29.0`
- upstream commit: `98dff2a81d747d1dba01a47f939f48c3526d4206`
- fork: <https://github.com/poojithdevan4D/vllm>
- fork commit: `c7bb7b7308ad174c75609052dd8fc13340dd921c`
- patch SHA-256: `981dc066e73bfa873d640e8f28eefa77fc42b9583d9b42fc3ff4db5a0de9e43a`
- patched image: `ghcr.io/poojithdevan4d/vllm-inferpilot@sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1`
- official amd64 base image: `vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`

The runtime diff adds one Prometheus counter,
`vllm:recomputed_token_executions_total`. The counter records scheduled token-position overlap below
the frontier discarded by preemption. Prefix-cache restoration is excluded because restored positions
are not scheduled.

The fork contains CPU-only tests for exact scheduler accounting and Prometheus exposition. The
focused validation at freeze was 9 passing tests; InferPilot's own suite was 484 passing tests and its
package build completed successfully.
