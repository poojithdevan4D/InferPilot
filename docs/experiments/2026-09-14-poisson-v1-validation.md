# `poisson-v1` real-host validation

## Purpose

Validate that InferPilot's open-loop dispatcher follows its generated arrival schedule on
the RTX 3050 host before using it for optimization studies. This run validates the load
generator; it does not select an engine configuration.

## Setup

- Config: `examples/experiment_poisson_qps2_seq4.json`
- Model/runtime: same pinned Qwen2.5-0.5B and vLLM 0.29.0 environment as the C4 study
- Engine: `max_num_seqs=4`, PyTorch sampler
- Workload: 4 sequential warm-ups, then 32 measured requests using `poisson-v1`
- Distribution parameter: 2 requests/s, seed 0
- Generated final-arrival offset: 19.591 s

## Result

- Status: `COMPLETED`, baseline-eligible, 32/32 successful requests
- Measured-window duration: 19.805 s
- Dispatch drift (actual minus scheduled):
  - p50: 1.149 ms
  - p95: 2.694 ms
  - maximum: 4.025 ms
- TTFT p50/p95: 22.43 / 32.66 ms
- TPOT p50/p95: 6.327 / 6.988 ms
- E2E p95: 241.84 ms
- Output throughput over the finite arrival-and-drain window: 51.70 tokens/s
- Telemetry: 78 samples, no error; mean GPU utilization 28.0%; peak KV-cache usage
  0.176% of available cache
- Lifecycle: no pre-teardown errors

The target rate is a Poisson distribution parameter, not the realized rate of one finite
schedule. With only 32 requests, seed 0 produced a longer-than-mean 19.591-second arrival
span, so the run's realized request rate and output throughput must not be interpreted as a
2-QPS steady-state capacity measurement.

The observed dispatch error is small relative to the measured tens-of-milliseconds TTFT and
hundreds-of-milliseconds E2E latency. `poisson-v1` is therefore suitable for the next bounded
study on this host. A rate/capacity study should use more requests and reuse the same seeded
schedule across candidate engine configurations (common random numbers).

## Reproducibility

- Result-store bundle ID:
  `75b0f421d10190b0ed3488a91f274650e0887b283d0a96f98b59048306a55866`
- Local raw bundle: `runs/exp-poisson-qps2-seq4-c0d7d7ed`
