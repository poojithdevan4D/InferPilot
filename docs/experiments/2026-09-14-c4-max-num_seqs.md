# C4 scheduling characterization: `max_num_seqs=1..4`

## Question

How does vLLM's active-sequence limit trade scheduling latency against decode speed and
throughput when four closed-loop clients issue identical fixed-length requests?

## Controlled setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct` at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`
- Hardware: NVIDIA RTX 3050 Laptop GPU, 4 GiB
- Runtime: Python 3.12, vLLM 0.29.0, PyTorch sampler
- Workload: 4 warm-up + 32 measured requests; approximately 128 input tokens and exactly
  32 output tokens; greedy, ignore EOS; closed-loop concurrency 4
- Varied field: `engine.max_num_seqs` only (`1`, `2`, `3`, `4`)
- Evidence: 3 baseline-eligible repetitions per setting, 32/32 successes in every run
- Comparison context fingerprint:
  `1a4318427b40b1a7358e470b6b42acc93c1d020fd0c52f64837661aed47d2a64`

## Results

Values are means over three run-level aggregates; parentheses contain run-to-run CV.

| `max_num_seqs` | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 648.97 (0.34%) | 681.57 (0.12%) | 6.619 (0.17%) | 881.34 (0.50%) | 150.32 (0.37%) |
| 2 | 268.07 (0.88%) | 287.14 (1.63%) | 7.527 (1.08%) | 516.33 (1.20%) | 258.81 (0.45%) |
| 3 | 47.61 (0.94%) | 271.83 (2.57%) | 7.752 (1.87%) | 509.49 (2.06%) | 355.63 (0.42%) |
| 4 | 49.71 (5.41%) | 61.30 (2.46%) | 8.071 (2.66%) | 283.12 (2.39%) | 466.56 (1.50%) |

Worst observed run-level values, used by InferPilot's conservative SLO gate:

| `max_num_seqs` | max TTFT p95 ms | max TPOT p95 ms | max E2E p95 ms | min output tok/s |
|---:|---:|---:|---:|---:|
| 1 | 682.29 | 6.630 | 884.63 | 149.68 |
| 2 | 290.58 | 7.616 | 521.30 | 257.49 |
| 3 | 279.02 | 7.887 | 520.61 | 353.99 |
| 4 | 62.38 | 8.250 | 288.06 | 462.30 |

## Interpretation

Throughput increases monotonically with admitted sequence count, from 150.32 to 466.56
output tok/s. TPOT p95 also worsens monotonically, from 6.619 to 8.071 ms, because more
sequences share decode capacity. This is a real latency/throughput trade-off, not a universal
optimization win.

The seq3 distribution demonstrates why median-only tuning is unsafe. Three of four clients
can begin promptly, so TTFT p50 is only 47.61 ms. The fourth waits behind an active
32-token generation, leaving TTFT p95 at 271.83 ms. At seq4 every client can be admitted;
TTFT p95 then falls to 61.30 ms. The sharp tail-latency knee is therefore aligned with the
closed-loop concurrency, as the scheduling model predicts.

With TTFT p95 (lower), TPOT p95 (lower), and throughput (higher) as explicit objectives,
all four settings are Pareto-nondominated. Selecting one requires a workload SLO. InferPilot
must not manufacture a scalar score or call seq4 the winner merely because it has the highest
throughput.

These are descriptive results from three repetitions, not confidence bounds or claims that
the same curve generalizes to another model, GPU, request-length distribution, arrival
process, or concurrency.

## Reproducibility references

The ignored local `result-store/` contains immutable, manifest-verified bundles with these
content IDs:

- seq1: `43ac6b11...`, `47a0ccd4...`, `f272be10...`
- seq2: `43dc336b...`, `94c81085...`, `986fbbbc...`
- seq3: `67984a0a...`, `af64ba5d...`, `cbd7d140...`
- seq4: `4c5b0bf1...`, `a1ca8803...`, `e6ab9e46...`

Local derived artifacts:

- `runs/frontier-v0.1-c4-max-num-seqs-1-4.json`
- `runs/comparison-v0.2-c4-seq2-vs-seq3.json`
- `runs/comparison-v0.2-c4-seq3-vs-seq4.json`
