# M2D binding-grid development evidence — completion

The supplemental study completed **12/12 accepted cells** with no retry required after resume. Combined
with accepted cells from the stopped predecessor, this supplies three complete development blocks for
decode (seeds 30/31/32) and burst (30/32/33). The rejected seed-31 burst attempt remains excluded.

| Shape | seq | tokens | TTFT p95 mean [range] ms | TPOT p95 mean [range] ms | Throughput tok/s |
|---|---:|---:|---:|---:|---:|
| decode | 1 | 512 | 68013.69 [66910.08, 68601.19] | 6.275 [6.265, 6.280] | 159.08 |
| decode | 1 | 2048 | 71079.32 [66941.94, 77193.42] | 6.758 [6.281, 7.699] | 156.08 |
| decode | 4 | 512 | 617.30 [185.01, 1202.66] | 7.050 [7.015, 7.082] | 247.31 |
| decode | 4 | 2048 | 621.58 [185.77, 1215.86] | 7.045 [7.005, 7.081] | 247.31 |
| burst | 1 | 512 | 4687.16 [3557.74, 6629.21] | 6.288 [6.280, 6.302] | 145.47 |
| burst | 1 | 2048 | 4722.03 [3573.41, 6693.38] | 6.279 [6.275, 6.283] | 145.41 |
| burst | 4 | 512 | 609.07 [267.62, 1081.20] | 7.801 [7.785, 7.822] | 173.07 |
| burst | 4 | 2048 | 611.59 [265.70, 1076.26] | 7.858 [7.801, 7.951] | 173.09 |

The token budget is strongly active for width-4 prefill (previous report), but negligible for
width-4 decode and small for burst. This is the desired workload-dependent second dimension. Width
remains the dominant queueing control. Development evidence now suffices to freeze a held-out corpus;
no significance, deployment, or generality claim is made.
