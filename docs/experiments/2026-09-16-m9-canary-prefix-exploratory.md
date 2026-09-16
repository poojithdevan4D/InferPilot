# Exploratory canary-prefix analysis

Across the existing M3 held-out grid (15 full passes, 9 full failures) and M6 rate-boundary evidence
(12 full passes), prefix classifications under the 250-ms TTFT / 8.5-ms TPOT SLO were:

| Prefix requests | True pass | True fail | False pass | False fail | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 32 | 27 | 6 | 3 | 0 | 91.7% |
| 64 | 27 | 7 | 2 | 0 | 94.4% |
| 128 | 27 | 9 | 0 | 0 | 100% |

The smaller prefixes missed slowly developing TTFT queueing. This analysis selected 128 requests for
M9, but makes no confirmatory claim because all source outcomes were already available during design.
