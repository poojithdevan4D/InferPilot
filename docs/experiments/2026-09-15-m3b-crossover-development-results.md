# M3b crossover development results

M3b completed 24/24 accepted first-attempt cells with maximum dispatch-drift p95 2.442 ms. No M3
evidence was reused. Under the frozen TTFT p95 ≤250 ms / TPOT p95 ≤8.5 ms policy:

| Rate | Width | TTFT p95 range (ms) | TPOT p95 range (ms) | Feasible blocks |
|---:|---:|---:|---:|---:|
| 2 | 1 | 310.40–568.40 | 6.337–6.774 | 0/3 |
| 2 | **2** | **42.50–159.44** | **7.239–7.314** | **3/3** |
| 2 | 3 | 34.99–41.67 | 7.389–7.605 | 3/3 |
| 2 | 4 | 34.17–37.81 | 7.333–7.641 | 3/3 |
| 6 | 1 | 8263.13–16250.88 | 6.353–6.777 | 0/3 |
| 6 | 2 | 347.26–729.54 | 7.291–7.627 | 0/3 |
| 6 | 3 | 127.51–256.59 | 7.636–8.190 | 2/3 |
| 6 | **4** | **41.45–135.73** | **7.907–8.029** | **3/3** |

The preregistered crossover is observed: width 2 is the unique TPOT-minimizing robust-feasible
choice at 2 QPS; width 4 is the only robust-feasible choice at 6 QPS. This is development evidence
for freezing a policy, not held-out proof of optimization benefit.
