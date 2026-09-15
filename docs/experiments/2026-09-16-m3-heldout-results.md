# M3 confirmatory held-out results

All 24 sealed cells were accepted on their first attempt with no validity problem. The frozen
workload-aware policy chose width 2 at 2 QPS and width 4 at 6 QPS.

Held-out oracle decisions were width 2 at 2 QPS and width 3 at 6 QPS. The workload-aware width-4
action at 6 QPS missed the exact oracle but remained robust-feasible in all blocks.

| Frozen policy | SLO success | Oracle hit | Mean feasible TPOT regret |
|---|---:|---:|---:|
| workload-aware | **100% (2/2)** | **50% (1/2)** | **0.137 ms** |
| static width 4 | 100% (2/2) | 0% (0/2) | 0.219 ms |
| seeded-random, seeds 0–99 | 69.5% (139/200) | 23.5% (47/200) | 0.136 ms among feasible actions |

The preregistered primary condition is satisfied: workload-aware exceeds static oracle-hit rate
without reducing SLO success. It also reduces mean simple regret versus static by 37.8%. This is the
first held-out evidence that InferPilot's workload-conditioned choice improves over the fixed static
baseline in the declared two-regime, four-candidate experiment.

Scope is deliberately narrow: one model, one laptop GPU, two synthetic rate regimes, three arrival
blocks, and one observation per cell. Random's regret is conditional on feasible selections and must
not be read without its much lower SLO-success rate. No significance, production-SLO, switching-cost,
deployment-safety, or other-hardware/model claim is made.
