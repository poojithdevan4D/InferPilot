# M3 confirmatory held-out preregistration

Fresh held-out prompt seed 4002 and arrival seeds 73–75 test the frozen development result without
reusing M3/M3b evidence. Tasks are 2 and 6 QPS; candidates are widths 1–4 with the identical fixed
engine/workload context, SLO (TTFT p95 ≤250 ms, TPOT p95 ≤8.5 ms), and TPOT objective.

Frozen one-shot policies are: workload-aware `{2 QPS: width2, 6 QPS: width4}`; static `width4`; and
seeded-random first actions over deterministic seeds 0–99. Primary scores are SLO-success rate,
oracle-hit rate, and mean simple TPOT regret among feasible selections. Workload-aware must exceed
static oracle-hit rate and must not reduce SLO-success rate to support the optimization claim.

There are 24 cells. Fixed order is seed 73→75, width 1→4, and within each width rate 2 then 6.
Offline loading is mandatory. Acceptance requires COMPLETED, baseline-eligible, verified config,
complete telemetry/lifecycle/phases/arrivals, 256/256 successes, and dispatch-drift p95 ≤10 ms. One
identical retry is permitted only for sole dispatch drift; otherwise stop fail-closed. No outcome may
alter seeds, policy, scoring, order, or stopping.
