# M6 rate bands — held-out confirmation

Before measurement, this freezes fresh prompt seed 6002 and arrival seeds 93–95 for the development-
supported boundary actions: width 2 at 1.5/2.5 QPS and width 4 at 5.5/6.5 QPS. All workload, engine,
SLO (TTFT p95 ≤250 ms; TPOT p95 ≤8.5 ms), acceptance, offline-loading, retry, stopping, and ordering
rules are identical to M6 development. There are 12 cells. A band is enabled only if its action passes
both endpoints in all six held-out cells; otherwise that advisor regime remains exact-match only.
