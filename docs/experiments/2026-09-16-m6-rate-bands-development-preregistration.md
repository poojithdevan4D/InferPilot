# M6 development — advisor rate-band boundaries

M5 correctly abstains for almost every passive profile because realized rate rarely equals nominal
2.0 or 6.0 QPS. M6 tests narrow applicability boundaries without rounding or searching for a new
winner.

The frozen actions are width 2 for the low-rate regime and width 4 for the high-rate regime. Boundary
points are 1.5 and 2.5 QPS for width 2, and 5.5 and 6.5 QPS for width 4. Each point uses independent
arrival seeds 83–85, fixed prompt seed 5002, 256 measured + 4 warm-ups, 128/32 tokens, Poisson
arrivals, 512-token batching, and explicit chunked prefill: 12 cells total.

The preregistered acceptance policy is TTFT p95 ≤250 ms and TPOT p95 ≤8.5 ms in every block. A band
may advance only if its frozen action passes both endpoints in all six cells. Passing establishes an
evidence-supported safety/applicability interval, not optimality within the interval. Failure leaves
the corresponding advisor regime exact-match only.

Execution order is seed 83→85 and, within each seed, 1.5→2.5→5.5→6.5 QPS. Offline loading is
mandatory. Run validity gates, sole-dispatch-drift one-retry rule, evidence preservation, and
fail-closed stopping match M3. No endpoint, action, seed, or threshold may change after measurement.
