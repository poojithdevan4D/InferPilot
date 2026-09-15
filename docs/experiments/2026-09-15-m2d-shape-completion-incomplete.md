# M2D development shape completion — stopped incomplete

The preregistered runner stopped at cell 13 of 24. The first **12 cells were accepted**: all four
decode candidates for arrival seeds 30 and 31, and all four burst candidates for seed 30. No model,
server, fidelity, telemetry, lifecycle, or request failure occurred.

`m2d-bind-burst-a31-seq4-tok2048` completed successfully but was rejected because dispatch-drift
p95 was **12.143 ms**, above the preregistered 10 ms gate (maximum 30.534 ms). The raw bundle and
manifest record are preserved. It was not ingested, accepted, or silently retried.

This study is incomplete and cannot support a three-block conclusion. Previously accepted cells
remain valid development measurements, but no robust decision is drawn from them. A separately
preregistered supplemental collection may fill missing development blocks; it must identify this
failure and define retry semantics before execution.
