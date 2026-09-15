# M2D development shape supplement — preregistration

This development-only supplement fills the missing evidence after the prior study stopped on OS
dispatch jitter. It runs the four-candidate binding grid for decode seed 32 and burst seeds 32 and
33 (12 cells). Existing committed schema-0.5.0 configs are reused for seed 32; burst seed-33 configs
are generated before measurement.

The 10 ms p95 and 250 ms maximum drift gates remain unchanged. Each cell permits at most **two
attempts only when the sole rejection is `dispatch_drift_exceeded`**; all attempts are preserved in
the manifest, and any other rejection stops immediately. This retry rule is fixed before execution.
The first valid attempt is accepted. No SLO or winner is declared; this is not held-out evidence.
