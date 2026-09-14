"""Deterministic open-loop arrival schedules.

``poisson-v1`` algorithm:

* Arrival offsets are seconds relative to the measured arrival origin (t0).
* The first measured request arrives at offset ``0.0``.
* Each subsequent inter-arrival time is drawn from an exponential distribution
  with mean ``1 / request_rate_qps`` (``random.Random(seed).expovariate(rate)``),
  and offsets are the running cumulative sum.
* The sequence is fully determined by ``(n, request_rate_qps, seed)`` — identical
  inputs yield identical schedules; different seeds yield different schedules.

Limitations (bounded MVP): the whole schedule is generated up front and every
request is dispatched at its scheduled offset regardless of earlier completions
(no back-pressure, no dropping, no maximum in-flight cap). This is appropriate
only for small, bounded measured workloads; it is not a load generator that
throttles to a sustainable rate.
"""

from __future__ import annotations

import random

POISSON_VERSION = "poisson-v1"


def generate_poisson_offsets(num_requests: int, request_rate_qps: float, seed: int) -> list[float]:
    """Return deterministic arrival offsets (seconds) for ``poisson-v1``."""
    if num_requests <= 0:
        return []
    if request_rate_qps <= 0:
        raise ValueError("request_rate_qps must be > 0 for open-loop arrivals")

    rng = random.Random(seed)
    offsets = [0.0]
    cumulative = 0.0
    for _ in range(num_requests - 1):
        cumulative += rng.expovariate(request_rate_qps)
        offsets.append(cumulative)
    return offsets
