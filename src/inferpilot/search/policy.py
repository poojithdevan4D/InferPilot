"""Outcome-blind candidate ordering for conventional search baselines."""

from __future__ import annotations

import hashlib

from .models import SearchPolicy


def candidate_order(policy: SearchPolicy, candidate_count: int, seed: int) -> list[int]:
    """Return a deterministic full order without accepting outcome data.

    ``seeded_random-v1`` assigns every candidate a SHA-256 priority derived only
    from the policy seed and candidate index. This is stable across Python
    versions and makes outcome leakage structurally impossible for this baseline.
    """
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if policy == "declared_order-v1":
        return list(range(candidate_count))
    if policy == "seeded_random-v1":
        def priority(index: int) -> bytes:
            return hashlib.sha256(f"seeded_random-v1:{seed}:{index}".encode()).digest()

        return sorted(range(candidate_count), key=priority)
    raise ValueError(f"unsupported search policy: {policy}")
