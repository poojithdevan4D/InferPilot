"""Deterministic fingerprints for exact cohorts and controlled comparisons."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from ..results import ExperimentResult


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _delete_path(payload: dict, path: str) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        value = current.get(part)
        if not isinstance(value, dict):
            return
        current = value
    current.pop(parts[-1], None)


# Environment reflections of an engine field: when the field is allowlisted as
# varied, ONLY these mapped paths are stripped (everything else in the
# environment/runtime stays compatibility-significant).
_REFLECTION_PATHS: dict[str, tuple[str, ...]] = {
    "sampler_backend": (
        "environment.effective_sampler_backend",
        "environment.runtime_overrides.VLLM_USE_FLASHINFER_SAMPLER",
    ),
}


def _identity_payload(
    result: ExperimentResult, varied_engine_fields: Iterable[str] = ()
) -> dict:
    """Return all conditions that must agree for a valid comparison.

    Human labels and provenance timestamps are excluded. For each explicitly
    varied engine field, its requested value, effective-config value, and only
    its mapped environment/runtime reflection paths are removed; every unrelated
    environment/runtime value remains part of the identity.
    """
    config = result.config.model_dump(mode="json")
    for key in ("experiment_id", "name", "description", "tags"):
        config.pop(key, None)

    # Backward compatibility: the 0.4.0 split-seed fields are stripped when they
    # are absent/None (legacy behavior), so a loaded 0.3.0 config yields exactly
    # its historical identity payload. When explicitly set they stay in the
    # payload and remain compatibility-significant.
    workload = config.get("workload")
    if isinstance(workload, dict):
        for key in ("prompt_seed", "arrival_seed"):
            if workload.get(key) is None:
                workload.pop(key, None)
        # Explicit 0.5.0 defaults describe the historical behavior. Remove them
        # so read-supported 0.3.0/0.4.0 identities remain byte-stable.
        if workload.get("arrival_pattern") == "poisson-v1":
            workload.pop("arrival_pattern", None)
        if workload.get("burst_size") is None:
            workload.pop("burst_size", None)

    environment = result.environment.model_dump(mode="json")
    environment.pop("captured_at", None)
    environment.pop("hostname", None)

    effective = (
        result.effective_config.model_dump(mode="json")
        if result.effective_config is not None
        else None
    )

    payload = {"config": config, "environment": environment, "effective": effective}
    for field in varied_engine_fields:
        _delete_path(payload, f"config.engine.{field}")
        if effective is not None:
            _delete_path(payload, f"effective.{field}")
        for reflection in _REFLECTION_PATHS.get(field, ()):
            _delete_path(payload, reflection)
    return payload


def exact_fingerprint(result: ExperimentResult) -> str:
    """Fingerprint an exact repeat, excluding labels and capture timestamps."""
    return _digest(_identity_payload(result))


def comparison_fingerprint(
    result: ExperimentResult, varied_engine_fields: Iterable[str]
) -> str:
    """Fingerprint comparison controls after removing allowlisted engine fields."""
    fields = tuple(sorted(set(varied_engine_fields)))
    if not fields:
        raise ValueError("at least one varied engine field is required")
    return _digest(_identity_payload(result, fields))


def cross_block_fingerprint(
    result: ExperimentResult, varied_engine_fields: Iterable[str]
) -> str:
    """Cross-block context: like ``comparison_fingerprint`` but ALSO removes the
    workload seed, so runs that differ only in workload seed (and allowlisted
    engine fields) share it. Does not weaken the other fingerprints — the within-
    block comparison fingerprint still keeps the seed significant."""
    fields = tuple(sorted(set(varied_engine_fields)))
    if not fields:
        raise ValueError("at least one varied engine field is required")
    payload = _identity_payload(result, fields)
    # A block varies the ARRIVAL randomness: strip the legacy conflated seed and
    # the 0.4.0 arrival_seed. The prompt seed is NOT stripped — prompt content
    # must stay fixed across a study's blocks.
    _delete_path(payload, "config.workload.seed")
    _delete_path(payload, "config.workload.arrival_seed")
    return _digest(payload)
