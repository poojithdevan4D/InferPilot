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
