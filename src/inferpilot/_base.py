"""Shared schema base for all InferPilot data contracts.

Every serializable object in InferPilot inherits from :class:`SchemaModel`, which
fixes a single, strict Pydantic configuration so that:

* unknown / misspelled fields are rejected instead of silently dropped
  (important for hand-written experiment configs), and
* JSON round-trips are lossless (``model_dump_json`` -> ``model_validate_json``
  yields an equal object).

``SCHEMA_VERSION`` is stamped onto the top-level objects so that stored results
remain interpretable if the contracts evolve.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "0.1.0"


class SchemaModel(BaseModel):
    """Base model with strict, deterministic serialization behaviour."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        # `EngineConfig.model` is a legitimate field name; disable the
        # `model_` protected-namespace warning rather than renaming it.
        protected_namespaces=(),
    )
