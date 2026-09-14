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

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "0.2.0"

# Versions this codebase knows how to interpret. Loading an object stamped with
# any other version must fail loudly rather than being silently treated as the
# current schema (a mismatched contract can invalidate stored comparisons).
#
# 0.2.0 changed the result STRUCTURE and SEMANTICS materially vs 0.1.0:
#   * the single ambiguous `HardwareInfo.cuda_version` was split into distinct
#     driver / torch / nvcc / flashinfer provenance fields, and
#   * `EngineConfig.sampler_backend` + applied runtime env overrides are recorded.
# There is intentionally NO in-place migration: 0.1.0 artifacts are refused
# loudly (a 0.1.0 result cannot be reinterpreted as 0.2.0 without inventing the
# missing provenance). Re-run to produce a 0.2.0 artifact, or write an explicit
# migration before re-adding 0.1.0 here.
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"0.2.0"})


class SchemaModel(BaseModel):
    """Base model with strict, deterministic serialization behaviour."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        # `EngineConfig.model` is a legitimate field name; disable the
        # `model_` protected-namespace warning rather than renaming it.
        protected_namespaces=(),
    )


class VersionedSchemaModel(SchemaModel):
    """Base for top-level, persisted objects that carry a schema version.

    The version is validated on every load: an unknown version raises instead of
    being interpreted as the current schema.
    """

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Contract version; must be one of SUPPORTED_SCHEMA_VERSIONS.",
    )

    @field_validator("schema_version")
    @classmethod
    def _check_supported_version(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {value!r}; this build supports "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}. Refusing to load rather than "
                f"reinterpret it as {SCHEMA_VERSION!r}."
            )
        return value
