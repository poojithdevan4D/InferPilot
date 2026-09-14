"""InferPilot data contracts (Milestone 1).

Validated, JSON-serializable schemas for the autonomous inference-optimization
loop. This package intentionally contains *only* the data contracts — no engine
management, profiling, search, or decision logic yet.
"""

from __future__ import annotations

from ._base import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaModel,
    VersionedSchemaModel,
)
from .config import EngineConfig, ExperimentConfig, SLO
from .environment import EnvironmentMetadata, HardwareInfo
from .measurements import RequestMeasurement
from .results import AggregateMetrics, ExperimentResult
from .status import ExperimentStatus, FailureRecord
from .workload import WorkloadSpec

__version__ = SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaModel",
    "VersionedSchemaModel",
    "ExperimentConfig",
    "EngineConfig",
    "SLO",
    "WorkloadSpec",
    "EnvironmentMetadata",
    "HardwareInfo",
    "RequestMeasurement",
    "AggregateMetrics",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureRecord",
]
