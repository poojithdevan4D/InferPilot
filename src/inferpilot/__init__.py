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
from .config import EngineConfig, ExperimentConfig, SamplerBackend, SLO
from .effective import EffectiveConfig
from .environment import EnvironmentMetadata, HardwareInfo, ToolchainInfo
from .phases import PhaseEvent, PhaseSpan, RunnerPhaseTiming
from .telemetry import ResourceSample, ResourceTelemetry
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
    "SamplerBackend",
    "SLO",
    "WorkloadSpec",
    "EnvironmentMetadata",
    "HardwareInfo",
    "ToolchainInfo",
    "EffectiveConfig",
    "ResourceSample",
    "ResourceTelemetry",
    "PhaseEvent",
    "PhaseSpan",
    "RunnerPhaseTiming",
    "RequestMeasurement",
    "AggregateMetrics",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureRecord",
]
