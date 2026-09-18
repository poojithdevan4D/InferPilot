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
from .workload_profile import (
    WorkloadObservation,
    WorkloadProfile,
    build_workload_profile,
)
from .telemetry import ResourceSample, ResourceTelemetry
from .measurements import RequestMeasurement
from .results import AggregateMetrics, ExperimentResult
from .deployment import (
    DeploymentOption,
    FitAnalysis,
    ModelFootprint,
    ScaleRecommendation,
    analyze_fit,
    fetch_footprint,
    footprint_from_hf_config,
    recommend_deployment,
    recommend_scale,
)
from .diagnosis import BottleneckDiagnosis, diagnose
from .quality import QualityGate, QualitySpec, evaluate_quality, greedy_token_agreement
from .saturation import SaturationReport, detect_saturation
from .status import ExperimentStatus, FailureRecord
from .workload import WorkloadSpec

__version__ = SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaModel",
    "VersionedSchemaModel",
    "SaturationReport",
    "detect_saturation",
    "BottleneckDiagnosis",
    "diagnose",
    "QualityGate",
    "QualitySpec",
    "evaluate_quality",
    "greedy_token_agreement",
    "ModelFootprint",
    "FitAnalysis",
    "analyze_fit",
    "DeploymentOption",
    "recommend_deployment",
    "ScaleRecommendation",
    "recommend_scale",
    "footprint_from_hf_config",
    "fetch_footprint",
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
    "WorkloadObservation",
    "WorkloadProfile",
    "build_workload_profile",
    "RequestMeasurement",
    "AggregateMetrics",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureRecord",
]
