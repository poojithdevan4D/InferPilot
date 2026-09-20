"""Evidence-first benchmarking and diagnosis for LLM inference serving.

InferPilot provides strict data contracts, a vLLM benchmark runner, aligned load
assessment, comparison/search replay, and fail-closed advisory primitives.  It is
an offline research library, not an autonomous production control plane.
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
from .mechanism import (
    MechanismCanaryReport,
    MechanismEvidence,
    MechanismLogSlice,
    SchedulerIteration,
    evaluate_mechanism_canary,
)
from .evidence_card import OptimizationEvidenceCard, build_evidence_card
from .assessment import (
    AssessmentBudget,
    AssessmentPlan,
    CandidateExperimentPlan,
    build_assessment_plan,
    build_budget,
    build_candidate_plan,
)
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
from .quality import (
    KLQualityGate, KLQualitySpec, NeedleQualityGate, NeedleQualitySpec,
    QualityGate, QualitySpec, evaluate_kl_quality, evaluate_needle_quality,
    evaluate_quality, greedy_token_agreement,
)
from .saturation import SaturationReport, detect_saturation
from .status import ExperimentStatus, FailureRecord
from .trace import TraceRequest, WorkloadTrace, load_trace_jsonl, load_trace_v01, summarize_trace
from .workload import WorkloadSpec

__version__ = SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaModel",
    "VersionedSchemaModel",
    "SaturationReport",
    "detect_saturation",
    "TraceRequest",
    "WorkloadTrace",
    "summarize_trace",
    "load_trace_jsonl",
    "load_trace_v01",
    "BottleneckDiagnosis",
    "diagnose",
    "QualityGate",
    "QualitySpec",
    "evaluate_quality",
    "greedy_token_agreement",
    "KLQualitySpec",
    "KLQualityGate",
    "evaluate_kl_quality",
    "NeedleQualitySpec",
    "NeedleQualityGate",
    "evaluate_needle_quality",
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
    "MechanismLogSlice",
    "SchedulerIteration",
    "MechanismEvidence",
    "MechanismCanaryReport",
    "evaluate_mechanism_canary",
    "OptimizationEvidenceCard",
    "build_evidence_card",
    "AssessmentBudget",
    "AssessmentPlan",
    "CandidateExperimentPlan",
    "build_assessment_plan",
    "build_budget",
    "build_candidate_plan",
    "AggregateMetrics",
    "ExperimentResult",
    "ExperimentStatus",
    "FailureRecord",
]
