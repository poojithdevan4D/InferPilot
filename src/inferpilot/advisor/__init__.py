"""Evidence-bound, fail-closed configuration recommendations."""
from .models import AdvisorDecision,AdvisorPolicy,AdvisorRequest,RateRegime
from .recommend import load_verified_policy,recommend
from .rate_band_evidence import BoundaryCell,RateBandEvidenceReport,ValidatedRateBand
from .profile_adapter import ProfileAdvisorDecision,ProfileContext,advise_from_profile
from .controller import ControllerEvent,ControllerReplay,ControllerSpec,ControllerState,ControllerTransition,advance_controller,replay_controller
from .canary import CanaryEvaluation,CanarySpec,evaluate_canary
from .config_comparison import ComparisonSpec,ConfigComparison,compare_configs
from .optimization_gate import OptimizationPlan,plan_from_diagnosis,plan_optimization
from .session import AdvisorSession,SessionStep,run_session
__all__=["AdvisorDecision","AdvisorPolicy","AdvisorRequest","RateRegime","BoundaryCell","ValidatedRateBand","RateBandEvidenceReport","load_verified_policy","recommend","ProfileAdvisorDecision","ProfileContext","advise_from_profile","ControllerEvent","ControllerReplay","ControllerSpec","ControllerState","ControllerTransition","advance_controller","replay_controller","CanaryEvaluation","CanarySpec","evaluate_canary","ComparisonSpec","ConfigComparison","compare_configs","OptimizationPlan","plan_from_diagnosis","plan_optimization","AdvisorSession","SessionStep","run_session"]
