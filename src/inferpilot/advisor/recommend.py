from __future__ import annotations
import hashlib
from pathlib import Path
from inferpilot.search.policy_benchmark import PolicyBenchmarkReport
from .models import AdvisorDecision,AdvisorPolicy,AdvisorRequest
def load_verified_policy(policy_path:Path,evidence_path:Path)->AdvisorPolicy:
 policy=AdvisorPolicy.model_validate_json(policy_path.read_text());payload=evidence_path.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=policy.evidence_report_sha256:raise ValueError("evidence report digest mismatch")
 PolicyBenchmarkReport.model_validate_json(payload)
 return policy
def _derive(policy,request):
 reasons=[]
 for field in ("model","revision","gpu_name","prompt_tokens","output_tokens","arrival_pattern"):
  if getattr(request,field)!=getattr(policy,field):reasons.append(f"unsupported_{field}")
 matches=[r for r in policy.regimes if r.request_rate_qps==request.request_rate_qps]
 if not matches:reasons.append("unsupported_request_rate_qps")
 if reasons:return "abstain",None,reasons
 return "recommended",{**policy.fixed_engine_fields,**matches[0].engine_overrides},[]
def recommend(policy:AdvisorPolicy,request:AdvisorRequest)->AdvisorDecision:
 status,overrides,reasons=_derive(policy,request)
 return AdvisorDecision(policy=policy,request=request,status=status,engine_overrides=overrides,reasons=reasons)
