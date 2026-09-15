from __future__ import annotations
from .models import AdvisorDecision,AdvisorPolicy,AdvisorRequest
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
