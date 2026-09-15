import pytest
from pydantic import ValidationError
from inferpilot.advisor import AdvisorDecision,AdvisorPolicy,AdvisorRequest,recommend
def policy():return AdvisorPolicy(policy_id="m3",model="q",revision="abc",gpu_name="g",prompt_tokens=128,output_tokens=32,arrival_pattern="poisson-v1",fixed_engine_fields={"max_num_batched_tokens":512},regimes=[{"request_rate_qps":2,"engine_overrides":{"max_num_seqs":2}},{"request_rate_qps":6,"engine_overrides":{"max_num_seqs":4}}],evidence_report_sha256="a"*64)
def request(**kw):
 x=dict(model="q",revision="abc",gpu_name="g",prompt_tokens=128,output_tokens=32,arrival_pattern="poisson-v1",request_rate_qps=2);x.update(kw);return AdvisorRequest(**x)
def test_exact_supported_recommendation():
 d=recommend(policy(),request());assert d.status=="recommended" and d.engine_overrides=={"max_num_batched_tokens":512,"max_num_seqs":2}
@pytest.mark.parametrize("change",[{"request_rate_qps":3},{"model":"other"},{"gpu_name":"other"},{"prompt_tokens":129},{"arrival_pattern":"batched-poisson-v1"}])
def test_mismatch_abstains(change):
 d=recommend(policy(),request(**change));assert d.status=="abstain" and d.engine_overrides is None and d.reasons
def test_tampered_decision_rejected():
 raw=recommend(policy(),request()).model_dump(mode="json");raw["engine_overrides"]["max_num_seqs"]=4
 with pytest.raises(ValidationError,match="inconsistent"):AdvisorDecision.model_validate(raw)
