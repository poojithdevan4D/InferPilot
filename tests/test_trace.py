"""Real-trace ingestion: characterize an operator's actual traffic, self-validating."""
from __future__ import annotations
import json
import pytest
from pydantic import ValidationError
from inferpilot import TraceRequest, WorkloadTrace, summarize_trace, load_trace_jsonl


def _reqs(n=100, rate=5.0, prompt=1000, output=200):
    return [TraceRequest(arrival_s=i/rate, prompt_tokens=prompt+(i%50), output_tokens=output+(i%30)) for i in range(n)]


def test_summary_and_rate() -> None:
    t = summarize_trace(_reqs(100, rate=5.0))
    assert t.num_requests == 100
    assert abs(t.request_rate_qps - 5.0) < 0.1
    assert t.prompt_p95 >= t.prompt_p50 and t.prompt_max >= t.prompt_p95
    assert t.sizing_context_tokens == t.prompt_p95 + t.output_p95


def test_jsonl_with_alias_keys() -> None:
    lines = [json.dumps({"timestamp": i*0.2, "input_tokens": 800, "num_generated_tokens": 128}) for i in range(20)]
    t = load_trace_jsonl("\n".join(lines))
    assert t.num_requests == 20 and t.prompt_p50 == 800 and t.output_p50 == 128


def test_self_validating_tamper_rejected() -> None:
    t = summarize_trace(_reqs(50))
    assert WorkloadTrace.model_validate_json(t.model_dump_json()) == t
    raw = t.model_dump(mode="json"); raw["request_rate_qps"] = 999.0
    with pytest.raises(ValidationError, match="inconsistent with its requests"):
        WorkloadTrace.model_validate(raw)


def test_load_trace_v01_schema() -> None:
    import json
    from inferpilot import load_trace_v01
    rows=[json.dumps({"trace_version":"0.1","request_id":str(i),"arrival_offset_ms":i*200,
                      "input_tokens":1500,"output_tokens":200,"status":"ok",
                      "deadline_class":"interactive-500ms-40ms","ttft_ms":300,"e2e_ms":5000}) for i in range(30)]
    t=load_trace_v01("\n".join(rows))
    assert t.num_requests==30 and t.prompt_p50==1500 and t.output_p50==200
    assert abs(t.request_rate_qps-5.0)<0.1  # 200ms spacing = 5 qps


def test_load_trace_v01_rejects_wrong_version() -> None:
    import pytest
    from inferpilot import load_trace_v01
    with pytest.raises(ValueError, match="trace_version"):
        load_trace_v01('{"trace_version":"9.9","arrival_offset_ms":0,"input_tokens":1,"output_tokens":1}')
