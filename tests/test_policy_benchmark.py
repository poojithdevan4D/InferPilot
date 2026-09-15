import pytest
from inferpilot.search.policy_benchmark import PolicyBenchmarkSpec,PolicyDefinition
def test_policy_definitions_fail_closed():
 with pytest.raises(ValueError):PolicyDefinition(name="x",kind="task_map")
 with pytest.raises(ValueError):PolicyDefinition(name="x",kind="static")
 with pytest.raises(ValueError):PolicyDefinition(name="x",kind="seeded_random")
def test_frozen_spec_roundtrip():
 s=PolicyBenchmarkSpec(benchmark_id="b",task_ids=["low","high"],candidate_count=4,policies=[PolicyDefinition(name="aware",kind="task_map",task_actions={"low":1,"high":3}),PolicyDefinition(name="static",kind="static",candidate_index=3),PolicyDefinition(name="random",kind="seeded_random",random_seeds=list(range(100)))])
 assert PolicyBenchmarkSpec.model_validate_json(s.model_dump_json())==s
