"""Generate/check the sealed M3 confirmatory corpus and policy spec."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.comparison.models import BlockedStudySpec
from inferpilot.search.policy_benchmark import PolicyBenchmarkSpec,PolicyDefinition
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/m3b-crossover-development/m3b-dev-qps2-a63-seq1.json"
RATES=(2,6);SEEDS=(73,74,75);WIDTHS=(1,2,3,4);PROMPT_SEED=4002
def eid(r,s,w):return f"m3-held-qps{r}-a{s}-seq{w}"
def payload(r,s,w):
 p=json.loads(BASE.read_text());name=eid(r,s,w)
 p.update(experiment_id=name,name=name,description="Sealed M3 confirmatory held-out cell.",tags=["m3","heldout","sealed","confirmatory","offline"])
 p["engine"]["max_num_seqs"]=w;p["workload"].update(name=f"m3-held-qps{r}",request_rate_qps=float(r),prompt_seed=PROMPT_SEED,arrival_seed=s)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def study(r):return BlockedStudySpec(study_id=f"m3-held-qps{r}",objective="tpot_p95_ms",slo={"ttft_p95_ms":250,"tpot_p95_ms":8.5},varied_engine_fields=["max_num_seqs"],blocks=[{"seed":s,"experiment_ids":[eid(r,s,w) for w in WIDTHS]} for s in SEEDS],min_runs=1)
def benchmark():return PolicyBenchmarkSpec(benchmark_id="m3-confirmatory-v1",task_ids=["qps2","qps6"],candidate_count=4,policies=[PolicyDefinition(name="workload-aware",kind="task_map",task_actions={"qps2":1,"qps6":3}),PolicyDefinition(name="static-width4",kind="static",candidate_index=3),PolicyDefinition(name="seeded-random-100",kind="seeded_random",random_seeds=list(range(100)))])
def expected():
 x={f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for w in WIDTHS for r in RATES}
 x.update({f"study-qps{r}.json":(study(r).model_dump_json(indent=2)+"\n").encode() for r in RATES});x["policy-benchmark.json"]=(benchmark().model_dump_json(indent=2)+"\n").encode();return x
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");z=a.parse_args();want=expected()
 if z.check:
  got={p.name:p.read_bytes() for p in HERE.glob("*.json")}
  if got!=want:raise SystemExit("M3 held-out drift")
 else:
  for n,b in want.items():(HERE/n).write_bytes(b)
 return 0
if __name__=="__main__":raise SystemExit(main())
