"""Generate/check fresh M3b configs after the network-invalid M3 attempt."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.comparison.models import BlockedStudySpec
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/m3-crossover-development/m3-dev-qps2-a60-seq1.json"
RATES=(2,6);SEEDS=(63,64,65);WIDTHS=(1,2,3,4);PROMPT_SEED=3002
def eid(rate,seed,width):return f"m3b-dev-qps{rate}-a{seed}-seq{width}"
def payload(rate,seed,width):
 p=json.loads(BASE.read_text());name=eid(rate,seed,width)
 p.update(experiment_id=name,name=name,description="Fresh M3b crossover development cell; offline-only model loading.",tags=["m3b","development","crossover","offline","poisson-v1"])
 p["engine"]["max_num_seqs"]=width;p["workload"].update(name=f"m3b-crossover-qps{rate}",request_rate_qps=float(rate),prompt_seed=PROMPT_SEED,arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def study(rate):return BlockedStudySpec(study_id=f"m3b-dev-qps{rate}",objective="tpot_p95_ms",slo={"ttft_p95_ms":250,"tpot_p95_ms":8.5},varied_engine_fields=["max_num_seqs"],blocks=[{"seed":s,"experiment_ids":[eid(rate,s,w) for w in WIDTHS]} for s in SEEDS],min_runs=1)
def expected():
 x={f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for w in WIDTHS for r in RATES}
 x.update({f"study-qps{r}.json":(study(r).model_dump_json(indent=2)+"\n").encode() for r in RATES});return x
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");args=a.parse_args();want=expected()
 if args.check:
  got={p.name:p.read_bytes() for p in HERE.glob("*.json")}
  if got!=want:raise SystemExit("M3b config drift")
 else:
  for n,b in want.items():(HERE/n).write_bytes(b)
 return 0
if __name__=="__main__":raise SystemExit(main())
