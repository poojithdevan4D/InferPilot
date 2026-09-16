"""Generate/check the sealed M9b replacement corpus; design unchanged from M9."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;BASE=ROOT/"experiments/m9-canary-prefix/m9-held-qps2-a103-seq1.json"
RATES=(2,6);WIDTHS=(1,2,3,4);SEEDS=(103,104,105)
def eid(rate,seed,width):return f"m9b-held-qps{rate}-a{seed}-seq{width}"
def payload(rate,seed,width):
 p=json.loads(BASE.read_text());name=eid(rate,seed,width)
 p.update(experiment_id=name,name=name,description="Sealed M9b replacement cell; M9 design unchanged after pre-measurement environment failure.",tags=["m9b","heldout","sealed","confirmatory","canary-prefix"])
 p["engine"]["max_num_seqs"]=width;p["workload"].update(name=f"m9b-held-qps{rate}",request_rate_qps=float(rate),prompt_seed=7003,arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def expected():return {f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for w in WIDTHS for r in RATES}
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");args=a.parse_args();want=expected()
 if args.check:
  if {p.name:p.read_bytes() for p in HERE.glob("*.json")}!=want:raise SystemExit("M9b sealed config drift")
 else:
  for name,data in want.items():(HERE/name).write_bytes(data)
 return 0
if __name__=="__main__":raise SystemExit(main())
