"""Generate or verify the sealed M9 canary-prefix corpus."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/m3-heldout/m3-held-qps2-a73-seq1.json"
RATES=(2,6);WIDTHS=(1,2,3,4);SEEDS=(103,104,105);PROMPT_SEED=7003
def eid(rate,seed,width):return f"m9-held-qps{rate}-a{seed}-seq{width}"
def payload(rate,seed,width):
 p=json.loads(BASE.read_text());name=eid(rate,seed,width)
 p.update(experiment_id=name,name=name,description="Sealed M9 confirmatory prefix-prediction cell.",tags=["m9","heldout","sealed","confirmatory","canary-prefix"])
 p["engine"]["max_num_seqs"]=width
 p["workload"].update(name=f"m9-held-qps{rate}",request_rate_qps=float(rate),prompt_seed=PROMPT_SEED,arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def expected():return {f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for w in WIDTHS for r in RATES}
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");args=a.parse_args();want=expected()
 if args.check:
  got={p.name:p.read_bytes() for p in HERE.glob("*.json")}
  if got!=want:raise SystemExit("M9 sealed config drift")
 else:
  for name,payload_bytes in want.items():(HERE/name).write_bytes(payload_bytes)
 return 0
if __name__=="__main__":raise SystemExit(main())
