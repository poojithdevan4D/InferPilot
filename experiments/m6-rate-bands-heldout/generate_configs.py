"""Generate/check fresh M6 held-out boundary-confirmation cells."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/m6-rate-bands-development/m6-dev-qps1p5-a83-seq2.json"
POINTS=((1.5,2),(2.5,2),(5.5,4),(6.5,4));SEEDS=(93,94,95);PROMPT_SEED=6002
def label(rate):return str(rate).replace(".","p")
def eid(rate,seed,width):return f"m6-held-qps{label(rate)}-a{seed}-seq{width}"
def payload(rate,seed,width):
 p=json.loads(BASE.read_text());name=eid(rate,seed,width)
 p.update(experiment_id=name,name=name,description="Sealed M6 held-out confirmation of a frozen advisor rate-band action.",tags=["m6","heldout","sealed","rate-band","offline"])
 p["engine"]["max_num_seqs"]=width;p["workload"].update(name=f"m6-held-rate-band-qps{label(rate)}",request_rate_qps=rate,prompt_seed=PROMPT_SEED,arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def expected():return {f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for r,w in POINTS}
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");x=a.parse_args();want=expected()
 if x.check:
  got={p.name:p.read_bytes() for p in HERE.glob("*.json")}
  if got!=want:raise SystemExit("M6 held-out config drift")
 else:
  for n,b in want.items():(HERE/n).write_bytes(b)
 return 0
if __name__=="__main__":raise SystemExit(main())
