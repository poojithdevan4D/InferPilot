"""Generate/check M6 development boundary-validation cells."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BASE=ROOT/"experiments/m3-heldout/m3-held-qps2-a73-seq2.json"
POINTS=((1.5,2),(2.5,2),(5.5,4),(6.5,4));SEEDS=(83,84,85);PROMPT_SEED=5002
def label(rate):return str(rate).replace(".","p")
def eid(rate,seed,width):return f"m6-dev-qps{label(rate)}-a{seed}-seq{width}"
def payload(rate,seed,width):
 p=json.loads(BASE.read_text());name=eid(rate,seed,width)
 p.update(experiment_id=name,name=name,description="M6 development boundary cell validating a frozen advisor action; not an optimizer search.",tags=["m6","development","rate-band","applicability","offline"])
 p["engine"]["max_num_seqs"]=width;p["workload"].update(name=f"m6-rate-band-qps{label(rate)}",request_rate_qps=rate,prompt_seed=PROMPT_SEED,arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def expected():return {f"{eid(r,s,w)}.json":(json.dumps(payload(r,s,w),indent=2)+"\n").encode() for s in SEEDS for r,w in POINTS}
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");x=a.parse_args();want=expected()
 if x.check:
  got={p.name:p.read_bytes() for p in HERE.glob("*.json")}
  if got!=want:raise SystemExit("M6 config drift")
 else:
  for n,b in want.items():(HERE/n).write_bytes(b)
 return 0
if __name__=="__main__":raise SystemExit(main())
