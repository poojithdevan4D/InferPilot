"""Preregistered supplemental development collection with bounded jitter retry."""
import json
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment
from generate_shape_completion import CANDIDATES,eid
from run_study import ROOT,validate_run
HERE=Path(__file__).resolve().parent;OUT=ROOT/"runs/m2d-shape-supplement";MAN=OUT/"manifest.jsonl"
CELLS=[("decode",32,q,t) for q,t in CANDIDATES]+[("burst",s,q,t) for s in (32,33) for q,t in CANDIDATES]
def main():
 OUT.mkdir(parents=True,exist_ok=True);store=ResultStore(OUT/"store");rows=[] if not MAN.exists() else [json.loads(x) for x in MAN.read_text().splitlines()];accepted={x["experiment_id"] for x in rows if x.get("accepted")}
 with MAN.open("a") as stream:
  for shape,seed,seq,tok in CELLS:
   name=eid(shape,seed,seq,tok)
   if name in accepted:continue
   for attempt in (1,2):
    cfg=ExperimentConfig.model_validate_json((HERE/f"{name}.json").read_text());before=set(OUT.glob(f"{name}-*"));print("RUN",name,"attempt",attempt,flush=True);res=run_experiment(cfg,str(OUT),ready_timeout_s=900,request_timeout_s=600);rd=(set(OUT.glob(f"{name}-*"))-before).pop();check=validate_run(res,rd);rec={"experiment_id":name,"shape":shape,"arrival_seed":seed,"max_num_seqs":seq,"max_num_batched_tokens":tok,"attempt":attempt,"run_dir":str(rd.relative_to(ROOT)),"status":res.status.value,"accepted":not check["problems"],**check}
    if not check["problems"]:rec["store_run_id"]=store.ingest(rd).run_id;accepted.add(name)
    stream.write(json.dumps(rec)+"\n");stream.flush();print(json.dumps(rec),flush=True)
    if not check["problems"]:break
    if check["problems"] != ["dispatch_drift_exceeded"] or attempt==2:return 1
 expected={eid(h,s,q,t) for h,s,q,t in CELLS}
 if accepted!=expected:raise RuntimeError(sorted(expected-accepted))
 print("M2D SHAPE SUPPLEMENT: 12/12 accepted",flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
