"""Resume-safe runner for decode+burst binding-grid development completion."""
import json
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment
from generate_shape_completion import CANDIDATES,SEEDS,SHAPE_NAMES,eid
from run_study import ROOT,validate_run
HERE=Path(__file__).resolve().parent;OUT=ROOT/"runs/m2d-shape-completion";MAN=OUT/"manifest.jsonl"
ORDER=tuple((seed,shape,(CANDIDATES if (seed+len(shape))%2 else tuple(reversed(CANDIDATES)))) for seed in SEEDS for shape in SHAPE_NAMES)
def main():
 OUT.mkdir(parents=True,exist_ok=True);store=ResultStore(OUT/"store");old=[] if not MAN.exists() else [json.loads(x) for x in MAN.read_text().splitlines()];accepted={x["experiment_id"] for x in old if x.get("accepted")}
 with MAN.open("a") as stream:
  for seed,shape,order in ORDER:
   for seq,tok in order:
    name=eid(shape,seed,seq,tok)
    if name in accepted:continue
    cfg=ExperimentConfig.model_validate_json((HERE/f"{name}.json").read_text());before=set(OUT.glob(f"{name}-*"));print("RUN",name,flush=True);res=run_experiment(cfg,str(OUT),ready_timeout_s=900,request_timeout_s=600);made=set(OUT.glob(f"{name}-*"))-before
    if len(made)!=1:raise RuntimeError(made)
    rd=made.pop();check=validate_run(res,rd);rec={"experiment_id":name,"shape":shape,"arrival_seed":seed,"max_num_seqs":seq,"max_num_batched_tokens":tok,"run_dir":str(rd.relative_to(ROOT)),"status":res.status.value,"accepted":not check["problems"],**check}
    if not check["problems"]:rec["store_run_id"]=store.ingest(rd).run_id;accepted.add(name)
    stream.write(json.dumps(rec)+"\n");stream.flush();print(json.dumps(rec),flush=True)
    if check["problems"]:return 1
 expected={eid(h,s,q,t) for h in SHAPE_NAMES for s in SEEDS for q,t in CANDIDATES}
 if accepted!=expected:raise RuntimeError(sorted(expected-accepted))
 print("M2D SHAPE COMPLETION: 24/24 accepted",flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
