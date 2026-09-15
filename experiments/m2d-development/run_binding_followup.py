"""Resume-safe execution of the 12-cell binding-token-budget follow-up."""
import json, sys
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment
from generate_binding_followup import CANDIDATES, SEEDS, eid
from run_study import ROOT, validate_run

HERE=Path(__file__).resolve().parent; OUT=ROOT/"runs/m2d-binding-followup"; MANIFEST=OUT/"manifest.jsonl"
ORDER=((20,CANDIDATES),(21,tuple(reversed(CANDIDATES))),(22,CANDIDATES[2:]+CANDIDATES[:2]))

def records(): return [] if not MANIFEST.exists() else [json.loads(x) for x in MANIFEST.read_text().splitlines()]
def main():
    OUT.mkdir(parents=True,exist_ok=True); store=ResultStore(OUT/"store"); accepted={r["experiment_id"] for r in records() if r.get("accepted")}
    with MANIFEST.open("a") as stream:
        for seed,order in ORDER:
            for seq,tok in order:
                name=eid(seed,seq,tok)
                if name in accepted: continue
                config=ExperimentConfig.model_validate_json((HERE/f"{name}.json").read_text()); before=set(OUT.glob(f"{name}-*"))
                print(f"RUN {name}",flush=True); result=run_experiment(config,str(OUT),ready_timeout_s=900,request_timeout_s=600)
                made=set(OUT.glob(f"{name}-*"))-before
                if len(made)!=1: raise RuntimeError(f"expected one run dir: {made}")
                rd=made.pop(); check=validate_run(result,rd); rec={"experiment_id":name,"arrival_seed":seed,"max_num_seqs":seq,"max_num_batched_tokens":tok,"run_dir":str(rd.relative_to(ROOT)),"status":result.status.value,"accepted":not check["problems"],**check}
                if not check["problems"]: rec["store_run_id"]=store.ingest(rd).run_id; accepted.add(name)
                stream.write(json.dumps(rec)+"\n"); stream.flush(); print(json.dumps(rec,indent=2),flush=True)
                if check["problems"]: return 1
    expected={eid(s,q,t) for s in SEEDS for q,t in CANDIDATES}
    if accepted!=expected: raise RuntimeError(f"missing {sorted(expected-accepted)}")
    print("M2D BINDING FOLLOW-UP COMPLETE: 12/12 accepted",flush=True); return 0
if __name__ == "__main__": raise SystemExit(main())
