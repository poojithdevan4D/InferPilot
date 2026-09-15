"""Generate/check the frozen M2D held-out corpus and bind every config."""
import argparse,json,sys
from pathlib import Path
from inferpilot import ExperimentConfig
from inferpilot.corpus import HeldOutCorpusSpec,validate_corpus_configs
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;DEV=ROOT/"experiments/m2d-development"
CANDS=((1,512,"s1t512"),(1,2048,"s1t2048"),(4,512,"s4t512"),(4,2048,"s4t2048")); HELD_SEEDS=(50,51,52)
SHAPES={
 "prefill":{"base":"m2d-bind-prefill-a20","dev":(20,21,22),"prompt":2001,"slo":{"ttft_p95_ms":400,"tpot_p95_ms":20}},
 "decode":{"base":"m2d-bind-decode-a30","dev":(30,31,32),"prompt":2002,"slo":{"ttft_p95_ms":1500,"tpot_p95_ms":8}},
 "burst":{"base":"m2d-bind-burst-a30","dev":(30,32,33),"prompt":2003,"slo":{"ttft_p95_ms":1500,"tpot_p95_ms":8.5}},
}
def hid(shape,seed,seq,tok):return f"m2d-held-{shape}-a{seed}-seq{seq}-tok{tok}"
def did(shape,seed,seq,tok):return f"m2d-bind-{shape}-a{seed}-seq{seq}-tok{tok}"
def held_payload(shape,seed,seq,tok):
 base=DEV/f"{SHAPES[shape]['base']}-seq{seq}-tok{tok}.json";p=json.loads(base.read_text());name=hid(shape,seed,seq,tok)
 p.update(experiment_id=name,name=name,description="Frozen M2D held-out evaluation cell; outcome must remain unopened until policy freeze.",tags=["m2d","heldout","sealed",shape]);p["workload"].update(prompt_seed=SHAPES[shape]["prompt"],arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def block(shape,partition,seeds):return [{"seed":s,"experiment_ids":[(did if partition=="development" else hid)(shape,s,q,t) for q,t,_ in CANDS]} for s in seeds]
def corpus_raw():
 tasks=[]
 for shape,x in SHAPES.items():
  for part,seeds in (("development",x["dev"]),("heldout",HELD_SEEDS)):
   tasks.append({"task_id":f"{shape}-{part}","shape_id":shape,"partition":part,"study":{"study_id":f"m2d-{shape}-{part}","objective":"tpot_p95_ms","slo":x["slo"],"varied_engine_fields":["max_num_seqs","max_num_batched_tokens"],"blocks":block(shape,part,seeds)}})
 return {"corpus_id":"m2d-binding-grid-v1","varied_engine_fields":["max_num_seqs","max_num_batched_tokens"],"candidates":[{"candidate_id":n,"engine_values":{"max_num_seqs":q,"max_num_batched_tokens":t}} for q,t,n in CANDS],"tasks":tasks,"candidate_budgets":[1,2,4],"hypotheses":["A workload-aware policy outperforms a single static candidate under equal budgets","The binding token budget matters most for prefill-heavy traffic","No-feasible and budget-exhausted remain distinct outcomes"]}
def generate():
 configs={}
 for shape,x in SHAPES.items():
  for seed in x["dev"]:
   for q,t,_ in CANDS:
    p=DEV/f"{did(shape,seed,q,t)}.json";configs[did(shape,seed,q,t)]=ExperimentConfig.model_validate_json(p.read_text())
  for seed in HELD_SEEDS:
   for q,t,_ in CANDS:
    name=hid(shape,seed,q,t);payload=held_payload(shape,seed,q,t);path=HERE/f"{name}.json";path.write_text(json.dumps(payload,indent=2)+"\n");configs[name]=ExperimentConfig.model_validate(payload)
 spec=HeldOutCorpusSpec.model_validate(corpus_raw());validate_corpus_configs(spec,configs);(HERE/"corpus.json").write_text(spec.model_dump_json(indent=2)+"\n")
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");x=a.parse_args()
 if not x.check:generate();return 0
 before={p.name:p.read_bytes() for p in HERE.glob("*.json")};generate();after={p.name:p.read_bytes() for p in HERE.glob("*.json")}
 if before!=after:raise SystemExit("held-out corpus/config drift")
 return 0
if __name__=="__main__":raise SystemExit(main())
