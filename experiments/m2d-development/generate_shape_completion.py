"""Generate/check decode+burst binding-grid development completion."""
import argparse,json
from pathlib import Path
from inferpilot import ExperimentConfig
from generate_configs import SHAPES
HERE=Path(__file__).resolve().parent; BASE=HERE/"canary-low-token-budget.json"
SEEDS=(30,31,32); SHAPE_NAMES=("decode","burst"); CANDIDATES=((1,512),(1,2048),(4,512),(4,2048))
def eid(shape,seed,seq,tok): return f"m2d-bind-{shape}-a{seed}-seq{seq}-tok{tok}"
def payload(shape,seed,seq,tok):
 p=json.loads(BASE.read_text()); s=SHAPES[shape]; name=eid(shape,seed,seq,tok)
 p.update(experiment_id=name,name=name,description="Preregistered M2D binding-grid development completion; no SLO or winner.",tags=["m2d","development","binding-token-budget",shape])
 p["engine"].update(max_num_seqs=seq,max_num_batched_tokens=tok,enable_chunked_prefill=True)
 p["workload"].update(name=f"m2d-bind-{shape}",num_requests=128,prompt_tokens=s["prompt_tokens"],output_tokens=s["output_tokens"],request_rate_qps=s["request_rate_qps"],arrival_pattern=s["arrival_pattern"],burst_size=s["burst_size"],prompt_seed=s["prompt_seed"],arrival_seed=seed)
 return ExperimentConfig.model_validate(p).model_dump(mode="json")
def main():
 a=argparse.ArgumentParser();a.add_argument("--check",action="store_true");x=a.parse_args();bad=[]
 for shape in SHAPE_NAMES:
  for seed in SEEDS:
   for seq,tok in CANDIDATES:
    path=HERE/f"{eid(shape,seed,seq,tok)}.json"; expected=payload(shape,seed,seq,tok)
    if x.check:
     if not path.exists() or json.loads(path.read_text())!=expected:bad.append(path.name)
    else:path.write_text(json.dumps(expected,indent=2)+"\n")
 if bad:raise SystemExit(f"shape completion drift: {bad}")
 return 0
if __name__=="__main__":raise SystemExit(main())
