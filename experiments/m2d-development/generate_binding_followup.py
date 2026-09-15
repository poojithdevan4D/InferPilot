"""Generate/check the preregistered binding-token-budget follow-up."""
import argparse, json
from pathlib import Path
from inferpilot import ExperimentConfig

HERE = Path(__file__).resolve().parent
BASE = HERE / "canary-low-token-budget.json"
SEEDS = (20, 21, 22)
CANDIDATES = ((1, 512), (1, 2048), (4, 512), (4, 2048))

def eid(seed, seq, tok): return f"m2d-bind-prefill-a{seed}-seq{seq}-tok{tok}"

def payload(seed, seq, tok):
    p=json.loads(BASE.read_text()); name=eid(seed,seq,tok)
    p.update(experiment_id=name,name=name,description="Preregistered M2D binding-token-budget development follow-up; no SLO or winner.",tags=["m2d","development","binding-token-budget","prefill"])
    p["engine"].update(max_num_seqs=seq,max_num_batched_tokens=tok,enable_chunked_prefill=True)
    p["workload"].update(name="m2d-bind-prefill",num_requests=128,arrival_seed=seed)
    return ExperimentConfig.model_validate(p).model_dump(mode="json")

def main():
    check=argparse.ArgumentParser(); check.add_argument("--check",action="store_true"); args=check.parse_args()
    bad=[]
    for seed in SEEDS:
        for seq,tok in CANDIDATES:
            path=HERE/f"{eid(seed,seq,tok)}.json"; expected=payload(seed,seq,tok)
            if args.check:
                if not path.exists() or json.loads(path.read_text()) != expected: bad.append(path.name)
            else: path.write_text(json.dumps(expected,indent=2)+"\n")
    if bad: raise SystemExit(f"binding follow-up config drift: {bad}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
