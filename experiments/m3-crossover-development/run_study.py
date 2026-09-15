"""Resume-safe executor for the preregistered M3 development study."""
from __future__ import annotations
import json,statistics,subprocess,sys
from pathlib import Path
from inferpilot import ExperimentConfig,ExperimentResult,RunnerPhaseTiming
from inferpilot.comparison.store import ResultStore
from inferpilot.runner.orchestrator import run_experiment

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OUT=ROOT/"runs/m3-crossover-development"; MANIFEST=OUT/"manifest.jsonl"; STORE=OUT/"store"
SEEDS=(60,61,62); WIDTHS=(1,2,3,4); RATES=(2,6); MAX_ATTEMPTS=2
def eid(rate,seed,width): return f"m3-dev-qps{rate}-a{seed}-seq{width}"
def order(): return tuple((rate,seed,width) for seed in SEEDS for width in WIDTHS for rate in RATES)
def percentile(xs,q):
 o=sorted(xs); return o[min(len(o)-1,int(q*(len(o)-1)))]
def validate(result:ExperimentResult,run_dir:Path,rate:int,seed:int)->dict:
 p=[]; a=result.aggregates
 if result.status.value!="completed":p.append(f"status={result.status.value}")
 if not result.is_baseline_eligible:p.append("not_baseline_eligible")
 if a is None or (a.num_requests,a.num_successful,a.num_failed)!=(256,256,0):p.append("request_counts_invalid")
 if result.effective_config is None or not result.effective_config.verified or result.effective_config.unverified_fields:p.append("effective_not_verified")
 if result.telemetry is None or result.telemetry.error is not None or result.telemetry.num_samples<1:p.append("telemetry_incomplete")
 lp=run_dir/"lifecycle.json"
 if not lp.is_file() or json.loads(lp.read_text()).get("pre_teardown"):p.append("lifecycle_invalid")
 pp=run_dir/"phases.json"
 if not pp.is_file():p.append("phases_missing")
 else:
  try:
   timing=RunnerPhaseTiming.model_validate_json(pp.read_text()); by={x.name:x for x in timing.phases}; required={"server_startup","measured_window","teardown","total_occupancy"}
   if not required.issubset(by) or not all(by[x].completed for x in required):p.append("phases_incomplete")
  except ValueError:p.append("phases_invalid")
 drift={}; ap=run_dir/"arrivals.json"
 if not ap.is_file():p.append("arrivals_missing")
 else:
  ar=json.loads(ap.read_text()); scheduled=ar.get("scheduled_offsets_s",[]); actual=ar.get("actual_dispatch_offsets_s",[])
  if len(scheduled)!=256 or len(actual)!=256:p.append("arrival_counts_invalid")
  else:
   ds=[(x-y)*1000 for y,x in zip(scheduled,actual)]; p95=percentile(ds,.95)
   drift={"drift_mean_ms":round(statistics.mean(ds),3),"drift_p95_ms":round(p95,3),"drift_max_ms":round(max(ds),3)}
   if ar.get("algorithm")!="poisson-v1" or ar.get("arrival_seed")!=seed or ar.get("request_rate_qps")!=float(rate):p.append("arrival_provenance_mismatch")
   if p95>10:p.append("dispatch_drift_exceeded")
 return {"problems":p,"drift":drift}
def retry(problems,attempt):return problems==["dispatch_drift_exceeded"] and attempt<MAX_ATTEMPTS
def records():
 if not MANIFEST.is_file():return []
 return [json.loads(x) for x in MANIFEST.read_text().splitlines()]
def check_order(rs):
 expected=order(); cursor=0; attempts={}
 for r in rs:
  if cursor>=len(expected) or r["experiment_id"]!=eid(*expected[cursor]):raise ValueError("manifest violates preregistered order")
  n=attempts.get(r["experiment_id"],0)+1
  if r["attempt"]!=n:raise ValueError("manifest attempt mismatch")
  attempts[r["experiment_id"]]=n; allowed=retry(r["problems"],n)
  if r["retry_allowed"]!=allowed:raise ValueError("manifest retry mismatch")
  if r["accepted"]!=(not r["problems"]):raise ValueError("manifest acceptance mismatch")
  if not allowed:cursor+=1
def main():
 subprocess.run([sys.executable,str(HERE/"generate_configs.py"),"--check"],cwd=ROOT,check=True)
 OUT.mkdir(parents=True,exist_ok=True); store=ResultStore(STORE); rs=records(); check_order(rs)
 for r in rs:
  if r["accepted"]:
   store.load(r["store_run_id"])
   run_dir=ROOT/r["run_dir"]
   if validate(ExperimentResult.model_validate_json((run_dir/"result.json").read_text()),run_dir,r["nominal_qps"],r["arrival_seed"])["problems"]:raise ValueError("accepted run no longer validates")
  elif not r["retry_allowed"]:
   print("M3 DEVELOPMENT INVALID; terminal evidence preserved",file=sys.stderr);return 1
 accepted={r["experiment_id"] for r in rs if r["accepted"]}; attempts={}
 for r in rs:attempts[r["experiment_id"]]=max(attempts.get(r["experiment_id"],0),r["attempt"])
 with MANIFEST.open("a") as manifest:
  for rate,seed,width in order():
   name=eid(rate,seed,width)
   if name in accepted:continue
   while name not in accepted:
    attempt=attempts.get(name,0)+1; config=ExperimentConfig.model_validate_json((HERE/f"{name}.json").read_text()); before=set(OUT.glob(f"{name}-*"))
    print(f"RUN {name} attempt={attempt}",flush=True); result=run_experiment(config,str(OUT),ready_timeout_s=900,request_timeout_s=600)
    made=set(OUT.glob(f"{name}-*"))-before
    if len(made)!=1:raise RuntimeError(f"expected one run directory; got {made}")
    run_dir=made.pop(); check=validate(result,run_dir,rate,seed); ok=not check["problems"]; again=retry(check["problems"],attempt)
    rec={"experiment_id":name,"nominal_qps":rate,"arrival_seed":seed,"max_num_seqs":width,"attempt":attempt,"run_dir":str(run_dir.relative_to(ROOT)),"accepted":ok,"retry_allowed":again,**check}
    if ok:
     ingested=store.ingest(run_dir);rec["store_run_id"]=ingested.run_id;accepted.add(name)
    attempts[name]=attempt;manifest.write(json.dumps(rec)+"\n");manifest.flush();print(json.dumps(rec,indent=2),flush=True)
    if not ok and not again:return 1
 if accepted!={eid(*x) for x in order()}:raise RuntimeError("study incomplete")
 print("M3 DEVELOPMENT COMPLETE: 24/24 accepted",flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
