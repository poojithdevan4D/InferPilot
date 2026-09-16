"""Evaluate the preregistered 128-request prefix predictor after M9 completes."""
from __future__ import annotations
import json
from pathlib import Path
from inferpilot.results import ExperimentResult
from inferpilot.runner.aggregate import compute_aggregates
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/"runs/m9-canary-prefix";PREFIX=128
def passes(agg):return agg.ttft_p95_ms<=250 and agg.tpot_p95_ms<=8.5
def main():
 records=[json.loads(line) for line in (OUT/"manifest.jsonl").read_text().splitlines()]
 accepted=[record for record in records if record["accepted"]]
 if len(accepted)!=24 or len({record["experiment_id"] for record in accepted})!=24:raise SystemExit("M9 incomplete")
 cells=[]
 for record in accepted:
  result=ExperimentResult.model_validate_json((OUT/"store/objects"/record["store_run_id"]/"result.json").read_text())
  measurements=sorted(result.measurements,key=lambda item:item.start_time_s)[:PREFIX]
  prefix=compute_aggregates(measurements,max(item.end_time_s for item in measurements))
  cells.append({"experiment_id":record["experiment_id"],"prefix_pass":passes(prefix),"full_pass":passes(result.aggregates),"prefix_ttft_p95_ms":prefix.ttft_p95_ms,"prefix_tpot_p95_ms":prefix.tpot_p95_ms,"full_ttft_p95_ms":result.aggregates.ttft_p95_ms,"full_tpot_p95_ms":result.aggregates.tpot_p95_ms})
 tp=sum(c["prefix_pass"] and c["full_pass"] for c in cells);tn=sum(not c["prefix_pass"] and not c["full_pass"] for c in cells);fp=sum(c["prefix_pass"] and not c["full_pass"] for c in cells);fn=sum(not c["prefix_pass"] and c["full_pass"] for c in cells)
 result={"prefix_requests":PREFIX,"slo":{"ttft_p95_ms":250,"tpot_p95_ms":8.5},"confusion":{"true_pass":tp,"true_fail":tn,"false_pass":fp,"false_fail":fn},"discriminative":tp>=3 and tn>=3,"acceptance_passed":fp==0 and tp>=3 and tn>=3 and fn/max(1,tp+fn)<=.10,"cells":cells}
 print(json.dumps(result,indent=2));return 0 if result["acceptance_passed"] else 1
if __name__=="__main__":raise SystemExit(main())
