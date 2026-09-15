"""Produce immutable M3 blocked and frozen-policy reports after collection closes."""
import json
from pathlib import Path
from inferpilot.comparison import BlockedStudyReport,BlockedStudySpec,evaluate_blocked_study
from inferpilot.comparison.store import ResultStore
from inferpilot.search.policy_benchmark import PolicyBenchmarkSpec,evaluate_policy_benchmark
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/"runs/m3-heldout"
def write(path,payload):
 text=payload.model_dump_json(indent=2)+"\n"
 if path.exists():
  if json.loads(path.read_text())!=payload.model_dump(mode="json"):raise FileExistsError(path)
  return
 path.write_text(text)
def main():
 store=ResultStore(OUT/"store");reports={}
 if len(store.list_runs())!=24:raise ValueError("requires exactly 24 integrity-valid bundles")
 for rate in (2,6):
  spec=BlockedStudySpec.model_validate_json((HERE/f"study-qps{rate}.json").read_text())
  report=evaluate_blocked_study([[store.list_runs(e) for e in b.experiment_ids] for b in spec.blocks],spec);write(OUT/f"report-qps{rate}.json",report);reports[f"qps{rate}"]=report
 bench=PolicyBenchmarkSpec.model_validate_json((HERE/"policy-benchmark.json").read_text());report=evaluate_policy_benchmark(bench,reports);write(OUT/"policy-report.json",report);print(report.model_dump_json(indent=2))
if __name__=="__main__":main()
