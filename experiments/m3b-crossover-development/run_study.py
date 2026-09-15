"""Run fresh M3b by binding the tested M3 executor to new offline evidence."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/"experiments/m3-crossover-development/run_study.py"
SPEC=importlib.util.spec_from_file_location("m3_executor",SOURCE)
assert SPEC and SPEC.loader
EXECUTOR=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(EXECUTOR)
EXECUTOR.HERE=HERE;EXECUTOR.OUT=ROOT/"runs/m3b-crossover-development"
EXECUTOR.MANIFEST=EXECUTOR.OUT/"manifest.jsonl";EXECUTOR.STORE=EXECUTOR.OUT/"store"
EXECUTOR.SEEDS=(63,64,65)
def experiment_id(rate,seed,width):return f"m3b-dev-qps{rate}-a{seed}-seq{width}"
EXECUTOR.eid=experiment_id
def main():
 os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
 if os.environ.get("HF_HUB_OFFLINE")!="1" or os.environ.get("TRANSFORMERS_OFFLINE")!="1":raise RuntimeError("offline environment not fixed")
 return EXECUTOR.main()
if __name__=="__main__":raise SystemExit(main())
