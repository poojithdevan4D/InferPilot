"""Run the sealed M9 corpus with the established outcome-blind executor."""
from __future__ import annotations
import importlib.util,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/"experiments/m3-crossover-development/run_study.py";S=importlib.util.spec_from_file_location("m9exec",P);assert S and S.loader
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)
E.HERE=HERE;E.OUT=ROOT/"runs/m9-canary-prefix";E.MANIFEST=E.OUT/"manifest.jsonl";E.STORE=E.OUT/"store";E.SEEDS=(103,104,105)
def eid(rate,seed,width):return f"m9-held-qps{rate}-a{seed}-seq{width}"
E.eid=eid
def main():
 if not (Path(sys.executable).parent/"vllm").is_file():raise SystemExit("run with the benchmark environment: .venv-bench/bin/python")
 os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";return E.main()
if __name__=="__main__":raise SystemExit(main())
