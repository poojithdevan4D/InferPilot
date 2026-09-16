"""Execute the preregistered M6 boundary study with the tested offline executor."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/"experiments/m3-crossover-development/run_study.py";S=importlib.util.spec_from_file_location("m6exec",P);assert S and S.loader
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)
E.HERE=HERE;E.OUT=ROOT/"runs/m6-rate-bands-development";E.MANIFEST=E.OUT/"manifest.jsonl";E.STORE=E.OUT/"store"
POINTS=((1.5,2),(2.5,2),(5.5,4),(6.5,4));SEEDS=(83,84,85)
def label(rate):return str(rate).replace(".","p")
def experiment_id(rate,seed,width):return f"m6-dev-qps{label(rate)}-a{seed}-seq{width}"
def execution_order():return tuple((rate,seed,width) for seed in SEEDS for rate,width in POINTS)
E.eid=experiment_id;E.order=execution_order
def main():
 os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";return E.main()
if __name__=="__main__":raise SystemExit(main())
