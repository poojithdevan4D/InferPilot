"""Execute sealed M6 held-out boundaries with the tested offline executor."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/"experiments/m3-crossover-development/run_study.py";S=importlib.util.spec_from_file_location("m6held",P);assert S and S.loader
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)
E.HERE=HERE;E.OUT=ROOT/"runs/m6-rate-bands-heldout";E.MANIFEST=E.OUT/"manifest.jsonl";E.STORE=E.OUT/"store"
POINTS=((1.5,2),(2.5,2),(5.5,4),(6.5,4));SEEDS=(93,94,95)
def label(rate):return str(rate).replace(".","p")
def eid(rate,seed,width):return f"m6-held-qps{label(rate)}-a{seed}-seq{width}"
def order():return tuple((r,s,w) for s in SEEDS for r,w in POINTS)
E.eid=eid;E.order=order
def main():os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";return E.main()
if __name__=="__main__":raise SystemExit(main())
