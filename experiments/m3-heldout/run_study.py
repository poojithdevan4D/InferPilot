"""Execute the sealed M3 held-out corpus with the tested offline executor."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/"experiments/m3-crossover-development/run_study.py";S=importlib.util.spec_from_file_location("m3exec",P);assert S and S.loader
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)
E.HERE=HERE;E.OUT=ROOT/"runs/m3-heldout";E.MANIFEST=E.OUT/"manifest.jsonl";E.STORE=E.OUT/"store";E.SEEDS=(73,74,75)
def held_id(r,s,w):return f"m3-held-qps{r}-a{s}-seq{w}"
E.eid=held_id
def main():
 os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";return E.main()
if __name__=="__main__":raise SystemExit(main())
