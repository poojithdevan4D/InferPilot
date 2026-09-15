import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/m3-heldout/run_study.py";S=importlib.util.spec_from_file_location("m3held",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_sealed_order():
 assert len(M.E.order())==24
 assert M.E.order()[0]==(2,73,1) and M.E.order()[-1]==(6,75,4)
 assert M.E.eid(2,73,1)=="m3-held-qps2-a73-seq1"
