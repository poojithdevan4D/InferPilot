import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/m6-rate-bands-development/run_study.py";S=importlib.util.spec_from_file_location("m6",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_boundary_order_is_exact():
 assert len(M.E.order())==12 and len(set(M.E.order()))==12
 assert M.E.order()[0]==(1.5,83,2) and M.E.order()[-1]==(6.5,85,4)
 assert M.E.eid(1.5,83,2)=="m6-dev-qps1p5-a83-seq2"
