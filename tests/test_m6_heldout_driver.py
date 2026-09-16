import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/m6-rate-bands-heldout/run_study.py";S=importlib.util.spec_from_file_location("m6h",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_order():assert len(M.E.order())==12 and M.E.order()[0]==(1.5,93,2) and M.E.order()[-1]==(6.5,95,4)
