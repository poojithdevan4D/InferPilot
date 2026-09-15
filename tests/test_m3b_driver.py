import importlib.util,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/m3b-crossover-development/run_study.py"
S=importlib.util.spec_from_file_location("m3bdriver",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_fresh_order_and_ids():
 assert M.EXECUTOR.order()[0]==(2,63,1)
 assert M.EXECUTOR.order()[-1]==(6,65,4)
 assert M.EXECUTOR.eid(2,63,1)=="m3b-dev-qps2-a63-seq1"
 assert M.EXECUTOR.OUT.name=="m3b-crossover-development"
def test_main_sets_offline_before_delegating(monkeypatch):
 monkeypatch.delenv("HF_HUB_OFFLINE",raising=False);monkeypatch.delenv("TRANSFORMERS_OFFLINE",raising=False)
 monkeypatch.setattr(M.EXECUTOR,"main",lambda:7)
 assert M.main()==7
 assert os.environ["HF_HUB_OFFLINE"]==os.environ["TRANSFORMERS_OFFLINE"]=="1"
