import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"experiments/m3-crossover-development/run_study.py"
S=importlib.util.spec_from_file_location("m3driver",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_order_and_retry_contract():
 assert len(M.order())==24 and len(set(M.order()))==24
 assert M.order()[:4]==((2,60,1),(6,60,1),(2,60,2),(6,60,2))
 assert M.retry(["dispatch_drift_exceeded"],1)
 assert not M.retry(["dispatch_drift_exceeded"],2)
 assert not M.retry(["telemetry_incomplete"],1)
def test_manifest_requires_immediate_retry():
 first=M.eid(2,60,1);second=M.eid(6,60,1)
 base={"accepted":False,"problems":["dispatch_drift_exceeded"],"retry_allowed":True}
 M.check_order([{**base,"experiment_id":first,"attempt":1},{"experiment_id":first,"attempt":2,"accepted":True,"problems":[],"retry_allowed":False},{"experiment_id":second,"attempt":1,"accepted":True,"problems":[],"retry_allowed":False}])
