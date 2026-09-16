"""M9b frozen replacement: preflight, sealed order, and faithful-clone corpus."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from inferpilot import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
M9B = ROOT / "experiments/m9b-canary-prefix"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRIVER = _load(M9B / "run_study.py", "m9b_run")
GEN = _load(M9B / "generate_configs.py", "m9b_gen")


def test_preflight_blocks_wrong_environment() -> None:
    # Dev interpreter has no sibling vllm -> abort before any run/manifest record.
    # (The real bench study legitimately populates runs/m9b-canary-prefix, so we assert
    # the preflight writes NOTHING new rather than that the directory is absent.)
    manifest = ROOT / "runs/m9b-canary-prefix/manifest.jsonl"
    before = manifest.read_text() if manifest.exists() else None
    existed = (ROOT / "runs/m9b-canary-prefix").exists()
    with pytest.raises(SystemExit, match="venv-bench"):
        DRIVER.main()
    assert (ROOT / "runs/m9b-canary-prefix").exists() == existed
    assert (manifest.read_text() if manifest.exists() else None) == before


def test_sealed_order_and_ids() -> None:
    order = DRIVER.E.order()
    assert len(order) == 24
    assert order[0] == (2, 103, 1) and order[-1] == (6, 105, 4)
    assert DRIVER.eid(2, 103, 1) == "m9b-held-qps2-a103-seq1"
    assert DRIVER.E.eid is DRIVER.eid  # executor bound to the m9b id scheme


def test_configs_byte_stable_and_present() -> None:
    want = GEN.expected()
    assert len(want) == 24
    on_disk = {p.name: p.read_bytes() for p in M9B.glob("*.json")}
    assert on_disk == want  # deterministic + already generated


def test_configs_are_faithful_clones_of_m9() -> None:
    allowed = {"experiment_id", "name", "description", "tags"}

    def _norm(d):
        d = json.loads(json.dumps(d))
        for k in allowed:
            d.pop(k, None)
        d["workload"].pop("name", None)
        return d

    for m9 in (ROOT / "experiments/m9-canary-prefix").glob("m9-held-*.json"):
        m9b = M9B / m9.name.replace("m9-held", "m9b-held")
        assert _norm(json.loads(m9.read_text())) == _norm(json.loads(m9b.read_text())), m9.name


def test_configs_validate_and_scope() -> None:
    ids, seeds, widths, rates = set(), set(), set(), set()
    for p in M9B.glob("*.json"):
        cfg = ExperimentConfig.model_validate_json(p.read_text())
        assert cfg.schema_version == "0.5.0"
        assert cfg.workload.prompt_seed == 7003
        ids.add(cfg.experiment_id)
        seeds.add(cfg.workload.arrival_seed)
        widths.add(cfg.engine.max_num_seqs)
        rates.add(cfg.workload.request_rate_qps)
    assert len(ids) == 24
    assert seeds == {103, 104, 105} and widths == {1, 2, 3, 4} and rates == {2.0, 6.0}
