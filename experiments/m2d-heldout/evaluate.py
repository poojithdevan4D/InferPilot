"""Evaluate the complete sealed M2D held-out collection after collection closes."""

from __future__ import annotations

import json
from pathlib import Path

from inferpilot.comparison import evaluate_blocked_study
from inferpilot.comparison.models import BlockedStudyReport
from inferpilot.comparison.store import ResultStore
from inferpilot.corpus import HeldOutCorpusSpec


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs/m2d-heldout"


def main() -> int:
    corpus = HeldOutCorpusSpec.model_validate_json((HERE / "corpus.json").read_text())
    records = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()]
    accepted = [record for record in records if record.get("accepted")]
    if len(records) != 36 or len(accepted) != 36:
        raise ValueError("held-out evaluation requires exactly 36 accepted first-attempt records")
    if any(record["attempt"] != 1 or record.get("problems") for record in records):
        raise ValueError("held-out manifest contains a retry or invalid record")

    store = ResultStore(OUT / "store")
    all_runs = store.list_runs()
    if len(all_runs) != 36:
        raise ValueError(f"held-out store must contain exactly 36 verified bundles; got {len(all_runs)}")

    summaries = {}
    for task in corpus.tasks:
        if task.partition != "heldout":
            continue
        report = evaluate_blocked_study(
            [
                [store.list_runs(experiment_id) for experiment_id in block.experiment_ids]
                for block in task.study.blocks
            ],
            task.study,
        )
        path = OUT / f"{task.shape_id}-blocked-report.json"
        payload = report.model_dump_json(indent=2) + "\n"
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"refusing to overwrite changed report: {path}")
        path.write_text(payload)
        BlockedStudyReport.model_validate_json(path.read_text())
        summaries[task.shape_id] = {
            "status": report.status,
            "best_candidate_indices": report.best_candidate_indices,
            "recommended_engine_values": report.recommended_engine_values,
            "candidates": [
                {
                    "candidate_index": candidate.candidate_index,
                    "engine_values": candidate.engine_values,
                    "blocks_feasible": candidate.blocks_feasible,
                    "robust_feasible": candidate.robust_feasible,
                    "mean_objective_mean": candidate.mean_objective_mean,
                    "rank": candidate.rank,
                }
                for candidate in report.candidates
            ],
        }
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
