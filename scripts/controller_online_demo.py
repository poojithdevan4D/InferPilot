"""Controller 0.3.0 online-loop demo on REAL 7B campaign evidence.

Shows the live apply/rollback gate deciding from a measured ConfigComparison:
the current config is the vLLM default; a width-8 candidate was trialed; the
controller applies it only if it Pareto-dominates the default. On the real
2026-09-18 evidence it does not (better tpot, far worse ttft), so the loop keeps
the default.

    uv run python scripts/controller_online_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from inferpilot import ExperimentResult
from inferpilot.advisor import (
    ComparisonSpec,
    ControllerEvent,
    ControllerSpec,
    ControllerState,
    advance_controller,
    compare_configs,
)
from inferpilot.advisor.models import AdvisorPolicy

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "runs/gpu-campaign-7b"
DEFAULT_ID = "g7-default-qps6-a60"
CANDIDATE_ID = "g7-dev-qps6-a60-seq8"
# ControllerSpec requires a policy; the 0.3.0 canary_result path does not exercise it
# (it only gates on the measured comparison), so the committed m3 policy is a stand-in.
POLICY = ROOT / "policies/m3-rtx3050-qwen05b.json"


def _load(eid: str) -> ExperimentResult:
    rec = next(
        json.loads(l) for l in (STORE / "manifest.jsonl").read_text().splitlines()
        if json.loads(l).get("experiment_id") == eid and json.loads(l).get("accepted")
    )
    return ExperimentResult.model_validate_json(
        (STORE / "store/objects" / rec["store_run_id"] / "result.json").read_text()
    )


def main() -> int:
    if not (STORE / "manifest.jsonl").is_file():
        print("campaign store not found; run the GPU campaign first")
        return 2

    incumbent = _load(DEFAULT_ID)   # vLLM default (current serving config)
    candidate = _load(CANDIDATE_ID)  # width-8 trial
    cur = {"max_num_seqs": incumbent.config.engine.max_num_seqs}
    cand = {"max_num_seqs": candidate.config.engine.max_num_seqs}

    spec = ControllerSpec(
        controller_version="0.3.0",
        advisor_policy=AdvisorPolicy.model_validate_json(POLICY.read_text()),
        min_consecutive_windows=1,
        comparison_spec=ComparisonSpec(),
    )
    # State: default is live; width-8 has just been put forward for a measured canary.
    state = ControllerState(current_engine_overrides=cur, canary_candidate=cand)
    comparison = compare_configs(spec.comparison_spec, incumbent, candidate)

    transition = advance_controller(
        spec, state,
        ControllerEvent(event_version="0.3.0", event_type="canary_result",
                        config_comparison=comparison),
    )

    ia, ca = incumbent.aggregates, candidate.aggregates
    print(f"current (vLLM default) : ttft_p95={ia.ttft_p95_ms:8.1f}ms tpot_p95={ia.tpot_p95_ms:.2f}ms "
          f"thru={ia.throughput_requests_per_s:.2f}")
    print(f"candidate (max_num_seqs=8): ttft_p95={ca.ttft_p95_ms:8.1f}ms tpot_p95={ca.tpot_p95_ms:.2f}ms "
          f"thru={ca.throughput_requests_per_s:.2f}")
    print(f"\ncomparison verdict : {comparison.verdict}")
    print(f"reasons            : {comparison.reasons}")
    print(f"controller action  : {transition.action}")
    print(f"config after       : {transition.after.current_engine_overrides}  "
          f"({'switched' if transition.action == 'apply_candidate' else 'kept default'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
