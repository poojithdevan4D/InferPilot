"""M10 demonstrator: the autonomous advisor loop on REAL committed artifacts.

Uses:
  - the committed, evidence-bound policy  policies/m3-rtx3050-qwen05b.json
  - real stored bench measurements from   runs/m9b-canary-prefix/  (gitignored)

It prints TWO honest behaviors:

  Part 1 - fail-closed on raw evidence. A workload window built directly from the
  real measured arrivals ABSTAINS: the tokenized prompts are 128 or 129 tokens, so
  they do not match the policy's fixed-128 applicability envelope. The advisor
  refuses to act rather than extrapolate. This is the safety property, on real data.

  Part 2 - the full apply loop. Three policy-conformant rate-6 windows (uniform
  128/32 arrivals, the workload the policy was actually validated for) drive the
  controller to its stability threshold and call for a canary. The canary is a
  fully REAL measured qps6 / max_num_seqs=4 run; it passes the SLO, so the
  controller applies the candidate (max_num_seqs 2 -> 4).

Nothing here mutates a live server. Each AdvisorSession recomputes the whole chain
on construction, so the printed trace is self-validated, not asserted by hand.

    uv run python scripts/m10_advisor_session_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inferpilot import WorkloadObservation, build_workload_profile
from inferpilot.advisor import (
    AdvisorPolicy,
    ControllerSpec,
    ControllerState,
    ProfileContext,
    SessionStep,
    advise_from_profile,
    evaluate_canary,
    run_session,
)
from inferpilot.advisor.canary import CanarySpec
from inferpilot.results import ExperimentResult

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "policies/m3-rtx3050-qwen05b.json"
EVIDENCE = ROOT / "runs/m9b-canary-prefix"
# A passing rate-6, width-4 held-out cell = the real canary for the rate-6 candidate.
CANARY_CELL = "m9b-held-qps6-a105-seq4"
WINDOW = 40  # observations per synthetic passive window, drawn from the real arrivals


def _load_result(cell: str) -> ExperimentResult:
    records = [json.loads(l) for l in (EVIDENCE / "manifest.jsonl").read_text().splitlines()]
    rec = next(r for r in records if r["experiment_id"] == cell and r["accepted"])
    path = EVIDENCE / "store/objects" / rec["store_run_id"] / "result.json"
    return ExperimentResult.model_validate_json(path.read_text())


def _context(result: ExperimentResult) -> ProfileContext:
    return ProfileContext(
        model=result.config.engine.model,
        revision=result.config.engine.revision,
        gpu_name=result.environment.hardware.gpu_name,
        arrival_pattern="poisson-v1",
        declared_fixed_length=True,
    )


def _raw_window(result: ExperimentResult) -> SessionStep:
    """A passive profile from the real measured arrivals, exactly as recorded."""
    meas = sorted(result.measurements, key=lambda m: m.start_time_s)[:WINDOW]
    t0 = meas[0].start_time_s
    profile = build_workload_profile([
        WorkloadObservation(
            arrival_offset_s=m.start_time_s - t0,
            prompt_tokens=m.prompt_tokens,
            output_tokens=m.output_tokens,
            success=m.success,
        )
        for m in meas
    ])
    return SessionStep(kind="observation", profile=profile, context=_context(result))


def _conformant_window(result: ExperimentResult, rate: float = 6.0) -> SessionStep:
    """A rate-6 window matching the policy's validated 128/32 fixed-length workload."""
    profile = build_workload_profile([
        WorkloadObservation(
            arrival_offset_s=i / rate, prompt_tokens=128, output_tokens=32, success=True
        )
        for i in range(WINDOW)
    ])
    return SessionStep(kind="observation", profile=profile, context=_context(result))


def main() -> int:
    if not EVIDENCE.exists():
        print(f"evidence store not found: {EVIDENCE} (run the M9b bench study first)")
        return 2

    policy = AdvisorPolicy.model_validate_json(POLICY.read_text())
    result = _load_result(CANARY_CELL)

    spec = ControllerSpec(
        controller_version="0.2.0",
        advisor_policy=policy,
        min_consecutive_windows=3,
        cooldown_windows=2,
        canary_spec=CanarySpec(ttft_p95_limit_ms=250, tpot_p95_limit_ms=8.5),
    )
    initial = ControllerState(
        current_engine_overrides={
            "max_num_batched_tokens": 512, "enable_chunked_prefill": True, "max_num_seqs": 2,
        }
    )

    print(f"policy         : {policy.policy_id}  ({policy.model} @ {policy.gpu_name})")
    print(f"canary evidence: {CANARY_CELL}  ttft_p95={result.aggregates.ttft_p95_ms:.1f}ms "
          f"tpot_p95={result.aggregates.tpot_p95_ms:.2f}ms\n")

    # Part 1 - fail-closed on the raw measured arrivals.
    raw = _raw_window(result)
    raw_decision = advise_from_profile(policy, raw.profile, raw.context)
    raw_session = run_session(spec, initial, [raw])
    print("Part 1 - raw measured window under the exact-length policy (fail-closed):")
    print(f"  advisor: {raw_decision.status}  {raw_decision.reasons}")
    print(f"  action : {raw_session.replay.transitions[0].action}  "
          f"-> config unchanged {raw_session.final_state.current_engine_overrides}\n")

    # Part 1b - isolate the prompt-length axis: REAL recorded 128/129 prompt lengths,
    # arrival rate held at an in-band 6 qps (raw processing-start times are not a
    # faithful arrival trace). Exact policy abstains; the [128,129] validated-envelope
    # policy (the span the M3 evidence was collected on) recommends.
    real_lengths = [m.prompt_tokens for m in sorted(result.measurements, key=lambda m: m.start_time_s)][:WINDOW]
    mixed_profile = build_workload_profile([
        WorkloadObservation(arrival_offset_s=i / 6, prompt_tokens=real_lengths[i], output_tokens=32, success=True)
        for i in range(len(real_lengths))
    ])
    ctx = _context(result)
    envelope_policy = policy.model_copy(update={"prompt_tokens_min": 128, "prompt_tokens_max": 129})
    exact = advise_from_profile(policy, mixed_profile, ctx)
    env = advise_from_profile(envelope_policy, mixed_profile, ctx)
    print(f"Part 1b - real {sorted(set(real_lengths))}-token prompt mix at 6 qps:")
    print(f"  exact-length policy   : {exact.status}  {exact.reasons}")
    print(f"  [128,129] envelope    : {env.status}  -> candidate {env.engine_overrides}\n")

    # Part 2 - full apply loop on policy-conformant windows + the real canary.
    win = _conformant_window(result)
    decision = advise_from_profile(policy, win.profile, win.context)
    canary = evaluate_canary(spec.canary_spec, decision, result)
    steps = [win, win, win, SessionStep(kind="canary_result", canary_evaluation=canary)]
    session = run_session(spec, initial, steps)  # self-validates on construction

    print("Part 2 - policy-conformant windows + real measured canary:")
    print(f"  observed rate {win.profile.realized_request_rate_qps:.2f} qps "
          f"-> candidate {decision.engine_overrides}")
    print(f"  canary passed={canary.passed}")
    for i, t in enumerate(session.replay.transitions):
        print(f"  step {i} [{steps[i].kind:13}] -> {t.action:16} {t.reasons}")
    print(f"\n  final engine config: {session.final_state.current_engine_overrides}")
    print(f"  applied change     : max_num_seqs "
          f"{initial.current_engine_overrides['max_num_seqs']} -> "
          f"{session.final_state.current_engine_overrides['max_num_seqs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
