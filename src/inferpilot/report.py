"""Plain-English renderings of the capacity analyses, for a terminal or a report.

These turn the self-validating structs into the sentences an inference engineer
actually reads: what the ceiling is, whether a lever is worth trying, and what it
did. Rendering only — every number comes from an already-validated object, so these
functions add no claims of their own.
"""

from __future__ import annotations

from .capacity_frontier import CapacityFrontier
from .inference_plan import DeploymentPlan
from .lever_forecast import LeverForecast
from .lever_impact import LeverImpact
from .metrics_snapshot import LiveReading


def _slo_str(frontier: CapacityFrontier) -> str:
    s = frontier.slo
    parts = []
    if s.ttft_p95_ms is not None:
        parts.append(f"TTFT p95 <= {s.ttft_p95_ms:g} ms")
    if s.tpot_p95_ms is not None:
        parts.append(f"TPOT p95 <= {s.tpot_p95_ms:g} ms")
    if s.e2e_p95_ms is not None:
        parts.append(f"E2E p95 <= {s.e2e_p95_ms:g} ms")
    return ", ".join(parts) if parts else "no latency SLO set (keep-up only)"


def render_frontier(frontier: CapacityFrontier) -> str:
    rates = ", ".join(f"{p.offered_qps:g}" for p in sorted(frontier.points, key=lambda p: p.offered_qps))
    head = f"SLO-capacity frontier  ·  SLO: {_slo_str(frontier)}  ·  measured at {rates} QPS"
    if frontier.verdict == "ceiling_bracketed":
        line = f"  Capacity ceiling: ~{frontier.ceiling_qps:g} QPS before the SLO breaks."
        if frontier.cost_per_million_output_tokens_usd is not None:
            line += f"\n  Cost at the ceiling: ~${frontier.cost_per_million_output_tokens_usd:g} / 1M output tokens (conservative)."
        return head + "\n" + line
    if frontier.verdict == "no_ceiling_observed":
        return head + (
            f"\n  No ceiling found — every measured rate met the SLO."
            f"\n  Next: measure higher, at ~{frontier.suggested_next_qps:g} QPS, to find the wall."
        )
    if frontier.verdict == "already_over_slo":
        return head + (
            f"\n  Even the lowest measured rate already breaks the SLO."
            f"\n  Next: measure lower, at ~{frontier.suggested_next_qps:g} QPS."
        )
    nxt = f"\n  Next: measure at ~{frontier.suggested_next_qps:g} QPS." if frontier.suggested_next_qps else ""
    return head + f"\n  Insufficient evidence to pin a ceiling ({'; '.join(frontier.reasons)})." + nxt


def render_forecast(forecast: LeverForecast) -> str:
    probe = forecast.suggested_probe_qps
    lv = forecast.lever
    if forecast.verdict == "worth_testing":
        return (f"Lever forecast — {lv}: WORTH TESTING.\n"
                f"  {_first(forecast.reasons, 'the ceiling is limited by a bottleneck this lever relieves')}.\n"
                f"  Run the same sweep with {lv} on (bracket ~{probe:g} QPS), then compare.")
    if forecast.verdict == "not_worth_testing":
        return (f"Lever forecast — {lv}: NOT WORTH TESTING.\n"
                f"  {_first(forecast.reasons, 'the ceiling is not limited by anything this lever addresses')}.\n"
                f"  Spend the sweep on the right lever, or on scaling, instead.")
    if forecast.verdict == "collect_evidence_first":
        return (f"Lever forecast — {lv}: COLLECT EVIDENCE FIRST.\n"
                f"  The ceiling is pinned, but the limiting run carries no bottleneck diagnosis.\n"
                f"  Attach aligned load evidence at ~{probe:g} QPS and re-diagnose before forecasting.")
    return (f"Lever forecast — {lv}: MEASURE THE CEILING FIRST.\n"
            f"  {'; '.join(forecast.reasons)}."
            + (f"\n  Next: measure at ~{probe:g} QPS." if probe else ""))


def render_impact(impact: LeverImpact) -> str:
    lv = impact.lever
    cost = ""
    if impact.cost_delta_pct is not None:
        direction = "down" if impact.cost_delta_pct < 0 else "up"
        cost = f"  Cost/token {direction} {abs(impact.cost_delta_pct) * 100:.0f}%."
    if impact.verdict == "lever_raises_ceiling":
        return (f"Measured impact — {lv}: RAISES capacity.\n"
                f"  Ceiling {_shift(impact)} (+{impact.ceiling_delta_pct * 100:.0f}%).{cost}")
    if impact.verdict == "lever_lowers_ceiling":
        return (f"Measured impact — {lv}: LOWERS capacity.\n"
                f"  Ceiling {_shift(impact)} ({impact.ceiling_delta_pct * 100:.0f}%).{cost}  Keep the incumbent.")
    if impact.verdict == "lever_neutral":
        return (f"Measured impact — {lv}: NEUTRAL (within {impact.rel_tolerance:g} tolerance).\n"
                f"  Ceiling essentially unchanged.{cost}  Not worth the added risk.")
    if impact.verdict == "lever_raises_beyond_measured":
        return (f"Measured impact — {lv}: RAISES capacity beyond the measured range.\n"
                f"  The candidate met the SLO through every tested rate — measure higher to quantify the new ceiling.")
    return (f"Measured impact — {lv}: INCONCLUSIVE.\n"
            f"  {'; '.join(impact.reasons)}.")


_PLAN_HEADLINE = {
    "meets_target_now": "ALREADY MEETS TARGET",
    "lever_then_confirm": "TRY A LEVER, THEN CONFIRM",
    "scale_out": "SCALE OUT",
    "reduce_cost_via_lever": "MEETS QPS — CUT COST WITH A LEVER",
    "over_budget_no_lever": "MEETS QPS — OVER BUDGET",
    "collect_evidence": "COLLECT EVIDENCE FIRST",
    "measure_first": "FIND THE CEILING FIRST",
}


def render_plan(plan: DeploymentPlan) -> str:
    budget = ""
    if plan.budget_per_million_output_tokens_usd is not None:
        budget = f", budget ${plan.budget_per_million_output_tokens_usd:g}/1M tokens"
    lines = [f"Plan to {plan.target_qps:g} QPS{budget}: {_PLAN_HEADLINE[plan.verdict]}"]
    for step in plan.steps:
        lines.append(f"  {step.order}. [{step.action}] {step.detail}")
        if step.confirm:
            lines.append(f"     -> {step.confirm}")
    return "\n".join(lines)


_READING_HEADLINE = {
    "kv_capacity_bound_preempting": "KV-BOUND & PREEMPTING — fp8 worth a canary",
    "near_capacity": "NEAR CAPACITY — KV full, not preempting yet",
    "not_kv_bound": "COMPUTE/OTHER-BOUND — fp8 won't help",
    "healthy_or_underutilized": "HEALTHY / UNDERUTILIZED",
    "need_second_snapshot": "NEED A SECOND SNAPSHOT",
    "insufficient_metrics": "INSUFFICIENT METRICS",
}


def render_live_reading(reading: LiveReading) -> str:
    snap = reading.after or reading.before
    facts = []
    if snap.kv_cache_usage_perc is not None:
        facts.append(f"KV {snap.kv_cache_usage_perc * 100:.0f}%")
    if snap.num_requests_waiting is not None:
        facts.append(f"waiting {snap.num_requests_waiting:g}")
    if snap.num_requests_running is not None:
        facts.append(f"running {snap.num_requests_running:g}")
    factstr = ("  (" + ", ".join(facts) + ")") if facts else ""
    return (f"Live /metrics screening: {_READING_HEADLINE[reading.verdict]}{factstr}\n"
            f"  {reading.next_step}"
            + (f"\n  Lever to test: {reading.recommended_lever}" if reading.recommended_lever != "none" else ""))


def _shift(impact: LeverImpact) -> str:
    b = impact.baseline.ceiling_qps
    c = impact.candidate.ceiling_qps
    return f"{b:g} -> {c:g} QPS"


def _first(reasons: list[str], fallback: str) -> str:
    return reasons[0].replace("_", " ") if reasons else fallback
