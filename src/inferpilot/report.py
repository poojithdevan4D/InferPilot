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


_ANSI = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m",
         "green": "\033[32m", "yellow": "\033[33m", "cyan": "\033[36m", "grey": "\033[90m"}


def _paint(text: str, *styles: str, color: bool) -> str:
    if not color or not styles:
        return text
    return "".join(_ANSI[s] for s in styles) + text + _ANSI["reset"]


def _bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


# verdict -> (glyph+headline, headline color, sub-line fallback, sub color)
_READING = {
    "kv_capacity_bound_preempting": ("⚡ KV-BOUND & PREEMPTING", "yellow", None, "green"),
    "near_capacity": ("◐ NEAR CAPACITY", "yellow", "fp8 is pre-emptive insurance, not a measured win", "grey"),
    "not_kv_bound": ("✗ COMPUTE / OTHER-BOUND", "red", "fp8 KV cache → won't help here", "grey"),
    "healthy_or_underutilized": ("✓ HEALTHY / UNDERUTILIZED", "green", "no KV lever is warranted", "grey"),
    "need_second_snapshot": ("… NEED A SECOND SNAPSHOT", "cyan", "capture /metrics again in 30–60s", "grey"),
    "insufficient_metrics": ("… INSUFFICIENT METRICS", "cyan", "a needed metric is missing", "grey"),
}


def render_live_reading(reading: LiveReading, *, color: bool = False) -> str:
    snap = reading.after or reading.before
    headline, hcolor, sub_fallback, scolor = _READING[reading.verdict]
    sub = sub_fallback
    if reading.recommended_lever != "none":
        sub = f"{reading.recommended_lever} → worth testing"

    out = [
        _paint("InferPilot · the vLLM doctor", "bold", color=color),
        _paint("─" * 46, "grey", color=color),
        _paint(headline, "bold", hcolor, color=color),
    ]
    if sub:
        out.append("   " + _paint(sub, scolor, color=color))
    out.append("")

    # Evidence: only what /metrics actually gives us.
    def row(label: str, body: str) -> str:
        return "   " + _paint(f"{label:<12}", "grey", color=color) + body

    if snap.kv_cache_usage_perc is not None:
        kv = snap.kv_cache_usage_perc
        kvcol = "red" if kv >= 0.95 else ("yellow" if kv >= 0.8 else "green")
        out.append(row("KV cache", _paint(_bar(kv), kvcol, color=color) + f"  {kv * 100:.0f}%"))
    if snap.num_requests_waiting is not None:
        w = snap.num_requests_waiting
        out.append(row("queue", f"{w:g} waiting"
                       + (_paint("  ← backing up", "yellow", color=color) if w > 0 else "")))
    if snap.num_requests_running is not None:
        out.append(row("running", f"{snap.num_requests_running:g} requests"))
    pre = _preempt_delta(reading)
    if pre is not None:
        body = (_paint(f"rising (+{pre:g})", "red", color=color) if pre > 0
                else _paint("none", "green", color=color))
        out.append(row("preemptions", body))

    out.append("")
    out.append(_paint("→ ", "bold", color=color) + reading.next_step)
    return "\n".join(out)


_SHORT = {
    "kv_capacity_bound_preempting": ("⚡ kv-bound + preempting", "yellow"),
    "near_capacity": ("◐ near capacity", "yellow"),
    "not_kv_bound": ("✗ compute/other-bound", "red"),
    "healthy_or_underutilized": ("✓ healthy", "green"),
    "need_second_snapshot": ("… warming up", "cyan"),
    "insufficient_metrics": ("… missing metrics", "cyan"),
}


def render_live_line(reading: LiveReading, *, stamp: str = "", color: bool = False) -> str:
    """A compact one-line status for `doctor --watch`."""
    snap = reading.after or reading.before
    label, lcolor = _SHORT[reading.verdict]
    parts = [_paint(stamp, "grey", color=color)] if stamp else []
    # pad the (uncolored) label to a fixed width so columns line up
    pad = " " * max(0, 24 - len(label))
    parts.append(_paint(label, lcolor, color=color) + pad)
    kv = snap.kv_cache_usage_perc
    parts.append(f"KV {kv * 100:>3.0f}%" if kv is not None else "KV   —")
    w = snap.num_requests_waiting
    parts.append(f"q{w:g}" if w is not None else "q—")
    d = _preempt_delta(reading)
    if d is not None:
        parts.append(_paint(f"preempt +{d:g}", "red", color=color) if d > 0 else "preempt 0")
    return "  ".join(parts)


def _preempt_delta(reading: LiveReading):
    b, a = reading.before, reading.after
    if a is None or b.num_preemptions_total is None or a.num_preemptions_total is None:
        return None
    return a.num_preemptions_total - b.num_preemptions_total


def _shift(impact: LeverImpact) -> str:
    b = impact.baseline.ceiling_qps
    c = impact.candidate.ceiling_qps
    return f"{b:g} -> {c:g} QPS"


def _first(reasons: list[str], fallback: str) -> str:
    return reasons[0].replace("_", " ") if reasons else fallback
