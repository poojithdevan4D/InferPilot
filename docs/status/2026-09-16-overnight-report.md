# Overnight report — 2026-09-16

Ran as both planner and implementer while you slept. Everything below is committed and green.

## What landed

**M9b — canary-prefix confirmatory study (GPU, bench env).** Completed clean, exit 0, all 24
held-out cells accepted. A 128-request canary prefix reproduces the full 256-request SLO verdict on
**every cell: 14 pass / 10 fail, 0 false-pass, 0 false-fail** — exact agreement, acceptance PASSED.
Validates the controller's measured-canary gate. Report:
`docs/experiments/2026-09-16-m9b-canary-prefix-results.md` (`61d1278`). Raw evidence stays gitignored.

**M10 — AdvisorSession (GPU-free).** Wired the existing pieces (profile → advise → controller event
→ transition → replay) into one self-validating artifact that recomputes its whole chain on load, so
tampering anywhere is rejected. Applying a candidate still requires a passing measured canary; no live
mutation. 7 new tests (full loop, rollback, noisy-no-apply, round-trip, tamper). Commit `8e46058`.

**M10 demonstrator — on real artifacts.** `scripts/m10_advisor_session_demo.py` (`24bc5cd`) drives
the loop from the committed `m3-rtx3050-qwen05b` policy + real M9b evidence, showing two honest
behaviors:
- **Fail-closed** on the raw measured window — the advisor abstains (`mixed_prompt_lengths`).
- **Full apply loop** on policy-conformant rate-6 windows gated by a **real** qps6/seqs=4 canary that
  passes the SLO → `max_num_seqs 2 → 4`.

## Status
- `uv run --extra dev pytest`: **367 passed**.
- `uv build`: OK.
- Commits (newest first): `24bc5cd`, `8e46058`, `61d1278`. No amends/squashes; code and reports
  committed separately from (gitignored) raw evidence.

## One real finding worth your eye
The demonstrator surfaced a genuine gap: the raw bench workload's tokenized prompts are **128 or 129
tokens** (tokenizer variance), while the policy's applicability envelope is fixed-128. The advisor
therefore **fail-closes on its own validation workload**. That's the safe behavior, but it means the
policy would abstain in production on the very traffic it was tuned for. Next-milestone question:
should applicability tolerate a small prompt-length band (e.g. ±1 token / a tolerance window), or
should the profiler bucket near-equal lengths? Needs a decision + a preregistered check, not a silent
loosening of the fail-closed guard.

## The one decision I need from you
We are at the ceiling of the 4 GB RTX 3050: single small model (Qwen2.5-0.5B), two rates, fixed
128/32. Every result above is real but **envelope-specific**. To generalize (bigger models, wider
prompt/output mixes, higher rates, multi-model), we need more VRAM. Options:
1. **onebit RTX 5070 12 GB** — you own it, but it's the shared Windows box with a separate read-only
   project; I have not touched it per your constraint.
2. **Rented ≥24 GB GPU** — clean isolation, costs money (I will not spend autonomously).
3. **Stay on 3050** — keep hardening the loop/contracts on the current envelope (no new hardware).

Tell me which and I'll proceed.
