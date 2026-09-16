# Finding: the fixed-length applicability guard is stricter than the policy's own evidence

## Trigger

The M10 demonstrator abstained on real bench traffic with reason `mixed_prompt_lengths`. The advisor's
applicability guard (`profile_adapter._decision_fields`) requires the observed prompt lengths to be a
single value (`len(prompts) == 1`); the real workload has prompts of **128 or 129** tokens.

## Measurements (no code changed)

1. **The split is deterministic, content-driven tokenizer variance — not sampling noise.** All 24 M9b
   cells share `prompt_seed 7003`, so prompt *content* is identical across cells. The set of 61
   requests (of 256) that tokenize to 129 is **identical across every cell** (Jaccard 61/61). Specific
   prompt texts tokenize to 129; the rest to 128. Every request nominally targets 128 tokens.

   Overall M9b: `{128: 4680, 129: 1464}` (23.8% at 129).

2. **The policy's OWN validation evidence spanned the same 128/129 mix.** The committed policy
   `m3-rtx3050-qwen05b-v1` declares `prompt_tokens = 128`, but the evidence behind it was measured on:
   - M3 development runs (`prompt_seed 3001`): `{128: 1211, 129: 581}` — 32.4% at 129.
   - M3 held-out runs (`prompt_seed 4002`): `{128: 4056, 129: 2088}` — 34.0% at 129.

   The policy was **never validated on pure-128 traffic.** Its `prompt_tokens=128` label is a nominal
   scalar over a distribution that always contained 129s.

## Conclusion

The exact-length guard rejects the very prompt-length distribution the policy was validated on.
Tolerating a 128/129 profile is **within-evidence**, not extrapolation. This is a genuine
over-strictness in the applicability contract, surfaced on real data — the fail-closed behavior is
"correct" against the literal `prompt_tokens=128` label but wrong against the evidence that label
summarizes.

This is safe to relax **only** in a way that stays anchored to evidence. It must not become "tolerate
±k for any k" — that would re-admit extrapolation. The relaxation must be bounded by what the policy's
evidence actually observed.

## Decision required (does not silently touch the protected guard)

Two principled directions; both keep fail-closed for genuinely out-of-envelope traffic:

- **A. Policy declares its validated prompt-length envelope.** Evolve `AdvisorPolicy` (new schema
  version, backward-compatible default) to carry the prompt-length distribution/band its evidence
  covered (e.g. `prompt_tokens_min/max` or an explicit validated set `{128, 129}`). The profiler
  abstains only if observed lengths fall outside that recorded envelope. Most faithful; touches the
  policy schema (done carefully, versioned, never regenerating frozen policies in place).

- **B. Profiler matches on nominal (requested) length.** Record the *requested* prompt length on each
  observation (what the workload asked for) alongside the tokenized length, and match applicability on
  nominal. Keeps the policy scalar, but requires the workload/observation schema to carry nominal
  length and assumes nominal is the right matching basis.

Recommendation: **A** — it makes the envelope an explicit, auditable property of the evidence rather
than trusting a re-derived nominal. But this is a contract change to a protected surface, so it is your
call before any code moves.

Until a direction is chosen, the guard stays exactly as-is (fail-closed). No code changed by this
finding.

## Resolution (2026-09-17, commit `2ed53ae`)

Direction **A** implemented. `AdvisorPolicy` gained optional `prompt_tokens_min`/`prompt_tokens_max`;
`recommend` and `profile_adapter` apply inside the declared band (all observed prompts must be in-band)
and abstain (`prompt_lengths_outside_validated_envelope`) outside it. Both `None` (default) preserves
the exact-length behavior byte-for-byte, so the frozen M3 policy is untouched and still fail-closes.
Demonstrated on the real recorded 128/129 mix (demo Part 1b): exact policy abstains, `[128,129]`
envelope recommends. 372 tests pass.

**Still open (needs the GPU campaign):** issuing an evidence-*backed* envelope on a real policy
requires the applicability report to record the validated prompt-length span, and the M3 applicability
sha is frozen. The mechanism is ready; a v-next policy with a measured envelope is a follow-on, best
folded into the next hardware campaign rather than regenerating frozen M3 evidence in place.
