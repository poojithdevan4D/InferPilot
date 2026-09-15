# Evidence-bound advisor

InferPilot's first advisor is intentionally fail-closed. It recommends an engine override only when
model revision, GPU name, prompt/output lengths, arrival algorithm, and measured rate exactly match a
validated policy regime. It abstains on every unsupported input; it does not interpolate or claim
generality beyond the evidence.

```bash
python -m inferpilot.advisor policies/m3-rtx3050-qwen05b.json examples/advisor_request.json
```

The included policy returns width 2 at 2 QPS and width 4 at 6 QPS, with the fixed 512-token batching
budget. Its evidence digest binds the policy to the immutable M3 held-out policy report. This is a
research demonstrator, not an automatic production deployment controller.
