# M9b canary-prefix replacement — preregistration

M9b is an unchanged replacement for M9, which terminated before server launch because the wrong
Python environment lacked a vLLM executable. No warm-up, request, metric, or outcome existed.

Every scientific choice remains exactly as committed for M9 in `512aae4`: the same model/GPU, 24
cells, prompt seed 7003, arrival seeds 103–105, rates, widths, 256-request full window, first-128
predictor, SLO, execution order, acceptance/retry/stopping rules, confusion metrics, discriminative
requirement, zero-false-pass requirement, and ≤10% false-fail requirement. Only experiment IDs and
output directory change from `m9` to `m9b`.

The execution driver adds a preflight requiring the selected Python environment to contain a sibling
`vllm` executable. This check occurs before the shared executor creates a run or manifest record.
