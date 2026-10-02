# Contributing to InferPilot

The most valuable contribution is a reproducible field report from a real vLLM deployment. InferPilot
is intentionally conservative: a negative result, an abstention, or a contradicted hypothesis is useful
evidence when the configuration and acceptance conditions are preserved.

## Report a real deployment result

Use the **vLLM field report** issue form. Before posting:

1. Remove prompts, generated text, hostnames, tokens, and customer identifiers.
2. Include the InferPilot and vLLM versions, model revision, GPU, command, and workload shape.
3. State whether the result came from `doctor`, a synthetic demo, or a controlled benchmark.
4. Preserve failures and abstentions; do not rerun only the cells with disappointing outcomes.

Never upload model weights, private prompts, API keys, raw customer traffic, or proprietary datasets.

## Change code

```bash
uv sync --extra dev --locked
uv run --extra dev pytest -q
uv build
```

Keep changes fail-closed and backward-compatible unless a schema migration is explicit. New persisted
conclusions need load-time validation and negative tamper tests. Real performance claims need a frozen
protocol, controlled comparison, and clearly stated scope.

Small, reviewable pull requests are preferred. Explain the user-visible problem, the contract change,
the tests, and any remaining uncertainty.
