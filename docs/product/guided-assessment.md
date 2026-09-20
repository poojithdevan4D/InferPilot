# Guided assessment v0.1

`inferpilot assess` is the first complete operator workflow. It turns one `ExperimentConfig` into a
bounded measurement and an understandable decision without introducing another benchmark engine.

## Dry run

```bash
inferpilot assess config.json \
  --dry-run \
  --max-wall-time-s 600 \
  --gpu-cost-per-hour 1.10 \
  --max-cost-usd 0.25
```

The dry run:

- validates the strict experiment schema, exact model revision, workload, and operator SLO;
- verifies that the current interpreter owns the pinned `vllm==0.29.0` executable;
- checks visible GPU count without loading a model;
- checks that the workload can collect aligned evidence (open loop, ≥100 requests, warm-up);
- prints the exact server command with its runtime-selected port represented explicitly;
- calculates `max wall time × GPU count` and the corresponding maximum cost;
- records model fit and candidate quality as unknown rather than guessing; and
- writes an immutable, self-validating `assessment-plan.json`.

It does not start vLLM, load weights, allocate VRAM, send requests, claim configuration fidelity, or
produce a runtime recommendation. A failed check still leaves the plan artifact and prevents GPU
execution.

## Execution

Removing `--dry-run` authorizes the baseline only. The command calls the existing
`run_experiment()` implementation, with a hard whole-experiment POSIX deadline in addition to the
existing readiness and request timeouts. The normal runner continues to own process lifecycle,
effective-configuration verification, workload dispatch, telemetry, aligned evidence, aggregation,
and cleanup.

The assessment reserves the runner's 15-second forced-cleanup allowance inside the operator's wall
ceiling. The active-run deadline is therefore `max_wall_time - cleanup_reserve`; even a child that
ignores graceful termination remains within the declared GPU-time bound.

The exact generated run directory is returned to the assessment layer. The immutable result and
phase/lifecycle artifacts are written before diagnosis begins.

## Outputs

Every invocation creates a unique assessment directory:

```text
<experiment>-assessment-<suffix>/
  assessment-plan.json
  run/<experiment>-<suffix>/   # real execution only
    result.json
    phases.json
    lifecycle.json
    telemetry.json
    ...
  evidence-card.json           # completed execution only
  experiment-plan.json         # supported candidate only
```

Operational `FAILED`, `OOM`, and `TIMEOUT` outcomes retain the runner bundle and produce no
measurement-based recommendation. A completed but ineligible or insufficient baseline produces an
abstaining Evidence Card. Baseline eligibility is an absolute prerequisite for any candidate.

## Candidate boundary

`experiment-plan.json` embeds the source Evidence Card and a complete candidate
`ExperimentConfig`. Its varied engine fields and acceptance criteria are recomputed on every load.
It always carries `execution_authorized=false`.

For the currently supported FP8-KV hypothesis, acceptance still requires representative task
quality, long-context accuracy, and registered latency/Pareto gates. Guided assessment never marks
those conditions passed and never launches the candidate.

## Cost meaning

The preflight number is a strict operator-owned ceiling:

```text
max_cost = max_wall_time × GPU_count × GPU_price_per_hour / 3600
```

It is not a throughput prediction. After a successful run, the Evidence Card separately reports
observed cost and an estimate for a fresh-server baseline/candidate pair. A configured
`--max-cost-usd` below the preflight ceiling prevents execution.
