# InferPilot

**The vLLM doctor — screen live signals, then verify changes with controlled evidence.**

Point InferPilot at a running vLLM: it reads the live Prometheus `/metrics` and screens for KV
pressure, preemption, and queueing. It tells you whether a change like fp8 KV cache is worth a
controlled test, and abstains when the signals cannot support one. Open source, MIT, no telemetry.

## Try InferPilot in 60 seconds

Choose the path that matches what you have:

| You have… | Do this | What you get |
|---|---|---|
| Nothing installed | **[Open the interactive demo](https://poojithdevan4d.github.io/InferPilot/)** | See how live signals change the next experiment |
| A running vLLM server | `uvx inferpilot doctor --url http://localhost:8000` | A live, non-invasive screening result |
| No GPU, but want the full pipeline | `uvx inferpilot demo` | A synthetic Evidence Card through the real decision contracts |

No account, API key, or telemetry. `doctor` reads vLLM's Prometheus endpoint; it does not change the
server. The synthetic demo proves software behavior, not an empirical performance claim.

![InferPilot diagnosing a live vLLM](docs/launch/doctor.gif)

## One measured result—including the failed hypothesis

In a preregistered Qwen2.5-7B/A10G study, fp8 KV improved throughput by **33.6%** on the
KV-pressured workload and by only **0.2%** on the unpressured control. But fp8 did not reduce
recomputation as preregistered, so the study verdict was **INCONCLUSIVE**, not a claimed mechanism
win. InferPilot preserved the useful performance result and the contradicted explanation.

[Read the 12-cell result and its limitations →](docs/experiments/2026-09-22-fp8-mechanism-7b-repair-results.md)

That is the project boundary: screen cheaply, measure controlled candidates, and abstain rather
than turn a plausible story into a deployment recommendation.

## Diagnose your vLLM in one line

Already running vLLM? Point it at the server. No install, no benchmark run:

```bash
uvx inferpilot doctor --url http://localhost:8000
```

```text
InferPilot · the vLLM doctor
Live screening · confirm changes with a controlled benchmark
──────────────────────────────────────────────
⚡ KV PRESSURE + PREEMPTION SIGNAL
   kv_cache_dtype=fp8 → worth testing

   KV cache    ████████████████████  98%
   queue       8 waiting  ← backing up
   running     12 requests
   preemptions rising (+9)

→ The KV cache is full and the scheduler is preempting. This screening signal makes
  fp8 KV worth testing in a controlled canary; it does not predict the size or cause
  of any gain.
```

Or the honest verdict most tools won't give you:

```text
○ NO KV CAPACITY SIGNAL
   prioritize compute or scaling tests

   KV cache    ███████████░░░░░░░░░  55%
   queue       7 waiting  ← backing up
   preemptions none

→ Requests are queuing while the KV cache has headroom. This snapshot does not support
  KV capacity as the limiting signal, so prioritize a controlled scaling or compute-side test.
```

It scrapes `/metrics` twice and returns a screening state — `KV pressure + preemption`
(fp8 worth testing), `no KV capacity signal`, `near-capacity`, or `healthy` — and names the
missing metric when it cannot decide. These states choose the next experiment; they are not
validated bottleneck labels or performance predictions.
`--json` for scripts, `--plain` for no color.

Leave it running to watch for trouble — it prints a line per check and flags the moment
the regime changes:

```bash
inferpilot doctor --url http://localhost:8000 --watch
```

```text
18:00:05  … warming up             KV  98%  q7
18:00:06  ⚡ KV pressure + preemption  KV  98%  q7  preempt +9   ⚠ changed: warming up → preempting
18:00:16  ✓ healthy                KV  41%  q0  preempt 0     ⚠ changed: preempting → healthy
```

**Install:** `uvx inferpilot …` runs it with zero install. To keep it around:
`uv tool install inferpilot` or `pipx install inferpilot` (or plain `pip install inferpilot`).

Once you have a rate sweep, `inferpilot capacity` turns it into an SLO-capacity ceiling,
a `$/token`, and an action plan to a target QPS. It fails closed unless every run has the
same resolved configuration, environment, workload, and seeds; only offered QPS may differ.

## Go deeper: the full decision path (no GPU)

No GPU, model download, repository checkout, or API key is required:

```bash
uvx inferpilot demo
```

Expected output:

```text
InferPilot quickstart (synthetic metadata; no GPU)

Status: experiment_recommended
Diagnosis: kv_pressure (load=overloaded)
Evidence: aligned; transferability=test_before_use
Candidate: {'kv_cache_dtype': 'fp8'}
Estimated paired experiment: 120.0 GPU-s / $0.0333
...
The candidate is a test, not a deployment.
```

This example is explicitly synthetic. It exercises the same contracts, diagnosis, economics, and
Evidence Card used for a real run; it is not included in the empirical evidence corpus. To inspect
the machine-readable result: `uvx inferpilot demo --output evidence-card.json`.

## The pipeline

```text
benchmark bundle
  ├─ request timing and token counts
  ├─ resolved engine configuration
  ├─ GPU / KV / queue telemetry
  └─ aligned arrival and completion evidence
                  │
                  ▼
       conservative diagnosis
                  │
                  ▼
       operator SLO + GPU price
                  │
                  ▼
           Evidence Card
  ├─ decision and reasons
  ├─ evidence strength
  ├─ one candidate or abstention
  ├─ estimated experiment cost
  └─ acceptance and quality gates
```

The important design choice is the aligned evidence layer. A high GPU reading is not enough to call
a server compute-bound. InferPilot requires a conserved view of arrivals, completions, queued work,
and delivered tokens. If that view is incomplete, it abstains.

## Assess a configuration safely

The guided path validates the plan before spending GPU time:

```bash
.venv-bench/bin/inferpilot assess examples/assessment_config.json \
  --dry-run \
  --max-wall-time-s 600 \
  --gpu-cost-per-hour 1.10 \
  --max-cost-usd 0.25
```

The dry run never starts a server, loads model weights, or allocates GPU memory. It writes a
self-validating `assessment-plan.json`, prints the exact planned vLLM command, checks whether this
environment can execute it, and calculates a strict maximum from the operator's wall-time and price.
The ceiling reserves time for forced cleanup. Static metadata deliberately reports model fit and
candidate quality as unknown.

Run the same command without `--dry-run` to execute one bounded baseline. InferPilot reuses the
existing runner, persists its bundle before diagnosis, and then writes an Evidence Card. If the card
supports a candidate, it also writes `experiment-plan.json`; that plan is explicitly **not authorized
for automatic execution** and includes the still-unverified quality, long-context, and Pareto gates.
Replace the example's illustrative SLO and price with your own constraints.

See the [guided-assessment contract](docs/product/guided-assessment.md).

## Gate a measured candidate

After collecting an operator-reviewed baseline, candidate, and the preregistered quality evidence:

```bash
.venv-bench/bin/inferpilot bind-quality teacher-forced-kl evidence/kl-gate.json \
  runs/baseline runs/candidate \
  --corpus-id operator-eval-v1 \
  --corpus-sha256 <sha256> \
  --output evidence/teacher-forced-kl.json

.venv-bench/bin/inferpilot gate examples/configuration_gate_spec.json \
  runs/baseline runs/candidate \
  --quality-evidence evidence/teacher-forced-kl.json \
  --quality-evidence evidence/needle-retrieval.json \
  --output configuration-gate-report.json
```

The result is `PASS`, `FAIL`, or `ABSTAIN`. `PASS` requires compatible aligned-load evidence,
candidate Pareto improvement, every declared SLO, and quality evidence bound to the exact
experiment IDs, configuration fingerprints, and corpus. Precision-changing candidates require both
teacher-forced KL and long-context retrieval gates. The report always keeps deployment unauthorized. See the
[configuration-gate contract](docs/product/configuration-gate.md).

## Analyze an existing run

Given an InferPilot runner bundle:

```bash
uv run inferpilot analyze runs/<run-directory> \
  --ttft-p95-ms 500 \
  --tpot-p95-ms 40 \
  --min-throughput-tokens-per-s 150 \
  --gpu-cost-per-hour 1.10 \
  --target-qps 4 \
  --output evidence-card.json
```

The operator supplies the SLO and price; InferPilot does not invent business constraints. The card
is self-validating: changing a persisted conclusion without changing its supporting inputs makes it
fail to load.

The current executable recommendation is deliberately narrow. Aligned overload together with high
KV occupancy and observed preemptions may nominate an `fp8` KV-cache canary. Everything unfamiliar
or inconclusive remains an abstention. See [the product boundary](docs/product/offline-optimizer.md).

## Run a benchmark

Automated tests use a fake server. Real measurements require a GPU and a separate Python 3.12
environment containing the exactly pinned `vllm==0.29.0`; vLLM is intentionally not a package
dependency.

```bash
uv venv --python 3.12 .venv-bench
uv pip install --python .venv-bench "vllm==0.29.0"
uv pip install --python .venv-bench -e .

.venv-bench/bin/python -m inferpilot.runner \
  examples/example_experiment.json --output-dir runs
```

The runner starts vLLM without a shell, verifies the effective configuration, warms up, measures
requests with a monotonic clock, captures telemetry and evidence, stops the server on every path, and
writes an immutable run directory under `runs/`.

## What exists today

| Capability | Status |
|---|---|
| Reproducible vLLM benchmark runner | Working |
| Strict result, provenance, timing, and integrity contracts | Working |
| Aligned load assessment with explicit abstention | Working |
| Operator Evidence Card and costed next experiment | Working |
| GPU-free preflight and hard-bounded guided assessment | Working |
| Deterministic performance/SLO/quality acceptance gate | Working |
| Cohort comparison, SLO gates, Pareto and blocked studies | Working |
| FP8-KV quality and long-context gates | Working, limited evidence |
| Automatic production traffic capture | Not built |
| Deployment changes and rollback integration | Not built |
| General optimizer across arbitrary engines and knobs | Not claimed |

InferPilot is currently an offline decision-support tool, not an autonomous production control plane.

## Empirical result so far

Exploratory runs across a small matrix found roughly **40–53% higher throughput when the BF16
baseline was KV-full and preempting**, but only **1.7%** in a no-preemption case. The later
preregistered 7B repair study found **+33.6%** on its pressured workload and **+0.2%** on its control,
while contradicting the registered recomputation mechanism. Its verdict was therefore
`INCONCLUSIVE`.

That is a useful mechanism hypothesis, not a universal law. The quality preflight also found real
distributional shift, so InferPilot always requires task-quality and long-context checks before an
FP8 candidate can be accepted. Read the concise
[finding](docs/blog/when-does-fp8-kv-actually-help.md) and the
[latest preregistered result](docs/experiments/2026-09-22-fp8-mechanism-7b-repair-results.md).

## Repository map

Start with the paths you actually use:

- [`src/inferpilot/metrics_snapshot.py`](src/inferpilot/metrics_snapshot.py) — the `doctor` read-only screening from `/metrics`.
- [`src/inferpilot/capacity_frontier.py`](src/inferpilot/capacity_frontier.py) — the SLO-capacity ceiling from a rate sweep.
- [`src/inferpilot/inference_plan.py`](src/inferpilot/inference_plan.py) — the action plan to a target QPS.
- [`src/inferpilot/cli.py`](src/inferpilot/cli.py) — the operator-facing command line.

Deeper internals: [`saturation.py`](src/inferpilot/saturation.py) (conservative load assessment),
[`diagnosis.py`](src/inferpilot/diagnosis.py) (regime classifier), and
[`runner/load_evidence.py`](src/inferpilot/runner/load_evidence.py) (aligned evidence collection).
[`docs/README.md`](docs/README.md) opens a deeper technical or research reading path.

The many files under `docs/experiments/` are the audit trail, including invalid and negative studies.
They are evidence for reviewers, not required reading for first use.

## Verify the repository

```bash
uv run --extra dev pytest -q  # no GPU
uv build
```

The package supports Python 3.10–3.14. Dependencies are locked in `uv.lock`.

## Near-term direction

The next product milestone is quality-evidence collection that produces the gate's bound
teacher-forced-KL and long-context retrieval inputs without hand-authored adapters, followed by a
controlled canary handoff. Production integration comes only after this local safety boundary is
trustworthy and repeatedly useful.

InferPilot's rule is simple: **measure, bind the evidence, recommend one test, and abstain when the
claim is not supported.**

Found something surprising on a real vLLM deployment? Please use the
[field-report form](https://github.com/poojithdevan4D/InferPilot/issues/new?template=field-report.yml).
Negative results and abstentions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) before sharing
artifacts or deployment details.
