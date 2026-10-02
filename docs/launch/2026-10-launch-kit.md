# InferPilot launch kit

Canonical links:

- Repository: https://github.com/poojithdevan4D/InferPilot
- Browser demo: https://poojithdevan4d.github.io/InferPilot/
- Technical case study: https://github.com/poojithdevan4D/InferPilot/blob/master/docs/blog/fp8-throughput-improved-hypothesis-failed.md
- Full study: https://github.com/poojithdevan4D/InferPilot/blob/master/docs/experiments/2026-09-22-fp8-mechanism-7b-repair-results.md
- Field report: https://github.com/poojithdevan4D/InferPilot/issues/new?template=field-report.yml

## Hacker News

**Title**

Show HN: InferPilot – vLLM screening that abstains when evidence is weak

**Text**

I built InferPilot after repeatedly finding that GPU utilization and a single benchmark number were
not enough to explain why a vLLM deployment was saturated.

It has a read-only `doctor` command that screens live Prometheus metrics, plus a stricter path for
controlled candidate comparisons, SLO checks, provenance, and explicit abstention.

The most useful result so far was also uncomfortable: FP8 KV improved throughput by 33.6% on a
preregistered 7B/A10G pressured workload and by 0.2% on the control, but it did not reduce
recomputation as preregistered. The tool kept the performance result and marked the mechanism study
INCONCLUSIVE.

No GPU is needed to try the software path: `uvx inferpilot demo`. If you run vLLM, the read-only path
is `uvx inferpilot doctor --url http://localhost:8000`.

I would especially value criticism from people operating vLLM: which evidence would you need before
trusting a tool like this to nominate a canary?

## Reddit: r/LocalLLaMA and r/vllm

**Title**

FP8 KV gave us +33.6% vLLM throughput on 7B — and we still marked the study inconclusive

**Text**

I have been building InferPilot, an evidence-first vLLM screening and controlled-comparison tool.
The latest study tested Qwen2.5-7B-Instruct on an A10G across 12 accepted cells.

- KV-pressured workload: FP8/BF16 throughput ratio = 1.336×
- Unpressured control: 1.0017×
- TTFT improved under pressure
- But the preregistered recomputation reduction did not occur

So the honest verdict was INCONCLUSIVE: a real configuration win, but not the causal mechanism we
registered. The server remained overloaded and continued preempting.

The tool's live command only screens for the next experiment; it does not call that screen a proven
bottleneck diagnosis:

`uvx inferpilot doctor --url http://localhost:8000`

Repo and full protocol/results: https://github.com/poojithdevan4D/InferPilot

I would appreciate feedback on the evidence boundary, especially from people running long-context
or high-concurrency vLLM deployments. Negative field reports are welcome too.

## LinkedIn

I got a result every inference engineer wants: **+33.6% throughput** from one vLLM configuration
change.

Then I marked the study **INCONCLUSIVE**.

In a preregistered Qwen2.5-7B/A10G experiment, FP8 KV improved throughput on the KV-pressured
workload and produced almost no change (+0.2%) on the unpressured control. But it did not reduce
recomputation—the mechanism we had registered before measuring the outcome.

That distinction is why I built InferPilot. It separates:

• live metric screening,
• controlled candidate measurement,
• SLO and quality gates,
• and explicit abstention when the evidence does not support the story.

The result is still useful. The explanation simply did not earn confirmation.

You can exercise the whole decision path without a GPU:
`uvx inferpilot demo`

Or point the read-only screen at a running vLLM:
`uvx inferpilot doctor --url http://localhost:8000`

Project: https://github.com/poojithdevan4D/InferPilot

If you operate vLLM, I would value an anonymized field report—including negative results and
abstentions.

#vLLM #LLMInference #MLOps #OpenSource #GPU

## vLLM Slack

Hi all — I built InferPilot, a read-only vLLM metrics screen plus a controlled evidence pipeline for
configuration candidates. I would value feedback on the evidence boundary rather than promotion.

In a preregistered 7B/A10G study, FP8 KV improved pressured-workload throughput by 33.6% and the
control by 0.2%, but failed the registered recomputation-reduction mechanism, so the report remained
INCONCLUSIVE. The exact recomputed-token counter used by the study is also proposed upstream in
vLLM PR #57698.

Repo: https://github.com/poojithdevan4D/InferPilot
Study: https://github.com/poojithdevan4D/InferPilot/blob/master/docs/experiments/2026-09-22-fp8-mechanism-7b-repair-results.md

For operators: what minimum evidence would make a configuration-canary recommendation useful in
your environment? I am especially interested in failures and cases where the tool should abstain.
