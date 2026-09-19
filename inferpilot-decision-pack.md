# InferPilot decision pack

Prepared 2026-09-19. Scope: the current public repository and the narrowed wedge in `docs/CRITIQUE-RESPONSE.md`.

## Executive decision

InferPilot has a credible open-source project wedge, but not yet a credible standalone-company wedge.

The defensible job is not “optimize LLM inference” or even “capacity-plan vLLM.” Those markets are already occupied by vLLM Production Stack, llm-d, NVIDIA Dynamo, Ray Serve, managed inference vendors, and increasingly capable benchmark suites. The narrow job that remains incompletely served is:

> Turn a sanitized production trace into a local, auditable acceptance test for a vLLM configuration change: a capacity-vs-SLO curve, workload-specific quality verdict, exact config diff, cost estimate, and rollback command.

The next milestone should therefore be a concierge audit with real traces, not another synthetic lever. The technical priority is a rate-sweep capacity estimator with an explicit `indeterminate` state; every diagnosis and cost recommendation depends on getting that boundary right.

---

# A. Competitive and market map

## Evaluation criteria

“Yes” to the central question requires all of the following, not merely load testing:

1. accepts the customer’s own arrival and input/output-length distribution;
2. launches controlled incumbent/candidate canaries;
3. estimates the maximum offered load meeting explicit tail-latency/error SLOs;
4. quality-gates the candidate on customer-relevant tasks;
5. persists raw evidence, resolved configuration, provenance, and uncertainty;
6. produces an actionable diff and rollback procedure.

No reviewed product clearly offers that entire workflow as an engine-independent artifact. Several already cover most of the performance half, however, so the remaining gap is much smaller than “inference engineer in a box.”

## Competitive table

| Competitor | What it actually does | Own-trace, auditable, quality-gated SLO curve? | Model / price | Gap InferPilot could own |
|---|---|---|---|---|
| **vLLM Production Stack + GuideLLM** | Kubernetes deployment, dashboards, routing, KEDA autoscaling, prefix-aware routing and KV offload. vLLM’s GuideLLM can sweep request rates and identify maximum performance/rates against an OpenAI-compatible server. | **Partial.** Strong deployment and benchmarking primitives. No evidence found of a single workflow that imports a production trace, compares config canaries, runs task-quality gates, and emits rollback evidence. | Open source; vLLM ecosystem. | A CI-style acceptance bundle for a specific trace and config change. This gap can be absorbed upstream quickly. Sources: [Production Stack](https://docs.vllm.ai/projects/production-stack/en/latest/), [production integration](https://github.com/vllm-project/vllm/blob/main/docs/deployment/integrations/production-stack.md), [GuideLLM](https://github.com/vllm-project/guidellm), [performance dashboard](https://docs.vllm.ai/en/latest/benchmarking/dashboard/). |
| **NVIDIA Dynamo Planner + AIConfigurator** | Distributed serving across vLLM/SGLang/TRT-LLM; profiling, KV-aware routing, cache management, PD disaggregation and autoscaling. Planner supports throughput, load, latency and SLA targets; advisory mode exposes predicted replica counts before applying them. | **Partial and strategically dangerous.** It profiles a deployment and targets TTFT/ITL SLAs, including separate prefill/decode scaling. No public evidence found of a customer-task quality gate joined to an auditable config acceptance report. | Apache 2.0. Kubernetes/Slurm/local; NVIDIA-led. | Quality-aware change acceptance and portable evidence could remain complementary. Generic SLO capacity planning will not. Sources: [Dynamo overview](https://docs.nvidia.com/dynamo/), [Planner](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview), [planner modes](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/choose-a-planner-mode), [advisory validation](https://docs.nvidia.com/dynamo/kubernetes/auto-deployment/dynamo-planner), [disaggregation](https://docs.nvidia.com/dynamo/dev/kubernetes/disaggregated-serving/overview). |
| **llm-d + llm-d-benchmark** | Kubernetes-native vLLM platform: load/prefix-aware routing, flow control, tiered prefix cache, PD disaggregation and autoscaling. Its benchmark tool deploys scenarios, runs synthetic staged rate sweeps, automates setup × workload experiment matrices, records manifests/results, and emits a universal benchmark report. | **Closest OSS substitute.** It already has reproducible experiment matrices and analysis. The documented examples use benchmark profiles; no end-to-end task-quality gate or rollback artifact was found. Production-trace ingestion is not the obvious default. | Apache 2.0. | A lightweight single-deployment workflow that ingests a trace and produces a quality-aware accept/reject artifact without requiring an llm-d Kubernetes stack. Sources: [project proposal](https://github.com/llm-d/llm-d/blob/main/proposals/llm-d.md), [benchmark](https://github.com/llm-d/llm-d-benchmark), [existing-deployment rate sweep](https://github.com/llm-d/llm-d-benchmark/blob/main/docs/tutorials/run/run_against_existing_example.md), [flow-control tuning](https://github.com/llm-d/llm-d/blob/main/guides/flow-control/tuning.md), [SLO autoscaling model](https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/autoscaling/hpa-wva.md). |
| **Ray Serve / Anyscale** | Ray Serve LLM supplies autoscaling, monitoring, fault tolerance, vLLM metrics, PD disaggregation, cache/session-aware routing, multi-LoRA and multi-node serving. Anyscale packages Ray as hosted or BYOC infrastructure with usage monitoring. | **Partial.** Rich operation and metrics, but no documented trace-to-config acceptance artifact with model-quality gating. | Ray is OSS; Anyscale is usage-based. Published hosted examples include T4 AC0.5682/h, L4 AC0.9542/h, A10G AC1.3635/h and A100 AC4.9591/h; larger GPU pricing is sales-led. | Local, framework-small audit for teams that do not want to adopt Ray, plus workload-specific quality comparison. Sources: [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/), [observability](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html), [autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html), [Anyscale pricing](https://www.anyscale.com/pricing). |
| **Baseten** | Managed/BYOC model deployment with autoscaling, cold-start work and optimized serving. Its value is taking operational responsibility, not producing a portable advisory for an independently operated vLLM stack. | **No public evidence of the full artifact.** Customers receive an operated platform, not an engine-neutral decision record. | Proprietary, usage/contract based; self-hosted/BYOC pricing is sales-led. | InferPilot can serve teams that explicitly reject platform migration and need evidence usable on their current cluster. Sources: [pricing](https://www.baseten.co/pricing/self-hosted/), [autoscaling](https://www.baseten.co/blog/model-autoscaling-features-on-baseten/). |
| **Fireworks AI** | Proprietary inference engine, serverless and dedicated endpoints, configurable autoscaling and per-second GPU billing. | **No.** Performance engineering is bundled into Fireworks’ service; no portable vLLM canary or customer-owned evidence format was found. | Proprietary; published usage and on-demand prices vary by model/GPU. | Same non-migration niche, but Fireworks is an alternative to self-hosting, not a tool for improving it. Sources: [pricing](https://fireworks.ai/pricing), [deployment positioning](https://fireworks.ai/scale), [dedicated endpoints](https://fireworks.ai/blog/custom-models-h100s-on-demand-deployments). |
| **Together AI** | Serverless and dedicated inference with optimized engine, autoscaling, rollouts, rollback, cold starts and multi-region failover. | **No portable artifact.** It supplies an operated endpoint and internal optimization rather than advising a customer-operated vLLM deployment. | Proprietary, serverless token pricing and dedicated endpoint pricing. | InferPilot only wins where control/data residency or existing GPU commitments make migration unacceptable. Sources: [dedicated inference](https://www.together.ai/dedicated-model-inference), [dedicated endpoint economics](https://www.together.ai/blog/on-demand-dedicated-endpoints). |
| **BentoML / BentoCloud** | OSS Python serving plus commercial deployment/BYOC. BentoCloud provides concurrency-based autoscaling, external queues, instance selection, rollout strategies and multi-cloud gateways. Documentation tells users to stress-test manually to choose concurrency. | **Partial.** It operates deployments and exposes knobs; it does not document an automatic trace→quality-gated SLO curve. | BentoML OSS; BentoCloud managed/BYOC commercial. | Automate the stress-test/acceptance step BentoML currently leaves to the operator, but this is adjacent rather than a protected market. Sources: [platform](https://docs.bentoml.com/en/latest/), [autoscaling](https://docs.bentoml.com/en/latest/scale-with-bentocloud/scaling/autoscaling.html), [deployment options](https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/configure-deployments.html), [BYOC](https://docs.bentoml.com/en/latest/scale-with-bentocloud/administering/bring-your-own-cloud.html). |
| **FriendliAI / similar optimized engines** | Proprietary engine, dedicated endpoints and BYOG; combines kernels, scheduling, caching, speculative decoding, cache-aware routing and autoscaling. It explicitly sells inference optimization on customer GPUs. | **No public portable acceptance report found.** It replaces/operates the serving stack and offers engineering expertise. | Proprietary. Dedicated published rates currently list A100-80GB $2.90/h, H100 $3.90/h and H200 $4.50/h through a stated promotional date; BYOG is custom. | Again, “keep vLLM and prove a change locally.” This is a services wedge, not strong product defensibility. Sources: [BYOG](https://friendli.ai/product/bring-your-own-gpu-byog), [pricing](https://friendli.ai/pricing), [optimization guide](https://promotion.friendli.ai/inference-performance-optimization-guide). |
| **Vidur / simulators / benchmark projects** | Vidur combines profiling and simulation to search for cost-effective configurations meeting performance constraints; it reported sub-9% latency error in its validation. LLMServingSim and emerging fleet simulators target realistic traces and heterogeneous systems. | **Performance half only.** Potentially stronger than InferPilot’s current fit arithmetic, but not an operational quality/rollback acceptance workflow. | Mostly research/OSS. | Reuse or interoperate rather than recreate a weak simulator. InferPilot’s artifact can use simulation to shortlist, then canary to accept. Sources: [Vidur paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf), [LLMServingSim](https://llmservingsim.ai/), [inference-fleet-sim](https://arxiv.org/abs/2603.16054). |

## What survives scrutiny

### Gap 1: workload-specific change acceptance, not generic benchmarking

The best remaining artifact is a signed/local evidence bundle answering:

- Does candidate configuration X dominate incumbent Y on this trace?
- What is each configuration’s lower-confidence-bound SLO capacity?
- Did candidate quality regress on the customer’s representative tasks?
- What exact command applies X and what exact command restores Y?

llm-d comes closest on experiment automation and reports; Dynamo comes closest on SLO-aware planning. Neither public workflow clearly joins performance, task quality, config provenance and rollback into one acceptance test.

### Gap 2: inference-engine upgrade/config regression CI

“Can we upgrade vLLM/change quantization/enable prefix caching without violating our workload’s latency or quality?” is recurring, budgeted and easier to trust than autonomous optimization. It also creates history: curves across engine versions, drivers, models and hardware. vLLM already benchmarks releases and accuracy internally, so InferPilot must be customer-trace-specific to add value. See vLLM’s [production-quality benchmarking description](https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality).

### Sharpest differentiation

> InferPilot is the local acceptance test for vLLM changes: replay your sanitized trace, measure the SLO-capacity boundary and task quality, then emit an auditable accept-or-rollback bundle.

Do not say “cheapest config” until the system actually measures every candidate’s capacity and includes the customer’s real all-in GPU price.

### Feature or company?

**Today: feature/open-source project plus paid audit, not a venture-scale company.**

The functionality is strategically natural for vLLM/llm-d/Dynamo and has low switching costs. It could support a small durable business if customers pay for audits, CI integration, private workload adapters and support. It becomes a larger company only if it accumulates a defensible cross-customer performance corpus or becomes the trusted release gate across engines, hardware vendors and model changes. That expansion would contradict the current vLLM-only wedge and should be earned through demand, not declared.

---

# B. Customer-discovery kit

## Objective and funnel

Four-week target:

- 30 qualified conversations requested;
- 10 interviews completed;
- 5 sanitized traces received;
- 3 audits completed;
- at least 1 written statement of willingness to pay, conditioned on a measured result.

Do not count compliments, stars, wait-list signups or “we might use this later.” Count prior incidents, current spend, trace access, audit permission and a named budget owner.

## Twelve-question Mom-Test interview

Ask these in order. Do not demo until question 10.

1. “Walk me through the last time you changed a model, GPU type, vLLM version, quantization mode or serving flag in production.”
2. “What triggered that change, and who owned the decision?”
3. “What data did you collect before approving it? Please show me the benchmark, dashboard or spreadsheet if you can.”
4. “How did you reproduce production traffic—arrival bursts, prompt/output lengths, prefix reuse, tenants and LoRAs?”
5. “What SLO actually blocked release: TTFT, inter-token latency, end-to-end latency, errors, throughput, or cost? What percentile and threshold?”
6. “Tell me about the last change that looked good in a benchmark but failed in production. What did it cost in engineer time, GPU spend or incident impact?”
7. “How do you check whether quantization, a new kernel or an engine upgrade changed answer quality? What passed or failed most recently?”
8. “How do you determine safe per-replica capacity and autoscaling targets today? Who maintains that number?”
9. “Over the last quarter, how many engineer-days and GPU-hours went into serving benchmarks, regressions or capacity planning?”
10. “Which artifacts could you export today: request timestamps, input/output token counts, latency, status, model/config metadata, prefix identifiers? Which are prohibited?”
11. “Who would need to approve a non-production replay? What security review or data-processing agreement has blocked similar tools?”
12. “What did you pay for the last external performance engagement or tool? If an audit found a measured ≥15% saving or prevented an SLO regression, which budget would pay and who signs it?”

Follow-up rules:

- Replace opinions with dates, documents, numbers and named people.
- “Would you use it?” is forbidden.
- If they cannot name a recent change or incident, they are not an early customer.
- End with one concrete ask: a schema-only trace sample within seven days and a scheduled audit window.

## Thirty target profiles and how to find them

| # | Target profile | Observable signal / where to find |
|---:|---|---|
| 1 | AI coding-assistant startup serving open models | Engineering posts mentioning vLLM/SGLang, speculative decoding, H100/A100 or model serving roles. |
| 2 | Voice-agent platform with self-hosted ASR/LLM stack | Job posts for real-time inference, strict latency, GPU platform; conference demos. |
| 3 | Customer-support automation vendor | Case studies citing data residency or private models; MLOps/platform lead on LinkedIn. |
| 4 | Enterprise-search/RAG company | GitHub issues and job posts mentioning long context, prefix caching, vLLM, Kubernetes. |
| 5 | Document-intelligence vendor | On-prem/private-cloud product pages plus GPU inference roles. |
| 6 | Legal AI company | Security/data-residency claims, private deployment offering, open-model engineering posts. |
| 7 | Healthcare AI company with on-prem inference | HIPAA/private VPC positioning and platform engineering hiring. |
| 8 | Financial-services AI platform | Bank/fintech private-model deployment case studies; internal AI platform teams. |
| 9 | Cybersecurity copilot vendor | Self-hosted appliance or customer-VPC deployment, large-context log analysis. |
| 10 | Observability/AIOps vendor adding LLM features | Existing high-volume event platform plus GPU/LLM serving jobs. |
| 11 | Translation/localization platform | High token volume, batch + interactive mix, open-model customization. |
| 12 | Content moderation provider | High sustained throughput, strict cost sensitivity, custom classifiers/LLMs. |
| 13 | Ad-tech/personalization company using generative models | GPU platform roles and real-time latency commitments. |
| 14 | Game studio operating NPC/dialogue models | GDC talks, self-hosted latency requirements, inference engineer openings. |
| 15 | Education/tutoring AI with high evening peaks | Public scale numbers, cost complaints, self-hosted model posts. |
| 16 | AI meeting/transcription company | Streaming pipeline, bursty post-meeting summarization, GPU infrastructure roles. |
| 17 | E-commerce product-content/search platform | Catalog-scale batch plus live assistant traffic; private inference announcements. |
| 18 | Robotics/autonomy company serving VLMs centrally | Edge/cloud split, private model infrastructure, multimodal serving roles. |
| 19 | Telecom internal GenAI platform | Kubernetes GPU platform talks; vendor-neutral/on-prem requirements. |
| 20 | Large SaaS company’s central AI platform team | “LLM platform,” “inference platform,” “GPU efficiency” job titles. |
| 21 | University/research computing center offering LLM endpoints | Public service docs, Slurm/Kubernetes GPU clusters, fixed budgets. |
| 22 | Government contractor/private-cloud AI provider | Air-gapped or sovereign deployment requirements. |
| 23 | Regional cloud/GPU neocloud | Marketplace models, managed vLLM offering, utilization/TCO messaging. |
| 24 | Managed Kubernetes provider adding inference | Helm/operator repos and Gateway API Inference Extension activity. |
| 25 | Model fine-tuning company that also hosts outputs | Dedicated endpoint product, LoRA serving, vLLM engineers. |
| 26 | Open-source AI application vendor offering enterprise self-hosting | Helm charts containing vLLM/LiteLLM; enterprise support page. |
| 27 | Consultancy repeatedly deploying private LLM stacks | Case studies and architects posting vLLM benchmark guides. |
| 28 | BPO/contact-center operator building internal GenAI | Large stable token volume and data-residency constraint. |
| 29 | Data-labeling/evaluation platform self-hosting judge models | High batch throughput, many model/version changes. |
| 30 | Agent-platform company running tool-use models | Long outputs, structured schemas, multi-turn traces, model-upgrade cadence. |

Discovery channels, in priority order:

1. Search GitHub code for `vllm serve`, `gpu_memory_utilization`, Helm values and Prometheus metric names; contact maintainers from company domains.
2. Search LinkedIn jobs for `vLLM`, `SGLang`, `inference platform`, `GPU efficiency`, `TTFT`, `TPOT` and `KV cache`.
3. Ask in vLLM Slack/Discussions, llm-d Slack, MLOps Community and local inference communities with a research/audit offer—not a product launch.
4. Review KubeCon/AI Infra Summit/MLSys speaker lists for production inference talks.
5. Ask GPU cloud and systems integrator contacts for teams with committed but underutilized capacity.

## Outreach copy

### Email

**Subject:** Free vLLM capacity audit using your sanitized traffic shape

Hi {{first_name}},

I’m testing InferPilot, an open-source audit tool for teams already running vLLM. It replays a sanitized traffic trace against an incumbent and one candidate config, then returns the measured request-rate boundary that meets your TTFT/ITL SLO, a task-quality check, the exact config diff, and rollback steps.

I’m looking for five design partners operating roughly 5–100 GPUs. This is not a sales demo and it will not touch production. The trace can contain only timestamps, token counts, status/latency and hashed prefix/tenant classes—no prompts or outputs. We can also generate the replay inside your environment and export only aggregate evidence.

In exchange for 30 minutes and one sanitized trace, I’ll run a free audit and give you the complete result bundle. The useful outcome may be “your current config is already best”; I won’t manufacture a saving.

Would you be the person who owns vLLM capacity or release benchmarks at {{company}}?

— Poojith

### LinkedIn DM

Hi {{first_name}}—I’m looking for five teams running 5–100 self-hosted GPUs for a free vLLM capacity audit. We replay only a sanitized traffic shape (timestamps, token counts, latency/status; no text), compare the current config with one candidate, and return an auditable SLO-capacity curve plus quality check and rollback steps. It stays non-prod and can run inside your VPC. Who owns inference benchmarking at {{company}}?

### Community post

**Looking for 5 real vLLM traces for a free, no-production-access capacity audit**

I’m building InferPilot, an open-source acceptance test for vLLM configuration changes. The current research is synthetic, so I’m deliberately stopping feature work until it survives real traffic.

If your team runs roughly 5–100 GPUs, I’ll replay a sanitized traffic trace against your current config and one candidate, sweep load around the TTFT/ITL SLO boundary, run a small workload-specific quality gate, and return every raw measurement, resolved config, cost assumption and rollback command.

Required trace fields are timestamps, input/output token counts, latency/status and optional hashed prefix/tenant classes. No prompts, outputs, user IDs or secrets. It can run entirely in your environment.

This is free for five teams. The answer may be “keep the incumbent”; there is no requirement to show a win. Reply/DM if you can share a schema-only sample and describe one recent inference configuration decision.

## Sanitized trace contract

Use JSONL or Parquet, one row per request:

```json
{
  "trace_version": "0.1",
  "request_id": "random-per-export",
  "arrival_offset_ms": 1240,
  "input_tokens": 1827,
  "output_tokens": 244,
  "ttft_ms": 381,
  "itl_ms": 27.4,
  "e2e_ms": 7042,
  "status": "ok",
  "error_class": null,
  "deadline_class": "interactive-500ms-40ms",
  "prefix_class_hash": "daily-salted-opaque-value",
  "tenant_class": "interactive-standard",
  "lora_class_hash": null,
  "cache_hit": null,
  "timestamp_bucket": "2026-09-18T14:05:00Z"
}
```

Separate deployment metadata:

```json
{
  "model_id": "customer-alias-or-public-id",
  "model_revision": "commit-or-internal-release",
  "vllm_version": "exact",
  "container_digest": "sha256:...",
  "gpu_sku": "A100-80GB",
  "gpu_count": 1,
  "engine_args_redacted": {},
  "slo": {"ttft_p95_ms": 500, "itl_p95_ms": 40, "success_rate_min": 0.995},
  "effective_gpu_cost_per_hour_usd": 0
}
```

Privacy rules:

- Never collect prompt/output text, token IDs, API keys, IPs, account IDs or raw tenant names.
- Convert wall-clock timestamps to offsets or coarse buckets; add a random export origin.
- Hash prefix classes only if repeat detection matters, with an export-specific salt destroyed after export. Plain cryptographic hashes of prompts are vulnerable to dictionary attacks and are not acceptable.
- Bucket rare tenants/LoRAs so no class has fewer than 20 requests.
- Clip token counts at the deployment maximum and document clipping.
- Let customers run the extractor locally and inspect the output before sharing.
- Best option: ship a container that reads Prometheus/logs locally and exports the schema plus aggregate histograms; raw rows never leave the VPC if policy forbids it.
- Sign a deletion commitment: delete received rows within 30 days; retain only customer-approved aggregates.

Minimum useful sample: one representative peak hour and one normal hour, at least 10,000 requests total if available. For an initial schema check, 100 rows are enough.

## Paid-problem score

Score each dimension 0–2; require **≥13/18**, with no zero in urgency, access or buyer, before offering an audit slot.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Recent pain | No relevant event | Vague/older concern | Specific incident/change in last 90 days |
| Spend | <5 GPUs or unknown | 5–20 GPUs | 20–100 GPUs or >$250k annual inference |
| Engineer cost | <1 day/quarter | 1–5 days/quarter | >5 days/quarter on benchmarking/capacity |
| SLO | None | Informal dashboard | Contracted/alerted percentile target |
| Change cadence | <2/year | Quarterly | Monthly or faster |
| Existing workaround | None because no need | Ad hoc script | Painful maintained benchmark/spreadsheet |
| Trace access | Cannot export/run locally | Approval uncertain | Can provide sanitized trace/run in VPC in 14 days |
| Buyer | No owner | Champion only | Named budget owner joins follow-up |
| Economic outcome | No measurable value | Avoided effort only | ≥15% GPU saving or material incident/purchase avoided |

Disqualify teams primarily seeking model recommendations, hosted API price comparison, hobby hardware sizing, or a production autoscaler. Those are different products.

---

# C. Deep component redesign

## C1. `saturation.py`: replace a TTFT-half ratio with flow-balance capacity evidence

### Current failure modes

The current algorithm sorts successful requests by send time and compares mean TTFT in the early and late halves. It fails when:

1. overload begins before measurement; TTFT is uniformly bad but not increasing;
2. decode is the bottleneck while admission/first-token service remains stable;
3. timeouts and failures are excluded, making the worst overload look healthier;
4. traffic mix or burstiness differs between halves;
5. cold-start or compilation pollutes one half;
6. eight successful observations are treated as sufficient, and fewer than eight are treated as not saturated rather than unknown;
7. the arbitrary 1.5× discontinuity flips a categorical downstream diagnosis;
8. the finite drain tail contaminates completion throughput, but discarding completion balance entirely throws away essential evidence.

### Proposed API

```text
assess_capacity_window(requests, server_samples, slo, window) -> LoadStateReport

LoadState = healthy | near_capacity | overloaded | indeterminate
```

Report raw estimates and intervals:

- arrival rate `lambda_hat`;
- completion rate within the observation window `mu_hat`;
- admitted/running/waiting request gauges where available;
- backlog slope and 95% CI;
- failure/timeout/censor rate and one-sided CI;
- TTFT/ITL/E2E SLO attainment and one-sided CI;
- input/output token demand and delivered token rates;
- reasons and missing signals.

### Algorithm

Define cumulative arrivals `A(t)`, terminal successes `D(t)`, and terminal failures `F(t)` on a common monotonic timeline. If a server queue gauge is available, use it as `Q(t)`. Otherwise define a relative backlog:

`B_rel(t) = A(t) - D(t) - F(t)`.

`B_rel` has unknown initial level but its slope is identifiable. Estimate the slope over the central measurement interval with Theil–Sen regression, excluding warm-up and the explicit drain period. Obtain a 95% block-bootstrap interval using time blocks longer than the observed autocorrelation/burst horizon.

Estimate rates over the same fixed interval, not arrival-plus-drain duration:

`lambda_hat = ΔA / Δt`

`mu_terminal = (ΔD + ΔF) / Δt`

`mu_success = ΔD / Δt`.

Failures remain terminal work outcomes and always fail the success-rate SLO; they must not disappear from saturation evidence. Requests still in flight at window end are right-censored and reported explicitly.

Compute offered token demand:

`lambda_in = lambda_hat * E[input_tokens]`

`lambda_out = lambda_hat * E[requested_output_tokens]`.

Compare those with measured prefill tokens/s and generation tokens/s. This detects a decode-throughput deficit even when TTFT remains flat.

Classification:

- **overloaded** if any one-sided 95% condition holds: backlog slope `> 0`; arrival rate exceeds terminal rate by more than 5%; timeout/error SLO fails; queue p95 is persistently positive and latency SLO fails; output-token demand exceeds delivered useful output capacity.
- **healthy** only if backlog-slope upper CI `≤ 0`, success-rate lower CI meets the SLO, all latency SLO upper confidence bounds pass, and `lambda_hat ≤ 0.9 * mu_capacity_lower`.
- **near_capacity** if healthy conditions pass except headroom is below 10%, or uncertainty overlaps the boundary.
- **indeterminate** for insufficient duration/events, missing terminal outcomes, nonstationary mix, or contradictory signals.

The capacity estimator should not infer a ceiling from one window. Run a geometric or binary rate sweep and fit monotone SLO attainment with isotonic regression. Define capacity as the largest offered rate whose **one-sided 95% lower confidence bound** on the required success fraction passes every SLO. Return an interval between the highest proven-pass rate and lowest proven-fail rate.

### Inputs required

- scheduled and actual arrival timestamps;
- request start, first token, finish or failure/timeout timestamp;
- requested and actual input/output tokens;
- explicit censor state at window end;
- vLLM waiting/running request gauges, queue-time histogram and prompt/generation token counters;
- trace class/prefix/tenant labels to check mix stability;
- SLO threshold and required attainment fraction.

### References

- vLLM exposes queue-time and per-request timing metrics; recent versions can return per-request timing data: [metrics design](https://docs.vllm.ai/en/v0.10.2/design/metrics.html), [per-request metrics](https://github.com/vllm-project/vllm/blob/main/docs/features/per_request_metrics.md).
- llm-d’s autoscaling design combines arrival rate, TTFT/ITL and queue demand instead of a TTFT-trend-only rule: [WVA/HPA design](https://github.com/llm-d/llm-d/blob/main/docs/architecture/advanced/autoscaling/hpa-wva.md).
- Queueing models are useful near saturation but should be calibrated with measurement; Vidur models request-level performance from profiling and workload simulation: [Vidur](https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf).

## C2. `diagnosis.py`: make diagnoses evidence-ranked hypotheses followed by discriminating probes

### Current failure modes

- Hard thresholds (`GPU ≥90%`, `KV ≥95%`, any preemption) have no uncertainty or duration requirement.
- `preemptions > 0` treats one transient event like sustained recomputation.
- GPU mean hides memory-bandwidth pressure, phase changes and idle/busy mixtures.
- `output_tokens >= prompt_tokens` is not a prefill/decode cost model.
- Architecture preconditions are emitted as strings rather than checked inputs.
- A single tree forces a regime even when evidence supports several causes.
- It jumps from correlation to intervention: preemption leads directly to fp8, without a discriminating canary.
- Engine versions can change preemption and cache semantics; metric-name compatibility is not semantic compatibility.

### Proposed model

Return a ranked set of bottleneck hypotheses with confidence and a cheapest next probe:

```text
DiagnosisReport:
  load_state
  hypotheses[]:
    kind: kv_recompute | kv_capacity_no_recompute | decode_compute_or_bandwidth |
          prefill_compute | scheduler_cpu | client_or_network | unknown
    evidence_for[]
    evidence_against[]
    score_0_1
    proposed_probe
  safe_candidates[]
  abstain_reasons[]
```

Derived signals over aligned windows:

- `r_recompute = recomputed_token_executions / useful_output_tokens`;
- `r_preempt = preempted_requests / admitted_requests`;
- KV pressure as time integral `P_kv = mean(max(0, kv_usage - 0.9))`, not peak only;
- queue pressure `P_q = waiting_request_seconds / wall_second`;
- prefill work share from actual prompt-token execution time/counters;
- decode work share from generation iterations/token executions;
- effective batch distributions, not maxima;
- GPU SM activity, DRAM bandwidth, power and kernel occupancy where DCGM/CUPTI is available;
- CPU event-loop saturation and network/client dispatch drift.

`recomputed_token_executions` must be instrumented at scheduler preemption/re-entry. Event count is not enough. At every recompute preemption, record the number of already-computed tokens discarded; separately count token executions repeated later. Export both counters:

```text
inferpilot_preempted_computed_tokens_total
inferpilot_recomputed_token_executions_total
```

Candidate logic:

1. Refuse to diagnose a limiting bottleneck unless `LoadState` is `near_capacity` or `overloaded`.
2. Mark `kv_recompute` high-confidence only when KV pressure is sustained, `r_recompute` exceeds a preregistered materiality threshold, and queue/backlog evidence indicates capacity stress.
3. Recommend an fp8 canary only after automatically checking engine support, attention architecture, head dimension, KV scaling mode, context envelope and quality protocol availability.
4. A successful canary must improve the lower confidence bound of SLO capacity and reduce recompute burden, without failing quality. If capacity improves but recompute does not fall, record “fp8 helped; proposed mechanism not confirmed.”
5. For decode-bound suspicion, run a short output-length × concurrency probe; for prefill-bound suspicion, run input-length × chunk-size probe. Use observed phase-time surfaces rather than the input/output token comparison.
6. Multiple plausible causes or missing telemetry returns `unknown`, never `other_bottleneck` with implied certainty.

### Inputs required

- all inputs from the saturation redesign;
- time-series rather than mean/peak telemetry;
- engine/version-specific capability manifest;
- scheduler iteration data: scheduled prefill/decode tokens, running/waiting requests, effective batch size, preempted and recomputed tokens;
- DCGM SM/memory-bandwidth/power metrics where available;
- verified model architecture/config and quantization scales.

### References

- vLLM documents recomputation as its default preemption mode and recommends monitoring preemption: [optimization guide](https://docs.vllm.ai/en/v0.12.0/configuration/optimization/), [CLI semantics](https://docs.vllm.cc/en/latest/cli/index.html).
- The current public metric is only a cumulative event count, supporting the need for richer instrumentation: [vLLM metric](https://docs.vllm.ai/en/v0.9.1/api/vllm/engine/metrics.html).
- Prefill and decode have distinct scaling behavior; Dynamo explicitly models them separately: [Dynamo Planner](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview).

## C3. `deployment.py`: split feasibility, performance prediction and measured acceptance

### Current failure modes

- `params × bytes` ignores quantization scales/metadata, tied embeddings, MoE residency and implementation-specific packing.
- Fixed 2 GB activation overhead ignores CUDA graphs, compilation workspace, allocator fragmentation, kernels, multimodal inputs and concurrent prefills.
- Continuous token-length distributions are collapsed into one fixed context length.
- KV block rounding, prefix sharing, sliding windows and cache offload are absent.
- TP is modeled as perfect memory sharding; communication and non-sharded state are ignored.
- A memory-fitting GPU is ranked as “cheapest” without predicting TTFT, ITL, throughput or failure rate.
- A baseline’s p50 TPOT is extrapolated to other GPUs and TP layouts.
- Sticker hourly prices omit reservations, utilization, host/network/storage and engineering overhead.

### Three-stage replacement

#### Stage 1: calibrated feasibility filter

Launch the exact engine/model/config once and read its profiled non-KV memory plus reported number and size of KV blocks. Prefer engine-resolved values over architecture arithmetic.

For each trace request `i`, estimate active KV blocks as:

`blocks_i(t) = ceil(active_context_i(t) / block_size_tokens) * layers_or_attention_groups`.

Replay the trace’s active-sequence evolution to estimate the p99 and maximum KV working set, including prefix sharing and sliding-window semantics. Apply an empirically measured fragmentation/headroom factor. “Fits” means no OOM/preemption in a short canary with the target concurrency; arithmetic only returns `plausibly_fits`.

#### Stage 2: empirical phase-performance surface

For each `(GPU, TP, engine version, weight dtype, KV dtype)` collect a small Latin-hypercube or grid profile:

- input-token buckets;
- output/context buckets;
- prefill token batch;
- decode sequence batch;
- prefix-hit fraction.

Fit monotone models:

`T_prefill = f_prefill(total_prefill_tokens, max_prompt, TP, prefix_hit)`

`T_decode_step = f_decode(active_sequences, total_KV_tokens, TP)`.

Gradient-boosted trees with monotonic constraints or piecewise bilinear interpolation are adequate; do not pretend an analytic roofline alone captures scheduler/kernel effects. Store out-of-domain bounds and prediction intervals. Vidur is a stronger existing design and should be evaluated for reuse rather than reimplemented poorly.

#### Stage 3: trace simulation then canary acceptance

Feed the actual arrival and length distribution through a discrete-event scheduler model using the fitted phase surfaces. Use simulation only to shortlist candidates. Run real rate-sweep canaries for the finalists and rank only by measured lower-bound SLO capacity per dollar:

`cost_per_SLO_output_token = all_in_hourly_cost / lower_bound(useful_output_tokens_per_hour_meeting_SLO)`.

Never compare price when the denominator fails SLO. Include both raw cloud price and customer-provided effective price. Return prediction error and refuse to rank candidates outside the calibrated domain.

### Inputs required

- exact container/driver/engine/model revisions and resolved flags;
- engine memory profiling, KV block size/count and CUDA graph modes;
- model architecture including MoE/sliding-window/hybrid attention;
- trace distribution and prefix-reuse structure;
- measured phase microprofiles per hardware/TP/dtype;
- topology bandwidth/latency for TP/multi-node;
- effective all-in hourly cost;
- final canary measurements.

### References

- Vidur combines operator profiling with predictive simulation and reports less than 9% latency error in its validation: [paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/b74a8de47d2b3c928360e0a011f48351-Paper-Conference.pdf).
- LLMServingSim uses vLLM-based layerwise profiling and trace-driven simulation: [project](https://llmservingsim.ai/).
- llm-d’s flow-control guide explicitly says concurrency must be tuned for the particular model, hardware and prompt structure: [guide](https://github.com/llm-d/llm-d/blob/main/guides/flow-control/tuning.md).

## Highest correctness per unit effort

**Replace `detect_saturation` first.** Implement `healthy/near_capacity/overloaded/indeterminate` from queue/backlog slope, fixed-window arrival/completion balance, failures and token demand; then make every diagnosis consume that report. This is lower effort than a trustworthy performance simulator and fixes the gate that currently contaminates diagnosis, goodput, configuration comparison and capacity advice.

---

# D. Preregistered held-out fp8/preemption validation

## Research question

Does baseline recomputation pressure predict a material fp8-KV improvement in SLO-constrained serving capacity, and is reduced recomputation a plausible mediator?

The primary outcome is **not overloaded-run raw throughput**. It is each arm’s maximum offered request rate meeting the preregistered TTFT, ITL and success-rate SLO.

## Frozen hypotheses

Before running held-out cells, publish code, container digests, model revisions, workload generator, analysis script and this decision rule.

- **H1—positive regime:** Cells whose bf16 baseline has sustained KV usage ≥95% for ≥25% of the steady window and recomputation burden `R = recomputed_token_executions / useful_output_tokens ≥0.10` will have fp8/bf16 SLO-capacity ratio with geometric mean ≥1.25; its cluster-bootstrap 95% lower bound must exceed 1.10.
- **H2—negative regime:** Cells with `R ≤0.01` will be practically equivalent in SLO capacity: two one-sided tests (TOST) on log capacity with equivalence margin ±10%, α=0.05.
- **H3—dose response:** Across all cells, Spearman correlation between baseline `R` and log capacity ratio will be ≥0.60, with a model/workload cluster-bootstrap 95% lower bound >0.
- **H4—mechanism:** In positive cells, fp8 will reduce recomputation burden by at least 50% median. If capacity rises without reduced recomputation, the performance result counts but the proposed mechanism does not.
- **Predictive claim:** A threshold selected now (`R ≥0.10`) must classify a material capacity gain (`≥15%`) with leave-one-model-family-out balanced accuracy ≥0.80. No threshold tuning after outcomes.

## Cell matrix

Main held-out matrix: 24 workload cells.

| Dimension | Levels |
|---|---|
| Model/architecture | Qwen2.5-3B-Instruct on A10; Mistral-7B-Instruct-v0.3 on A10; Qwen2.5-14B-Instruct on A100-40GB |
| Workload shape | long-prefill: empirical lognormal around 8k input / 256 output; decode-heavy: around 512 input / 1024 output |
| Load | 0.60, 0.85, 1.05 and 1.30 × a blinded bf16 pilot capacity estimate |
| KV arm | bf16/auto versus fp8, exact scaling behavior recorded |

That is `3 × 2 × 4 = 24` core cells. The load multiplier is fixed from a short pilot that may inspect only bf16 aggregate feasibility, never fp8 outcomes.

Transportability subset: eight additional cells selected before outcomes:

- four cells on a pinned SGLang release: both workload shapes at 0.85× and 1.05× for Qwen-3B;
- four cells on the immediately previous supported vLLM minor version using the same subset.

This brings total cells to 32 and tests engine/version interaction without pretending it is a full factorial study.

For every cell, run **three paired blocks**. Each block starts fresh servers for both arms and randomizes order AB or BA using a published seed. Use identical arrival schedules and request content within the pair, different seeds across blocks. Warm up compilation before measurement. Minimum steady measurement is 90 seconds and at least 100 terminal requests; extend up to 180 seconds if needed.

Do not use the existing development-run seeds, prompts or exact workload lengths. Model families may overlap prior work, but schedules and treatment cells are held out.

## SLOs and capacity estimation

Freeze per-shape SLOs from an external operator target or before candidate results. Example if no partner SLO is available:

- interactive/decode-heavy: TTFT p95 ≤1,000 ms, ITL p95 ≤60 ms, success ≥99%;
- long-prefill: TTFT p95 ≤10 s, ITL p95 ≤80 ms, success ≥99%.

At each load, compute pass/fail from all requests including timeouts. Estimate the boundary with monotone isotonic regression and report the interval `[highest proven pass, lowest proven fail]`. A load is a proven pass only if one-sided 95% bootstrap bounds pass every SLO. Capacity ratio uncertainty comes from paired, model/workload-cluster bootstrap resampling.

## Metrics required

Per request:

- scheduled/actual arrival, queue entry, first scheduled, first token, completion/failure;
- input, requested output and actual output tokens;
- TTFT, per-token ITLs, E2E, success/error/timeout, censoring;
- prefix/cache class and effective configuration fingerprint.

Per scheduler iteration/window:

- waiting/running requests;
- scheduled prefill, recompute and decode tokens;
- `preempted_computed_tokens_total` and `recomputed_token_executions_total`;
- useful prompt and output tokens;
- preemption mode/reason and events;
- KV occupancy distribution, block allocation/eviction and prefix hits;
- effective prefill/decode batch sizes;
- GPU SM activity, memory bandwidth, power, memory use; CPU and dispatch drift.

Primary normalized mechanism metric:

`R = recomputed_token_executions / useful_output_tokens`.

Also report discarded computed tokens per admitted request and recomputation GPU-time share if instrumentable. Do not substitute event count.

## Statistical analysis

1. Compute paired log capacity ratio per repetition block.
2. Report geometric means and percentile cluster-bootstrap 95% CIs, clustering by model × workload cell.
3. Test H1 with the preregistered lower-bound criterion; test H2 with paired TOST at ±log(1.10).
4. Test H3 with cluster-bootstrap Spearman correlation.
5. Fit a secondary mixed model, not used to rescue the primary decision:

   `log(capacity_ratio) = β0 + β1 log(1 + R) + β2 engine + β3 workload + (1|model_family)`.

6. Correct exploratory per-cell tests with Holm’s method. Publish all cells regardless of outcome.

## Counterexample hunting

Explicitly flag and investigate:

- `R ≥0.10` but capacity gain <10%;
- `R ≤0.01` but capacity gain >15%;
- fp8 worsens capacity or any SLO;
- capacity improves while recomputation does not fall;
- architecture/kernel incompatibility or quality failure;
- results that reverse across engine versions.

One credible counterexample refutes the word “iff.” The resulting model may still be a useful probabilistic heuristic.

## Quality guard

Run the protocol in section F on each model/config pair once per engine/version, not on every load cell. A quality failure makes fp8 **operationally no-go** even if H1 passes; retain the performance result in the scientific analysis.

## Go/no-go rule

- **Promote to validated heuristic:** H1, H2 and predictive balanced accuracy all pass; no unresolved severe counterexample; quality passes for the supported envelope.
- **Keep as weak heuristic:** performance association exists but H2/prediction/mechanism fails. Use only to select a canary, never recommend directly.
- **Refute/retire:** H1 lower bound ≤1.0, balanced accuracy <0.65, frequent sign reversals, or quality failures make the supported envelope impractical.

## Cost plan under $50

Budget 32 cells × 2 arms × 3 repetitions × 2 measured minutes = 6.4 GPU-hours. With 24 A10-equivalent and 8 A100-equivalent cells, approximate measured compute is about 4.8 A10-hours + 1.6 A100-hours. Even at $1.50/A10-hour and $3.00/A100-hour that is roughly $12. Add up to 2× for loading, warm-up, failed runs and quality work: about $25–35.

Controls:

- cache weights in a persistent volume;
- group randomized blocks by model while still restarting the server per arm;
- use A100 only for 14B cells;
- stop a run early only on preregistered safety conditions, never because the outcome looks unfavorable;
- hard budget alarm at $35 and termination at $45;
- publish actual invoices and excluded runs.

If the provider’s effective prices or cold-start billing make the pilot exceed $45, drop the transportability subset—not repetitions or the core held-out cells.

---

# E. Launch materials

## Tightened blog post

### FP8 KV cache gave us 40–52% more throughput—only after the server was already in trouble

FP8 KV cache is often described as a nearly free serving optimization: halve KV memory, fit more concurrent sequences, get more throughput.

Our measurements do not support that blanket claim.

Across Qwen2.5 3B, 7B and 14B plus Mistral-7B, on A10 and A100 GPUs, fp8 KV increased completed throughput by 40–52% in the cases where the bf16 server’s KV cache was full and vLLM was preempting requests. In a compute-bound 7B case with no preemption, the change was 1.7%—noise at the resolution of this study. A small SGLang batch probe showed +71%, but captured throughput only, not the telemetry needed to validate the same diagnosis.

This is not a law. It is a partly post-hoc heuristic from a small matrix of mostly single runs. We are publishing it because the mechanism is plausible and the contrast is useful, not because the evidence is finished.

#### GPU utilization gave the wrong answer

Both winning and non-winning regimes showed roughly 100% GPU utilization. Our first diagnostic therefore called the winning 3B case compute-bound and predicted no useful configuration lever.

The contradiction was in the scheduler telemetry. The bf16 run had full KV and preemptions; fp8 reduced KV pressure and increased throughput. That suggested the GPU was spending some work on scheduling/recomputation effects rather than useful output.

There is an important qualification: fp8 did not eliminate preemption in every positive case. The 14B run retained the same event count, and Mistral’s count increased. Therefore “fp8 wins by killing preemption” is not established. Event count is also a poor dose measure. The next experiment must record discarded/recomputed token executions per useful output token and test whether reducing that burden mediates the capacity gain.

The current claim is narrower:

> In our tested overloaded cells, baseline KV pressure plus preemption identified the cases where an fp8 canary was worth trying. It did not prove why fp8 helped or that the rule generalizes.

#### These were overload-resilience wins, not free speedups

Every positive run had extreme TTFT—roughly 20–94 seconds. Even at a lower offered load, fp8 improved throughput and roughly halved TTFT but the server remained saturated.

So the honest product metric is not “fp8 made requests faster.” It is the highest offered load that still meets a defined TTFT/ITL/error SLO. Our existing experiments did not measure a healthy capacity boundary; they measured behavior after the boundary had been crossed.

InferPilot is being narrowed around fixing that. Given a production traffic shape and an existing vLLM deployment, it should sweep load around the boundary, compare an incumbent and candidate canary, and return the measured capacity-vs-SLO curve with uncertainty.

#### Quality did not get a clean pass

On Qwen2.5-3B we ran four smoke tests:

- greedy token agreement was 12%, showing that trajectories changed but not whether they became worse;
- a 16-question factual set produced 12/16 correct in both arms and no baseline-to-fp8 regression;
- five single-needle probes at about 14k tokens passed in both arms;
- a top-20 teacher-forced KL approximation failed its threshold: mean KL exceeded 0.01 and p99 reached 0.39.

That is not “lossless.” It means a tiny factual/retrieval smoke test did not detect a regression while the distributional preflight detected a real shift. The result does not cover other models, 100k context, multi-hop retrieval, structured outputs or long free-running reasoning.

Any operational recommendation must therefore be conditional on a workload-specific quality gate. For a fresh deployment we want full-distribution—or explicitly bounded top-k—teacher-forced comparison, retrieval at the actual maximum supported context, and at least one task metric that resembles production.

#### What InferPilot is—and is not

InferPilot is not an autonomous inference engineer and it will not mutate production. vLLM Production Stack, llm-d and NVIDIA Dynamo are already building the operational control plane for deployment, routing and autoscaling.

The narrower goal is a local acceptance test for vLLM configuration changes:

1. ingest a sanitized traffic trace;
2. run incumbent and candidate canaries;
3. sweep load around the SLO boundary;
4. block on errors or workload-specific quality regression;
5. emit raw evidence, an exact config diff, cost assumptions and a rollback command.

The repository now includes the small raw result bundles used by the demos. It also includes the critique that forced this narrowing. The current evidence is enough to motivate a preregistered test and real customer audits—not enough to claim a general optimization law.

We are looking for five teams operating 5–100 self-hosted GPUs. The requested trace contains arrival offsets, token counts, latency/status and optional salted prefix classes—no prompts or outputs—and the replay can remain inside your environment. The audit is free, and “keep your current configuration” is an acceptable result.

Repository: [InferPilot](https://github.com/poojithdevan4D/InferPilot)

## Show HN

**Title:** Show HN: InferPilot – replay a vLLM trace and measure the SLO capacity of a config change

**First comment:**

I built InferPilot after getting a result wrong.

We tested fp8 KV cache on several vLLM deployments. GPU utilization was about 100% in both the cases where fp8 helped and where it did nothing, so the first diagnostic called a +52% case compute-bound. The overlooked signal was KV pressure plus scheduler preemption.

The current evidence is deliberately described as a heuristic, not a law: four model points, two GPU families, mostly single runs, synthetic traffic, and all positive results beyond a healthy SLO boundary. Quality is also not a clean pass—small factual and 14k retrieval smoke tests showed no regression, but a teacher-forced top-k KL gate failed.

That critique changed the project. The intended product is now an offline acceptance test, not an autonomous controller: replay a sanitized production trace, run incumbent/candidate canaries, estimate the request-rate boundary meeting TTFT/ITL/error SLOs, quality-gate the change, and emit the evidence plus rollback steps.

Raw result JSON for the demos and the full critique response are committed. I’m looking for five teams running 5–100 GPUs for free, non-production audits. The uncomfortable feedback I especially want: is this useful beyond GuideLLM/llm-d/Dynamo, and would you trust a local result bundle enough to change a serving config?

## r/LocalLLaMA post

**Title:** FP8 KV gave +40–52% only in our preempting vLLM runs; here are the raw results and the failed quality gate

We tested bf16/auto versus fp8 KV on Qwen2.5 3B/7B/14B and Mistral-7B across A10/A100. In the KV-full, preempting cells, fp8 improved completed throughput 40–52%. In a compute-bound, non-preempting 7B cell it moved 1.7%.

Before anyone turns that into folklore: the study is small, partly post-hoc, mostly single runs, synthetic, and the positive cells were already overloaded. We have not shown a 50% healthy-capacity gain. Preemption event count is also too crude: in two positive cells it did not fall, so the recomputation mechanism remains a hypothesis.

Quality was mixed. Qwen-3B had no regressions on 16 factual prompts and passed 5/5 single-needle retrieval at ~14k in both modes, but our top-20 teacher-forced KL preflight failed (p99 0.39). We do not call fp8 lossless.

The repo now includes the demo result JSON and a public critique response. I’m narrowing the tool to a local vLLM acceptance test: replay a sanitized traffic shape, sweep the SLO boundary, compare one candidate, run a workload-specific quality gate, and output the config diff plus rollback command.

I’m looking for five non-production traces. Required fields are arrival offsets, input/output token counts, latency/status and optional salted prefix classes—no text. Happy to run entirely in your environment. The most useful response would be a counterexample: preemption without an fp8 gain, or an fp8 gain without preemption.

## X/Twitter thread

**1/8** We measured fp8 KV cache on self-hosted LLM serving. It gave +40–52% throughput in our KV-full, preempting vLLM runs—and ~0% in a compute-bound run. Useful result. Not a law. Here is what broke, including our own first diagnosis.

**2/8** GPU utilization was ~100% in both regimes. Our first classifier therefore called the +52% Qwen-3B case compute-bound and predicted no config win. The scheduler was preempting under full KV pressure; GPU% alone hid that.

**3/8** But “preemption causes the win” is still only a hypothesis. The matrix is small, partly post-hoc and mostly single runs. In the 14B/Mistral positive cases, preemption events did not disappear. We need recomputed tokens per useful token—not an event count.

**4/8** The biggest caveat: every positive result was already overloaded, with TTFT around 20–94s. This is not evidence of a free low-load speedup. At best, fp8 may move the goodput ceiling. We have not measured that boundary rigorously yet.

**5/8** Quality was not a clean pass. Tiny Qwen-3B factual and 14k single-needle tests found no regression. Greedy outputs diverged. A teacher-forced top-20 KL gate failed (p99 0.39). So: smoke-tested, task-dependent, definitely not “lossless.”

**6/8** That forced a product reset. InferPilot is no longer pitched as an autonomous inference engineer. vLLM Production Stack, llm-d and Dynamo are already commoditizing that control plane.

**7/8** The narrower goal: replay your sanitized vLLM trace, run incumbent/candidate canaries, measure the TTFT/ITL/error SLO capacity curve, quality-gate the change, and emit an auditable config diff + rollback command. Offline only.

**8/8** Raw demo evidence and the critique response are in the repo. I’m looking for five teams running 5–100 GPUs for a free non-prod audit. No prompt text required; it can run inside your VPC. Counterexamples are more valuable than stars: https://github.com/poojithdevan4D/InferPilot

## 60-second demo video script

**0–5s — Terminal, before command**  
Voice: “This server is drowning: Qwen 3B on an A10, long prompts, 46-second p95 time to first token.”  
On screen: `uv run python scripts/cost_rescue_demo.py`

**5–15s — Baseline evidence**  
Voice: “InferPilot reads a committed result bundle—not a fabricated dashboard. GPU utilization is 100%, KV is full, and the server reports preemption.”  
On screen: highlight baseline throughput, TTFT, KV and preemption fields.

**15–25s — Diagnosis**  
Voice: “GPU utilization alone would call this compute-bound. InferPilot treats KV pressure plus preemption as a heuristic that makes an fp8 canary worth testing—not as proof.”  
On screen: highlight `candidate: kv_cache_dtype=fp8` and caveat label.

**25–38s — Candidate result**  
Voice: “In this stored run, fp8 increased completed throughput 52% and cut TTFT roughly in half. The deployment was still overloaded, so this is overload resilience—not a healthy-production speedup.”  
On screen: side-by-side measured numbers and `SLO: FAIL` if applicable.

**38–48s — Quality**  
Voice: “Quality does not get a green badge. Small factual and 14k retrieval smoke tests passed, but the teacher-forced KL preflight failed. Production apply remains blocked without a workload-specific gate.”  
On screen: `QA: smoke pass`, `needle: smoke pass`, `KL: FAIL`.

**48–57s — Intended workflow**  
Voice: “The target workflow is trace, rate sweep, incumbent-versus-candidate canary, quality check, exact diff, rollback.”  
On screen: simple pipeline with “offline—never mutates production.”

**57–60s — Ask**  
Voice: “We need five real sanitized traces. If you run vLLM, bring us a counterexample.”  
On screen: repository URL.

## Five skeptical comments and replies

1. **“This is just GuideLLM/llm-d with extra prose.”** — Fair risk. The only defensible addition is a single local acceptance artifact joining a real trace, candidate canary, task-quality gate, config provenance and rollback; if users do not value that bundle, the project should stop.
2. **“You fitted the rule after seeing the result.”** — Yes. It is labeled a post-hoc heuristic; the next test freezes thresholds/effect sizes and hunts counterexamples on held-out cells.
3. **“Your positive runs are unusably overloaded.”** — Correct. They show overload behavior, not healthy goodput; the next product milestone is a rate sweep that measures the actual SLO boundary.
4. **“Five needles and 16 questions prove nothing about quality.”** — Correct. They are smoke tests; the KL gate failed, and no production recommendation should pass without max-context plus workload-specific task evaluation.
5. **“Dynamo will absorb this.”** — Generic planning, yes. The bet is a lightweight, local, engine-change acceptance record for teams staying on plain vLLM; customer discovery must prove that is worth maintaining separately.

---

# F. Cheapest defensible quality protocol (<15 GPU-minutes per config)

## Principle

This is a **non-inferiority screen**, not proof of semantic equivalence. It must exercise the exact candidate engine path, including KV writes/reads, kernels, CUDA graphs and quantization scales. A Transformers forward pass that bypasses vLLM’s cache implementation is not an adequate substitute.

Run baseline and candidate in randomized order after identical warm-up. Use fixed model/container/driver revisions, tokenizer, prompts, decoding parameters and seeds. Persist prompt hashes, token IDs, raw scores and effective config. Keep evaluation prompts free of benchmark contamination where possible; prefer customer-owned examples.

## Time allocation per config

On a single target GPU after model load/warm-up:

| Component | Budget | Minimum sample |
|---|---:|---:|
| Teacher-forced distribution shift | 4 min | 4,096 scored positions across length buckets |
| Max-context retrieval/aggregation | 5 min | 12 probes at 3 depths × 4 task variants |
| Task-quality generation | 4 min | 40 short deterministic cases or 20 multi-hop cases |
| Repeat/noise calibration allowance | 2 min | baseline A/B subset and retries |

Total ≤15 measured GPU-minutes per configuration. Model loading is reported separately; if the operational promise includes cold starts, do not hide it.

## Gate 1: teacher-forced distribution shift

### Preferred: full vocabulary

Score the same reference history under baseline and candidate at 4,096 positions sampled from customer-domain text and stratified by context position: 0–4k, 4–16k, 16k–50% max, and 50–100% max where supported. Request all prompt log probabilities (`-1`/no cap in supported vLLM versions), in small chunks to avoid OOM. vLLM documents that uncapped logprobs can return output-length × vocabulary values and may itself cause OOM, so memory must be bounded carefully: [engine arguments](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html).

For each position compute:

- `KL(P_baseline || P_candidate)`;
- Jensen–Shannon divergence;
- total variation distance;
- baseline chosen-token log-probability delta;
- top-1 flip and top-5 set overlap.

Numerically compute in log space and verify probability mass sums.

### Fallback: union-support top-k with a residual-mass bound

Top-k renormalization is not full KL and can hide shifts in omitted mass. If full logits are unavailable, request at least top-100 from both arms, take the union of token supports, and include an explicit `OTHER` bucket equal to remaining probability mass. Compute a coarse-grained KL/JS over union + OTHER. Report the maximum possible unresolved tail contribution or label the test `approximate`. Never renormalize only on baseline top-k and call it KL.

### Noise floor

Run baseline twice on a 1,024-position subset, with separate server restarts and identical deterministic inputs. This measures implementation/restart nondeterminism, not “different seeds” under teacher forcing. Let `M0` and `P990` be baseline-vs-baseline mean and p99 JS/KL.

Candidate thresholds:

- mean KL ≤ `max(0.01, M0 + 3*SE0)`;
- p99 KL ≤ `max(0.10, P990 + 3*robust_scale0)`;
- top-1 flip rate ≤ `max(1%, baseline_flip + 1 percentage point)`;
- no context bucket may exceed 2× the global mean threshold;
- no non-finite logits, missing positions or unexpected vocabulary mismatch.

Why these thresholds: 0.01/0.10 preserve the project’s existing conservative preflight; the relative floor prevents deterministic numerical noise from triggering a false failure. They are engineering guardrails, not universal scientific constants. Calibrate them against known-safe engine restarts and known-bad perturbations before marketing them.

A distribution failure does not automatically prove task regression; it blocks a generic “quality safe” claim and requires stronger task evidence. For initial production policy, fail closed.

## Gate 2: retrieval at the true deployed maximum context

Run at the smaller of the configured production maximum and the model’s genuinely supported context—not 14k merely because it is cheap. Use 12 deterministic RULER-style cases:

- single-key retrieval at 10%, 50%, 90% depth;
- multi-key retrieval requiring two non-adjacent facts at the same depths;
- aggregation/counting over distributed facts;
- one negative case where the requested key is absent.

Use at least two haystack templates and randomized keys. Exact-match normalize only whitespace/case. Single-needle alone is too easy; RULER adds multi-hop tracing and aggregation specifically because vanilla needle tests overstate usable context: [RULER](https://arxiv.org/abs/2404.06654), [lm-eval implementation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ruler/README.md).

Thresholds:

- candidate must have zero baseline-pass → candidate-fail regressions in 12 paired probes;
- candidate accuracy must be ≥ baseline accuracy;
- if baseline itself is <10/12, the deployment context is not certified and the result is `indeterminate`, not a candidate pass.

Twelve cases are still a smoke gate, so advertise exactly that. For high-risk 100k+ deployments, increase to at least 30 and include customer documents.

## Gate 3: one production-shaped task metric

Default cheap option: 40 customer-like structured extraction cases with expected semantic fields, not merely JSON syntax. Use deterministic decoding and validate:

- schema validity;
- required-field exact match/F1;
- enum and numeric constraint accuracy;
- refusal/error behavior.

If the deployment is reasoning/retrieval-heavy, substitute 20 paired multi-hop QA cases with canonical answers and evidence requirements. JSONSchemaBench provides real schema diversity, while RULER/HELM long-context tasks cover multi-hop behavior: [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench), [paper](https://arxiv.org/abs/2501.10868), [HELM long context](https://crfm.stanford.edu/helm/long-context/latest/).

Thresholds:

- zero new catastrophic failures: invalid output, empty/repetition loop, refusal on a baseline-success case;
- candidate paired accuracy drop no worse than 2 percentage points **and** no more than one baseline-pass → candidate-fail case in 40;
- for 20 multi-hop cases, require zero paired regressions unless the candidate also fixes at least as many baseline failures; otherwise collect a larger sample because 20 cases cannot establish a 2-point margin.

The statistically honest result for tiny samples is often `indeterminate`. Do not turn lack of detected regression into equivalence.

## Combined decision

- **PASS for this envelope:** all three gates pass; engine/model/config/context/task are recorded.
- **FAIL:** any catastrophic task regression, retrieval regression, non-finite/corrupt output, or distribution threshold failure under the conservative initial policy.
- **INDETERMINATE:** baseline is weak, sample is incomplete, full/tail-bounded logits unavailable, or confidence is insufficient.

Greedy token agreement is reported only as a drift diagnostic. Factual QA may be part of the task gate but is not a separate certificate. No LLM judge is required in the 15-minute protocol.

## What this still cannot catch

- rare regressions below the small sample’s detection power;
- long free-running error accumulation, repetition and early stopping beyond tested output lengths;
- stochastic sampling changes at production temperature/top-p;
- safety, bias or multilingual regressions absent from the selected tasks;
- multi-turn agent/tool trajectories and stateful prefix-cache interactions;
- cross-tenant cache/isolation bugs;
- every context position or prompt distribution shift;
- changes caused by production concurrency if quality probes run only in isolation.

Therefore run a final small quality subset under the same concurrent canary load. If outputs differ between isolated and loaded execution, treat that as a correctness incident.

KV-cache quantization research itself evaluates real long-context tasks rather than relying on scalar perplexity; KIVI analyzes different key/value quantization sensitivity and evaluates downstream quality, while newer long-context benchmarks include multi-document QA, summarization and retrieval: [KIVI](https://arxiv.org/abs/2402.02750), [long-context KV optimization benchmark](https://arxiv.org/abs/2607.05399). vLLM’s quantized-cache behavior and scaling factors are implementation-specific and must be captured in provenance: [vLLM quantized KV cache](https://docs.vllm.ai/en/v0.18.2/features/quantization/quantized_kvcache/).

---

# Immediate execution order

1. Conduct ten interviews using section B; secure five schema samples before adding a new optimization lever.
2. Implement the flow-balance `LoadStateReport` and an adaptive rate sweep.
3. Complete one concierge audit end to end, manually if necessary, and publish a redacted acceptance bundle.
4. Only then run the preregistered fp8 matrix; scientific validation does not substitute for buyer validation.
5. Rewrite remaining README/blog remnants that still say “senior engineer,” “law for the headline,” “quality-verified,” or imply that memory fit equals a cheapest SLO-meeting deployment.

The project earns another six months only if real operators share traces, the audit finds material value beyond existing tools, and at least one buyer commits budget after seeing their own evidence.
