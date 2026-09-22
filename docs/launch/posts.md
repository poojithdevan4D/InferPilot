# Ready-to-post launch content

Links: repo https://github.com/poojithdevan4D/InferPilot · vLLM PR https://github.com/vllm-project/vllm/pull/57698 · blog `docs/blog/proving-when-fp8-kv-helps-with-an-exact-counter.md`

---

## LinkedIn (post as-is)

Everyone sells fp8 KV cache as a free inference speedup. It isn't free, and it isn't always a speedup — so I built the thing that tells you *which*, and *why*, with evidence you can check.

The short version:
• fp8 KV cache helps only when your vLLM server is preempting and recomputing tokens (pure wasted work). When it's compute-bound, fp8 does nothing — and GPU utilization can't tell the two apart (both sit near 100%). The real discriminator is recomputed tokens.
• vLLM didn't expose an exact count of those, so the claim wasn't checkable. I added the counter to vLLM's scheduler — it's now an open PR to vLLM core (#57698).
• Measured on an A10G: recomputed tokens went to zero under fp8 in every paired cell, +30.8% geomean throughput, latency down across the board. Total GPU cost: $0.81.

The part I'm most proud of: I still graded it as NOT a pass. My preregistered rule required conditions one cell didn't meet, so the verdict is "not yet" — I didn't move the goalposts to the result I wanted. A second study self-invalidated on a timing rule and I threw it out.

That discipline is the product. InferPilot diagnoses vLLM configs only when the evidence supports it and abstains loudly when it doesn't.

If you run vLLM at load and want to know whether a config change will actually help before you ship it — I'd love to test it on a (sanitized) trace of yours.

Repo + write-up in comments.

(first comment: 🔗 https://github.com/poojithdevan4D/InferPilot  and vLLM PR https://github.com/vllm-project/vllm/pull/57698)

---

## vLLM GitHub Discussions — Show and tell (post as-is)

**Title:** Exact recomputed-token counter + a reproducible study of when fp8 KV cache actually helps

I've been trying to pin down *when* fp8 KV cache helps on vLLM, mechanistically rather than by vibes.

Hypothesis: fp8 helps only when the server is KV-bound enough to preempt and recompute tokens; the discriminator is recomputed tokens, not GPU util (both ~100%). vLLM didn't expose an exact recomputed-token count, so I added one to the scheduler and exposed it via the existing Prometheus path — PR #57698. A canary confirmed it: 23,324 recomputed-token executions under forced pressure, 0 in a low-pressure control.

Paired A10G study (Qwen2.5-3B, 0.29.0, pinned image): recomputed tokens → 0 under fp8 in every cell, +30.8% geomean throughput, latency down, 100% success, $0.81 total. I graded it NOT_TARGET_REGIME against a preregistered rule (one cell was only near_capacity), so it's directional, not a validated law — no cross-model or quality-safety claims.

Full evidence, the honest negative, and the tooling: https://github.com/poojithdevan4D/InferPilot

Feedback on the metric's semantics (frontier definition, prefix-cache exclusion) very welcome — that's partly why the PR is up.

---

## X / Twitter (thread, post as-is)

1/ Everyone sells fp8 KV cache as a free vLLM speedup. It's not free and not always faster. I built the tool that tells you which, and why, with checkable evidence. 🧵

2/ Hypothesis: fp8 helps ONLY when the server preempts and recomputes tokens (wasted work). Compute-bound? fp8 does nothing. And GPU util can't tell them apart — both ~100%. The real signal is recomputed tokens.

3/ Problem: vLLM didn't expose an exact recomputed-token count. So the claim wasn't checkable. I added the counter to vLLM's scheduler. It's now an open PR to vLLM core: github.com/vllm-project/vllm/pull/57698

4/ Canary check: forced KV pressure → 23,324 recomputed-token executions. Low-pressure control → 0. The counter means what it says.

5/ Result (A10G, paired fp8 vs bf16): recomputed tokens → 0 under fp8 every cell. +30.8% geomean throughput. Latency down across the board. 100% success. Total GPU cost: $0.81.

6/ The best part: I graded it NOT a pass. My preregistered rule needed conditions one cell missed, so the verdict is "not yet." Didn't move the goalposts. A 2nd study self-invalidated on a timing rule — threw it out.

7/ That discipline is the product. InferPilot diagnoses vLLM configs only when evidence supports it, abstains loudly otherwise. Not a law, not a quality claim — one model family, honest limits.

8/ Repo + full write-up + the honest negative: github.com/poojithdevan4D/InferPilot — if you run vLLM at load, I'd love to test it on a sanitized trace of yours. @vllm_project
