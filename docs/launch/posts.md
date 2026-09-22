# Ready-to-post launch content

Links: repo https://github.com/poojithdevan4D/InferPilot · vLLM PR (open, under review) https://github.com/vllm-project/vllm/pull/57698 · blog `docs/blog/proving-when-fp8-kv-helps-with-an-exact-counter.md`
Images: `docs/launch/fp8-recompute-result.png` (primary chart) · PR screenshot (secondary).
Honesty rule: the PR is OPEN / under review — say "open PR to vLLM," never "merged" or "I contributed to vLLM."

---

## LinkedIn (post as-is · attach the chart)

Everyone sells fp8 KV cache as a free vLLM speedup. It's not free — and not always faster. So I built the thing that proves *when* it helps, and *why*, with evidence you can check.

The finding: fp8 only helps when your server is preempting and **recomputing tokens** — pure wasted work. Compute-bound? It does nothing. And GPU utilization can't tell the difference (both sit at ~100%). The real signal is recomputed tokens.

vLLM didn't expose that number. So I added the exact counter to vLLM's scheduler — it's now an **open PR to vLLM core** (#57698, under review).

Measured on an A10G: recomputed tokens → **0** under fp8, **+30% throughput**, latency down, 100% success. Total GPU cost: **$0.81**.

The part I'm proudest of: I graded it as **NOT a pass.** My preregistered rule required conditions one cell missed, so the verdict is "not yet" — I didn't move the goalposts to the result I wanted.

That discipline is the product. InferPilot diagnoses vLLM configs only when the evidence supports it — and says "I don't know" out loud when it doesn't.

If you run vLLM at load, I'd love to test it on a sanitized trace of yours. 👇

🔗 https://github.com/poojithdevan4D/InferPilot

---

## vLLM GitHub Discussions — Show and tell (post as-is · embed the chart)

**Title:** An exact recomputed-token counter + a reproducible study of when fp8 KV cache actually helps

I wanted to pin down *when* fp8 KV cache helps on vLLM — mechanistically, not by vibes.

Hypothesis: it helps only when the server is KV-bound enough to preempt and recompute tokens. The discriminator is recomputed tokens, not GPU util (both ~100%). vLLM didn't expose an exact count, so I added one to the scheduler via the existing Prometheus path — **open PR #57698** (feedback on the metric semantics very welcome; that's partly why it's up).

Canary: 23,324 recomputed-token executions under forced pressure, **0** in a low-pressure control.

Paired A10G study (Qwen2.5-3B, 0.29.0, pinned image): recomputed tokens → **0** under fp8 in every cell, **+30.8% geomean throughput**, latency down, 100% success, **$0.81** total. I graded it `NOT_TARGET_REGIME` against a preregistered rule — directional, not a validated law. No cross-model or quality-safety claims.

Full evidence + the honest negative: https://github.com/poojithdevan4D/InferPilot

---

## X / Twitter (thread · attach chart to tweet 4)

1/ Everyone sells fp8 KV cache as a free vLLM speedup. It's not free, and not always faster. I built the tool that proves *when* it helps — and *why*. 🧵

2/ fp8 only helps when your server preempts and **recomputes tokens** (wasted work). Compute-bound? It does nothing. GPU util can't tell them apart — both ~100%. The real signal is recomputed tokens.

3/ vLLM didn't expose that number. So I added the exact counter to its scheduler. It's now an **open PR to vLLM core** (under review): github.com/vllm-project/vllm/pull/57698

4/ Result (A10G, paired): recomputed tokens → **0** under fp8, **+30% throughput**, latency down, 100% success. Total GPU cost: **$0.81**.

5/ Best part: I graded it as NOT a pass. My preregistered rule needed conditions one cell missed — so the verdict is "not yet." Didn't move the goalposts.

6/ That discipline is the product. InferPilot diagnoses vLLM configs only when evidence supports it, and abstains loudly when it doesn't. Honest limits, one model family.

7/ Repo + full write-up + the honest negative: github.com/poojithdevan4D/InferPilot — run vLLM at load? I'd love a sanitized trace. @vllm_project
