# Launch content (research-tuned)

Positioning: **InferPilot — the vLLM doctor.** Hook: it often tells you *not* to bother.
Install headline everywhere: `uvx inferpilot doctor --url http://localhost:8000`
Links: repo https://github.com/poojithdevan4D/InferPilot · vLLM PR https://github.com/vllm-project/vllm/pull/57698
Asset: the `doctor.gif` at the top of the README (see RECORDING.md).
Rules from the research: lead with substance + open source, disclose AI assistance, no links in HN/Reddit titles, be in the thread for the first few hours, never beg upvotes.

---

## Hacker News — Show HN (post FIRST; it's the primary spike for CLI/infra tools)

**Title** (no link in title, <80 chars):
`Show HN: InferPilot – the vLLM doctor that tells you when fp8 won't help`

**First comment (post within 5 min of submitting):**

I run into the same thing every time I tune a vLLM server: every knob is sold as a free speedup, and GPU utilization sits near 100% whether you're genuinely compute-bound or just thrashing on KV cache. So I built a tool that reads a live vLLM's Prometheus /metrics and tells you which regime you're actually in — and whether a change like fp8 KV cache will help, or do nothing.

Two scrapes, no benchmark run:

    uvx inferpilot doctor --url http://localhost:8000

It prints one of: KV-bound & preempting (fp8 worth a canary), compute/other-bound (fp8 won't help — don't bother), near-capacity, or healthy — plus the evidence that forced the call. The part I'm proudest of is the honest negative: a tool that says "you're compute-bound, save your time" is more useful to me than one that always finds a speedup.

The mechanism behind it (an exact recomputed-token counter) is an open PR to vLLM core (#57698). Everything's MIT, no telemetry.

Honest limits: it's a screening read from /metrics, not a measured capacity guarantee — for that it runs a controlled rate sweep. I validated the fp8 mechanism on a 3B model (recompute → 0, +30% throughput) but a 7B held-out study came back inconclusive (the win held, the mechanism didn't replicate), and I say so in the repo. Also: this was built with heavy AI assistance, which I mention because I'd want to know.

Happy to answer anything.

---

## vLLM GitHub Discussions — Show and tell (lowest-risk, highest-credibility venue)

**Title:** InferPilot — a read-only "doctor" that screens a live vLLM from /metrics and says whether fp8 KV will help

Point it at a running vLLM and it screens the server from its Prometheus /metrics (two scrapes), classifying the regime: KV-bound & preempting → fp8 worth a canary; queue building while KV has headroom → compute/other-bound, fp8 won't help; near-capacity; healthy. One line:

    uvx inferpilot doctor --url http://localhost:8000

It's built on the exact recomputed-token counter I proposed in #57698 (the discriminator for preemption-driven recompute waste). The honest negative is the point — it tells you when a lever won't move anything. MIT, no telemetry, built with AI assistance (disclosed).

Feedback very welcome on the regime heuristics and the metric semantics. Repo: https://github.com/poojithdevan4D/InferPilot

---

## LinkedIn (attach the doctor.gif)

Every vLLM knob is sold as a free speedup. GPU utilization can't tell you which ones matter — it sits near 100% whether you're compute-bound or thrashing on KV cache.

So I built InferPilot — the vLLM doctor. Point it at your server and it reads the live /metrics and tells you the real bottleneck, and whether a change like fp8 KV cache will actually help:

uvx inferpilot doctor --url http://localhost:8000

The part I care about most: it's honest. When you're compute-bound, it says "fp8 won't help — don't bother." A tool that tells you NOT to do something is rarer, and more useful, than one that always sells a win.

Open source (MIT), no telemetry, built with heavy AI assistance (worth disclosing). The mechanism it relies on is an open PR to vLLM core.

If you run vLLM at load, I'd genuinely value your eyes on it. 🔗 in comments.

---

## X / Twitter (thread; attach the gif to tweet 1)

1/ Every vLLM knob is sold as a free speedup. GPU util can't tell you which ones matter — it's ~100% whether you're compute-bound or thrashing on KV cache. So I built the vLLM doctor. 🧵

2/ Point it at your server, no benchmark run:

   uvx inferpilot doctor --url http://localhost:8000

It reads live /metrics and tells you the regime + whether fp8 KV will help.

3/ The best part is the honest NO: if you're compute-bound it says "fp8 won't help — don't bother." A tool that tells you not to do something beats one that always finds a win.

4/ Under the hood it's an exact recomputed-token counter — now an open PR to vLLM core (#57698). MIT, no telemetry, built with AI assistance (disclosed).

5/ Repo + the honest limits (3B mechanism confirmed, 7B inconclusive — I say so): github.com/poojithdevan4D/InferPilot  @vllm_project
