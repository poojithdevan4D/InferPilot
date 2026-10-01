# Recording the headline GIF

The single most important viral asset is a ~12-second GIF at the top of the README
showing `inferpilot doctor` print a verdict about a live server. This kit records it
authentically with **no GPU** by driving the doctor against a local mock `/metrics`.

## One-time setup

```bash
# the tools (asciinema records a terminal; agg turns the cast into a GIF)
brew install asciinema agg          # macOS
# or: pipx install asciinema ; cargo install --git https://github.com/asciinema/agg

# inferpilot on PATH
uv tool install inferpilot          # or: pipx install inferpilot
```

## Record

In three terminals (or background the first two):

```bash
# 1. a server that is preempting  -> "fp8 worth testing"
python docs/launch/mock_vllm_metrics.py --mode preempting --port 8765 &

# 2. a server that is compute-bound -> the honest "fp8 won't help"
python docs/launch/mock_vllm_metrics.py --mode compute --port 8766 &

# 3. record the demo
asciinema rec demo.cast --overwrite -c "bash docs/launch/demo.sh"
```

Then stop the mock servers (`kill %1 %2`).

## Convert to GIF

```bash
agg --theme asciinema --font-size 22 demo.cast docs/launch/doctor.gif
```

Keep it under ~5 MB. If it's heavy: lower `--font-size`, or trim trailing idle in `demo.cast`.

## Polish before recording
- Use a clean, neutral prompt — no home paths, no internal branch names, no company name.
- A dark terminal theme reads best in README cards.
- The two-verdict sequence is deliberate: the **second** (honest "won't help") is the
  share-bait — "finally, a tool that admits it."

## Embed in the README

Put it right under the title:

```markdown
![InferPilot diagnosing a live vLLM](docs/launch/doctor.gif)
```

## A real recording (optional, even better)
If you have a real vLLM up, record against it instead of the mock — same demo, swap the
URLs for your server. Drive it into preemption first (high concurrency, long context) so
the KV-bound verdict is genuine.
