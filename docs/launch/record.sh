#!/usr/bin/env bash
# One command to record the headline GIF on Ubuntu/Debian — no GPU, no Rust, no brew.
#   bash docs/launch/record.sh
# Produces docs/launch/doctor.gif. Needs sudo once (to apt-install asciinema).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "==> inferpilot on PATH"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true
if ! command -v inferpilot >/dev/null; then
  echo "   inferpilot not found. Install it with:  uv tool install inferpilot   (or: pip install inferpilot)"
  exit 1
fi

echo "==> asciinema (records the terminal)"
if ! command -v asciinema >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y asciinema
fi

echo "==> agg (asciicast -> GIF; prebuilt binary, no Rust)"
AGG="$(command -v agg || true)"
if [ -z "$AGG" ]; then
  url="$(curl -fsSL https://api.github.com/repos/asciinema/agg/releases/latest \
        | grep -oE 'https://[^"]+x86_64-unknown-linux-gnu' | head -1)"
  [ -n "$url" ] || { echo "   could not find an agg linux binary; see https://github.com/asciinema/agg/releases"; exit 1; }
  curl -fsSL "$url" -o /tmp/agg && chmod +x /tmp/agg
  AGG=/tmp/agg
fi

echo "==> starting two no-GPU mock vLLM /metrics servers"
python docs/launch/mock_vllm_metrics.py --mode preempting --port 8765 >/dev/null 2>&1 & P1=$!
python docs/launch/mock_vllm_metrics.py --mode compute    --port 8766 >/dev/null 2>&1 & P2=$!
trap 'kill "$P1" "$P2" 2>/dev/null || true' EXIT
sleep 1

echo "==> recording (this runs the demo automatically)"
asciinema rec /tmp/inferpilot-demo.cast --overwrite --cols 92 --rows 22 -c "bash docs/launch/demo.sh"

echo "==> converting to GIF"
"$AGG" --theme asciinema --font-size 22 /tmp/inferpilot-demo.cast docs/launch/doctor.gif

echo
echo "✅ docs/launch/doctor.gif  ($(du -h docs/launch/doctor.gif | cut -f1))"
echo "   Add it to the top of the README (uncomment the placeholder) and commit:"
echo "     git add docs/launch/doctor.gif README.md && git commit -m 'Add headline demo GIF' && git push"
