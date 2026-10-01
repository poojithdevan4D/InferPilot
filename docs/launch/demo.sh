#!/usr/bin/env bash
# InferPilot headline demo — drives `inferpilot doctor` against a local mock vLLM so it
# records authentically with NO GPU. Record it with asciinema, convert with agg.
#
#   python docs/launch/mock_vllm_metrics.py --mode preempting --port 8765 &
#   python docs/launch/mock_vllm_metrics.py --mode compute    --port 8766 &
#   asciinema rec demo.cast -c "bash docs/launch/demo.sh"
#   agg demo.cast docs/launch/doctor.gif
#
# Requires `inferpilot` on PATH (uvx/pip/uv tool install). See RECORDING.md.
set -euo pipefail

type_cmd() {   # typewriter effect for the prompt line
  printf '\033[1;32m$\033[0m '
  local s=$1 i
  for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.035; done
  printf '\n'
}

clear
sleep 0.8

# 1) The actionable verdict: a server that IS preempting.
type_cmd "inferpilot doctor --url http://localhost:8765 --interval 2"
sleep 0.3
inferpilot doctor --url http://localhost:8765 --interval 2
sleep 3.0
printf '\n\n'

# 2) Queueing with KV headroom: prioritize compute or scaling tests.
type_cmd "inferpilot doctor --url http://localhost:8766 --interval 2"
sleep 0.3
inferpilot doctor --url http://localhost:8766 --interval 2
sleep 4.0
