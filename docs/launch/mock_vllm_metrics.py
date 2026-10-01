#!/usr/bin/env python3
"""A tiny mock vLLM Prometheus /metrics endpoint — for recording the InferPilot demo
without a GPU or a real vLLM server.

In 'preempting' mode the preemption counter rises on every scrape, so
`inferpilot doctor --url http://localhost:8765` sees a genuine preemption *rate* across
its two scrapes and reaches the KV-bound verdict. In 'compute' mode the queue builds
while the KV cache has headroom and preemptions stay flat, so it reports that the
snapshot has no KV capacity signal.

    python docs/launch/mock_vllm_metrics.py --mode preempting --port 8765
    inferpilot doctor --url http://localhost:8765 --interval 2
"""

from __future__ import annotations

import argparse
import http.server

MODES = {
    "preempting": {"kv": 0.98, "running": 12, "waiting": 7, "pre_step": 9},
    "compute": {"kv": 0.55, "running": 20, "waiting": 7, "pre_step": 0},
}


def _handler(cfg: dict):
    state = {"preemptions": 10}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the recording clean
            pass

        def do_GET(self):
            if self.path.rstrip("/") != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            body = (
                f'vllm:kv_cache_usage_perc{{model_name="demo"}} {cfg["kv"]}\n'
                f'vllm:num_requests_running{{model_name="demo"}} {cfg["running"]}\n'
                f'vllm:num_requests_waiting{{model_name="demo"}} {cfg["waiting"]}\n'
                f'vllm:num_preemptions_total{{model_name="demo"}} {state["preemptions"]}\n'
            )
            state["preemptions"] += cfg["pre_step"]
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=sorted(MODES), default="preempting")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"mock vLLM /metrics ({args.mode}) at http://localhost:{args.port}/metrics  (Ctrl-C to stop)")
    http.server.HTTPServer(("127.0.0.1", args.port), _handler(MODES[args.mode])).serve_forever()


if __name__ == "__main__":
    main()
