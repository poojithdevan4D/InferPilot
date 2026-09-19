"""A dependency-free fake of the vLLM OpenAI server for tests.

Implements just enough to exercise the runner without a GPU, a model, or vLLM:
* ``GET /health`` -> 200 (or 503 in ``never_ready`` mode)
* ``POST /v1/completions`` -> streaming SSE with configurable behaviour

Usable two ways:
* in-thread via :func:`serve_in_thread` (for client integration tests), and
* as a subprocess via ``python fake_vllm_server.py --port P --mode M`` (for
  orchestrator lifecycle tests, including crash/OOM startup modes).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

# Modes:
#   normal        health OK; stream N text chunks + usage + [DONE]
#   empty         health OK; stream no text, usage completion_tokens=0
#   malformed     health OK; stream a non-JSON data chunk
#   error500      health OK; completions returns HTTP 500
#   hang          health OK; completions sends headers then stalls (client timeout)
#   never_ready   health returns 503 forever (readiness timeout)
#   startup_crash exit(1) immediately (subprocess only)
#   oom_crash     print CUDA OOM to stderr then exit(1) (subprocess only)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence test noise
        pass

    @property
    def _mode(self) -> str:
        return self.server.mode  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            if self._mode == "never_ready":
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path.startswith("/metrics"):
            kv = getattr(self.server, "kv_usage", 0.42)  # type: ignore[attr-defined]
            preemptions = self.server.next_preemption_sample()  # type: ignore[attr-defined]
            body = (
                "# HELP vllm:kv_cache_usage_perc KV-cache usage.\n"
                "# TYPE vllm:kv_cache_usage_perc gauge\n"
                f'vllm:kv_cache_usage_perc{{model_name="fake"}} {kv}\n'
                "vllm:num_requests_waiting 0\n"
                "vllm:num_requests_running 0\n"
                f"vllm:num_preemptions_total {preemptions}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)

        mode = self._mode
        if mode == "error500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"internal error")
            return

        # Optional fixed delay before responding — lets tests prove open-loop
        # arrivals do not wait for earlier requests to complete.
        request_index = self.server.next_request_index()  # type: ignore[attr-defined]
        delay = (
            getattr(self.server, "response_delay", 0.0)
            + request_index * getattr(self.server, "response_delay_slope", 0.0)
        )
        if delay:
            time.sleep(delay)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if mode == "hang":
            time.sleep(3.0)
            return
        if mode == "malformed":
            self.wfile.write(b"data: {not valid json}\n\n")
            self.wfile.flush()
            return

        n = 0 if mode == "empty" else self.server.output_tokens  # type: ignore[attr-defined]
        for _ in range(n):
            self._send({"choices": [{"index": 0, "text": "x", "finish_reason": None}]})
            time.sleep(0.001)
        usage = {
            "prompt_tokens": self.server.prompt_tokens,  # type: ignore[attr-defined]
            "completion_tokens": n,
            "total_tokens": self.server.prompt_tokens + n,  # type: ignore[attr-defined]
        }
        self._send({"choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send(self, obj: dict) -> None:
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
        self.wfile.flush()


class _FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, addr, mode: str, output_tokens: int, prompt_tokens: int,
        kv_usage: float = 0.42, response_delay: float = 0.0,
        response_delay_slope: float = 0.0, preemption_increment: int = 0,
    ) -> None:
        super().__init__(addr, _Handler)
        self.mode = mode
        self.output_tokens = output_tokens
        self.prompt_tokens = prompt_tokens
        self.kv_usage = kv_usage
        self.response_delay = response_delay
        self.response_delay_slope = response_delay_slope
        self.preemption_increment = preemption_increment
        self._request_index = 0
        self._preemption_sample = 0
        self._counter_lock = threading.Lock()

    def next_request_index(self) -> int:
        with self._counter_lock:
            value = self._request_index
            self._request_index += 1
            return value

    def next_preemption_sample(self) -> int:
        with self._counter_lock:
            value = self._preemption_sample
            self._preemption_sample += self.preemption_increment
            return value


@contextlib.contextmanager
def serve_in_thread(
    mode: str = "normal", output_tokens: int = 8, prompt_tokens: int = 128,
    kv_usage: float = 0.42, response_delay: float = 0.0,
    response_delay_slope: float = 0.0, preemption_increment: int = 0,
) -> Iterator[str]:
    """Start the fake server on a free port in a daemon thread; yield base_url."""
    server = _FakeServer(
        ("127.0.0.1", 0), mode, output_tokens, prompt_tokens, kv_usage,
        response_delay, response_delay_slope, preemption_increment,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _emit_effective_config(prefix_caching: str) -> None:
    """Print vLLM-like startup lines to stdout so the runner can parse them."""
    sys.stdout.write(
        "INFO api_utils.py:286] non-default args: "
        "{'host': '127.0.0.1', 'model': 'fake/model', 'revision': 'deadbeef', "
        "'max_model_len': 2048, 'gpu_memory_utilization': 0.85, 'max_num_seqs': 1, "
        "'generation_config': 'vllm'}\n"
    )
    sys.stdout.write(
        "INFO core.py:123] Initializing a V1 LLM engine (v0.29.0) with config: "
        "model='fake/model', revision=deadbeef, max_seq_len=2048, "
        f"enable_prefix_caching={prefix_caching}, enable_chunked_prefill=True, "
        "kv_cache_dtype=auto, seed=0\n"
    )
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--kv-usage", type=float, default=0.42)
    parser.add_argument("--response-delay", type=float, default=0.0)
    parser.add_argument("--response-delay-slope", type=float, default=0.0)
    parser.add_argument("--preemption-increment", type=int, default=0)
    parser.add_argument(
        "--emit-effective",
        default=None,
        help="Emit a fake resolved-config log with enable_prefix_caching=True|False.",
    )
    args = parser.parse_args(argv)

    if args.mode == "startup_crash":
        sys.stderr.write("fake server: simulated startup crash\n")
        sys.stderr.flush()
        return 1
    if args.mode == "oom_crash":
        sys.stderr.write("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...\n")
        sys.stderr.flush()
        return 1

    if args.emit_effective is not None:
        _emit_effective_config(args.emit_effective)

    server = _FakeServer(
        ("127.0.0.1", args.port), args.mode, args.output_tokens, args.prompt_tokens,
        args.kv_usage, args.response_delay, args.response_delay_slope,
        args.preemption_increment,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
