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

    def __init__(self, addr, mode: str, output_tokens: int, prompt_tokens: int) -> None:
        super().__init__(addr, _Handler)
        self.mode = mode
        self.output_tokens = output_tokens
        self.prompt_tokens = prompt_tokens


@contextlib.contextmanager
def serve_in_thread(
    mode: str = "normal", output_tokens: int = 8, prompt_tokens: int = 128
) -> Iterator[str]:
    """Start the fake server on a free port in a daemon thread; yield base_url."""
    server = _FakeServer(("127.0.0.1", 0), mode, output_tokens, prompt_tokens)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    if args.mode == "startup_crash":
        sys.stderr.write("fake server: simulated startup crash\n")
        sys.stderr.flush()
        return 1
    if args.mode == "oom_crash":
        sys.stderr.write("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...\n")
        sys.stderr.flush()
        return 1

    server = _FakeServer(("127.0.0.1", args.port), args.mode, args.output_tokens, args.prompt_tokens)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
