"""vLLM server-process adapter (process management only).

Responsibilities, and nothing else:

* build the server argv **without a shell** (no ``shell=True``, no string parsing),
* pick a free local port,
* capture stdout/stderr to artifact files,
* poll a readiness endpoint with a timeout,
* stop gracefully (SIGTERM/terminate) and force-kill only after a timeout,
* always clean up, even on exception / interruption (context manager).

This module does not know about workloads, measurements, or results. It also
does not import vLLM — it launches it as a subprocess so automated tests need no
GPU, no model, and no vLLM install.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import monotonic, sleep
from types import TracebackType
from typing import IO, Mapping, Optional

from ..config import EngineConfig


class ServerError(RuntimeError):
    """Base class for server lifecycle problems."""


class ServerStartupError(ServerError):
    """The server process exited before becoming ready."""


class ServerReadinessTimeout(ServerError):
    """The server did not become ready within the allotted time."""


def find_free_port(host: str = "127.0.0.1") -> int:
    """Return a currently-free TCP port on ``host`` (best-effort, race-free enough)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return sock.getsockname()[1]


def default_vllm_executable(python_executable: str = sys.executable) -> str:
    """Resolve the `vllm` console script that lives next to the interpreter."""
    return str(Path(python_executable).with_name("vllm"))


def _bool_flag(cmd: list[str], value: Optional[bool], name: str) -> None:
    """Map a tri-state boolean to an explicit flag.

    True -> ``--name``; False -> ``--no-name``; None -> omit (engine default).
    Omission and False are NOT the same: omission accepts vLLM's default (which
    may be True), whereas False forces the feature off.
    """
    if value is True:
        cmd.append(f"--{name}")
    elif value is False:
        cmd.append(f"--no-{name}")


def build_vllm_command(
    engine: EngineConfig,
    port: int,
    host: str = "127.0.0.1",
    vllm_executable: Optional[str] = None,
) -> list[str]:
    """Build the argv for the supported ``vllm serve`` CLI (no shell).

    Uses the non-deprecated ``vllm serve <model>`` form. Returned as a list so it
    is passed to ``subprocess`` with ``shell=False``.
    """
    cmd: list[str] = [
        vllm_executable or default_vllm_executable(),
        "serve",
        engine.model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        engine.dtype,
        "--gpu-memory-utilization",
        str(engine.gpu_memory_utilization),
        "--kv-cache-dtype",
        engine.kv_cache_dtype,
        "--generation-config",
        engine.generation_config,
        "--shutdown-timeout",
        "10",
    ]
    if engine.revision is not None:
        cmd += ["--revision", engine.revision]
    if engine.max_model_len is not None:
        cmd += ["--max-model-len", str(engine.max_model_len)]
    if engine.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(engine.max_num_seqs)]
    if engine.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(engine.max_num_batched_tokens)]
    # Explicit tri-state mapping: True/False/None are all distinct.
    _bool_flag(cmd, engine.enable_prefix_caching, "enable-prefix-caching")
    _bool_flag(cmd, engine.enable_chunked_prefill, "enable-chunked-prefill")
    for key, value in engine.extra_args.items():
        cmd += [f"--{key}", str(value)]
    return cmd


def build_server_env(engine: EngineConfig) -> dict[str, str]:
    """Return the non-secret child-process env overrides implied by the config.

    Only the sampler backend is mapped today. ``auto`` contributes nothing (the
    engine default is left in place).
    """
    if engine.sampler_backend == "pytorch":
        return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    if engine.sampler_backend == "flashinfer":
        return {"VLLM_USE_FLASHINFER_SAMPLER": "1"}
    return {}


class ManagedServer:
    """Context-managed subprocess with health polling and guaranteed cleanup.

    Named "vLLM adapter" but mechanically it manages any HTTP subprocess exposing
    a health endpoint, which is what makes it testable without vLLM.
    """

    def __init__(
        self,
        command: list[str],
        *,
        host: str,
        port: int,
        stdout_path: Path,
        stderr_path: Path,
        health_path: str = "/health",
        ready_timeout_s: float = 300.0,
        poll_interval_s: float = 0.5,
        terminate_timeout_s: float = 15.0,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.command = command
        self.host = host
        self.port = port
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.health_path = health_path
        self.ready_timeout_s = ready_timeout_s
        self.poll_interval_s = poll_interval_s
        self.terminate_timeout_s = terminate_timeout_s
        # Non-secret runtime overrides applied on top of a COPY of the parent
        # environment. The parent os.environ is never mutated.
        self.env_overrides: dict[str, str] = dict(env) if env else {}

        self._process: Optional[subprocess.Popen[bytes]] = None
        self._stdout_fh: Optional[IO[bytes]] = None
        self._stderr_fh: Optional[IO[bytes]] = None

        # Teardown-phase boundary (set when stop() signals the child).
        self._teardown_started: bool = False
        self._teardown_stdout_offset: Optional[int] = None
        self._teardown_stderr_offset: Optional[int] = None

    # -- lifecycle -------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_fh = open(self.stdout_path, "wb")
        self._stderr_fh = open(self.stderr_path, "wb")
        # Build the child environment from a copy of the parent's, then apply only
        # the configured overrides. os.environ itself is left untouched.
        child_env = os.environ.copy()
        child_env.update(self.env_overrides)
        # shell=False (list argv): no shell interpretation of any argument.
        self._process = subprocess.Popen(
            self.command,
            stdout=self._stdout_fh,
            stderr=self._stderr_fh,
            shell=False,
            env=child_env,
        )

    def wait_until_ready(self) -> None:
        assert self._process is not None, "start() must be called first"
        deadline = monotonic() + self.ready_timeout_s
        health_url = f"{self.base_url}{self.health_path}"
        while monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise ServerStartupError(
                    f"server exited with code {exit_code} before readiness; "
                    f"see {self.stderr_path}"
                )
            if self._probe(health_url):
                return
            sleep(self.poll_interval_s)
        raise ServerReadinessTimeout(
            f"server not ready within {self.ready_timeout_s}s at {health_url}"
        )

    @staticmethod
    def _probe(url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310 (localhost)
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _log_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def stop(self) -> None:
        """Terminate gracefully; escalate to kill after ``terminate_timeout_s``.

        ``terminate_timeout_s`` is the (nonzero) graceful-shutdown window granted
        before SIGKILL. The byte offsets of the logs at the moment of SIGTERM are
        recorded so downstream code can distinguish teardown-phase log output
        (written after this point) from startup/measurement output.
        """
        # Mark the teardown boundary before signalling the child.
        self._teardown_started = True
        self._teardown_stdout_offset = self._log_size(self.stdout_path)
        self._teardown_stderr_offset = self._log_size(self.stderr_path)

        proc = self._process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.terminate_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=self.terminate_timeout_s)
                except subprocess.TimeoutExpired:
                    pass
        for fh in (self._stdout_fh, self._stderr_fh):
            if fh is not None and not fh.closed:
                fh.close()
        self._stdout_fh = None
        self._stderr_fh = None

    def read_stderr(self) -> str:
        try:
            return self.stderr_path.read_text(errors="replace")
        except OSError:
            return ""

    def read_stdout(self) -> str:
        try:
            return self.stdout_path.read_text(errors="replace")
        except OSError:
            return ""

    def classify_log_errors(self) -> dict:
        """Split ERROR/traceback lines into pre-teardown vs teardown phases.

        Errors written before the SIGTERM boundary are attributed to
        startup/measurement; errors after it are intentional-teardown noise
        (e.g. vLLM's AsyncLLM output_handler reacting to SIGTERM). Logs are never
        suppressed — this only classifies them.
        """
        pattern = re.compile(r"\bERROR\b|Traceback \(most recent call last\)|EngineDeadError")
        result = {"pre_teardown": [], "teardown_count": 0, "teardown_started": self._teardown_started}
        for path, offset in (
            (self.stdout_path, self._teardown_stdout_offset),
            (self.stderr_path, self._teardown_stderr_offset),
        ):
            try:
                data = path.read_bytes()
            except OSError:
                continue
            split = offset if offset is not None else len(data)
            pre = data[:split].decode(errors="replace")
            post = data[split:].decode(errors="replace")
            for line in pre.splitlines():
                if pattern.search(line):
                    result["pre_teardown"].append(line.strip()[:300])
            for line in post.splitlines():
                if pattern.search(line):
                    result["teardown_count"] += 1
        return result

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "ManagedServer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # Always clean up — normal exit, exception, or KeyboardInterrupt.
        self.stop()
