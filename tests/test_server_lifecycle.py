"""Server-adapter lifecycle tests: command building + guaranteed cleanup.

Uses benign subprocesses (no vLLM) to prove readiness-failure, force-kill, and
interruption cleanup paths.
"""

from __future__ import annotations

import signal
import sys
from time import monotonic, sleep

import pytest

from inferpilot.config import EngineConfig
from inferpilot.runner.server import (
    ManagedServer,
    ServerReadinessTimeout,
    build_vllm_command,
    find_free_port,
)

SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]
# Prints READY *after* installing SIG_IGN, so the test can avoid the startup race
# where SIGTERM would arrive before the handler is set.
SIGTERM_IGNORER = [
    sys.executable,
    "-c",
    "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "sys.stdout.write('READY\\n'); sys.stdout.flush(); time.sleep(30)",
]


def _wait_for_marker(path, marker: str, timeout: float = 5.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            if marker in path.read_text():
                return
        except OSError:
            pass
        sleep(0.02)
    raise AssertionError(f"marker {marker!r} not seen in {path} within {timeout}s")


def _server(cmd, tmp_path, **kw) -> ManagedServer:
    return ManagedServer(
        cmd,
        host="127.0.0.1",
        port=find_free_port(),
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        **kw,
    )


# --- command building ------------------------------------------------------- #


def test_build_vllm_command_is_shell_free_list() -> None:
    engine = EngineConfig(model="Qwen/Qwen2.5-0.5B-Instruct", revision="abc123", max_num_seqs=1)
    cmd = build_vllm_command(engine, port=8123, host="127.0.0.1")
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--model" in cmd and "Qwen/Qwen2.5-0.5B-Instruct" in cmd
    assert "--revision" in cmd and "abc123" in cmd
    assert "8123" in cmd


def test_build_vllm_command_omits_revision_when_absent() -> None:
    cmd = build_vllm_command(EngineConfig(model="m"), port=1, host="127.0.0.1")
    assert "--revision" not in cmd


def test_find_free_port_returns_usable_port() -> None:
    port = find_free_port()
    assert isinstance(port, int) and 1024 <= port <= 65535


# --- lifecycle cleanup ------------------------------------------------------ #


def test_readiness_failure_cleans_up(tmp_path) -> None:
    """A process that never serves /health times out AND is terminated."""
    server = _server(SLEEPER, tmp_path, ready_timeout_s=1.0, poll_interval_s=0.1)
    with pytest.raises(ServerReadinessTimeout):
        with server:
            server.wait_until_ready()
    assert server._process is not None
    assert server._process.poll() is not None  # process was cleaned up


def test_interruption_cleans_up(tmp_path) -> None:
    server = _server(SLEEPER, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with server:
            assert server._process.poll() is None  # running
            raise KeyboardInterrupt
    assert server._process.poll() is not None  # killed on __exit__


def test_graceful_stop(tmp_path) -> None:
    server = _server(SLEEPER, tmp_path)
    server.start()
    assert server._process.poll() is None
    server.stop()
    assert server._process.poll() is not None


def test_force_kill_after_sigterm_ignored(tmp_path) -> None:
    """A process ignoring SIGTERM is escalated to SIGKILL after the timeout."""
    server = _server(SIGTERM_IGNORER, tmp_path, terminate_timeout_s=1.0)
    server.start()
    _wait_for_marker(tmp_path / "out.log", "READY")  # handler installed
    assert server._process.poll() is None
    server.stop()
    assert server._process.poll() is not None
    assert server._process.returncode == -signal.SIGKILL
