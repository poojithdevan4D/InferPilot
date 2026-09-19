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


def test_build_vllm_command_uses_supported_serve_form() -> None:
    engine = EngineConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct", revision="abc123", max_num_seqs=1
    )
    cmd = build_vllm_command(engine, port=8123, host="127.0.0.1")
    assert isinstance(cmd, list)
    # Supported non-deprecated CLI: `vllm serve <model>` (no `-m api_server`).
    assert cmd[0].endswith("vllm")
    assert cmd[1] == "serve"
    assert cmd[2] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "vllm.entrypoints.openai.api_server" not in cmd
    assert "--revision" in cmd and "abc123" in cmd
    assert "8123" in cmd
    assert cmd[cmd.index("--generation-config") + 1] == "vllm"
    assert float(cmd[cmd.index("--shutdown-timeout") + 1]) > 0


def test_build_vllm_command_omits_revision_when_absent() -> None:
    cmd = build_vllm_command(EngineConfig(model="m"), port=1, host="127.0.0.1")
    assert "--revision" not in cmd


def test_build_vllm_command_maps_boolean_extra_args_to_explicit_flags() -> None:
    cmd = build_vllm_command(
        EngineConfig(
            model="m",
            extra_args={"async-scheduling": False, "some-feature": True},
        ),
        port=1,
    )
    assert "--no-async-scheduling" in cmd
    assert "--some-feature" in cmd
    assert "False" not in cmd
    assert "True" not in cmd


# --- explicit tri-state boolean mapping (true / false / default) ------------ #


def _cmd(**engine_kwargs) -> list[str]:
    return build_vllm_command(EngineConfig(model="m", **engine_kwargs), port=1)


def test_prefix_caching_true_emits_enable_flag() -> None:
    cmd = _cmd(enable_prefix_caching=True)
    assert "--enable-prefix-caching" in cmd
    assert "--no-enable-prefix-caching" not in cmd


def test_prefix_caching_false_emits_no_flag() -> None:
    cmd = _cmd(enable_prefix_caching=False)
    assert "--no-enable-prefix-caching" in cmd
    assert "--enable-prefix-caching" not in cmd


def test_prefix_caching_default_omits_both() -> None:
    cmd = _cmd()  # None => engine default
    assert "--enable-prefix-caching" not in cmd
    assert "--no-enable-prefix-caching" not in cmd


def test_chunked_prefill_tristate() -> None:
    assert "--enable-chunked-prefill" in _cmd(enable_chunked_prefill=True)
    assert "--no-enable-chunked-prefill" in _cmd(enable_chunked_prefill=False)
    default = _cmd()
    assert "--enable-chunked-prefill" not in default
    assert "--no-enable-chunked-prefill" not in default


def test_find_free_port_returns_usable_port() -> None:
    port = find_free_port()
    assert isinstance(port, int) and 1024 <= port <= 65535


def test_classify_log_errors_splits_at_teardown_boundary(tmp_path) -> None:
    server = _server(SLEEPER, tmp_path)  # not started; drive classify directly
    startup = "INFO ok\nERROR startup boom\n"
    (tmp_path / "out.log").write_text(startup)
    (tmp_path / "err.log").write_text("")
    # Teardown boundary is the byte offset at SIGTERM time.
    server._teardown_stdout_offset = len(startup)
    server._teardown_stderr_offset = 0
    server._teardown_started = True
    with open(tmp_path / "out.log", "a") as fh:
        fh.write("ERROR teardown noise\n")

    result = server.classify_log_errors()
    assert any("startup boom" in line for line in result["pre_teardown"])
    assert result["teardown_count"] == 1  # teardown error not counted as pre-teardown


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
