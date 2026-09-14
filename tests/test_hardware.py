"""Hardware discovery: graceful when NVIDIA tooling is missing; parses when present."""

from __future__ import annotations

import inferpilot.runner.hardware as hw


def test_missing_nvidia_smi_is_graceful(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not installed")

    monkeypatch.setattr(hw.subprocess, "run", _boom)
    info = hw.discover_hardware()  # must not raise

    assert info.gpu_count == 0
    assert info.gpu_name is None
    assert info.gpu_memory_total_mb is None
    assert info.driver_version is None
    # non-GPU fields still best-effort discovered
    assert info.cpu_count is None or info.cpu_count >= 1


def test_nvidia_smi_output_is_parsed(monkeypatch) -> None:
    class _Proc:
        returncode = 0

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "--query-gpu" in joined:
            return _Proc("NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 550.120.04\n")
        return _Proc("blah blah CUDA Version: 12.4 blah\n")

    monkeypatch.setattr(hw.subprocess, "run", _fake_run)
    # Force the nvidia-smi banner path for CUDA (ignore any installed torch).
    monkeypatch.setattr(hw, "_detect_cuda_version", lambda: "12.4")

    info = hw.discover_hardware()
    assert info.gpu_name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert info.gpu_count == 1
    assert info.gpu_memory_total_mb == 4096
    assert info.driver_version == "550.120.04"
    assert info.cuda_version == "12.4"
