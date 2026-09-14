"""Hardware/toolchain discovery: graceful when tooling is missing; parses when present.

Also proves the driver CUDA (nvidia-smi banner) and the local nvcc toolkit
version are captured as SEPARATE facts, never conflated.
"""

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
    assert info.driver_cuda_version is None
    # non-GPU fields still best-effort discovered
    assert info.cpu_count is None or info.cpu_count >= 1


def test_missing_nvcc_is_graceful(monkeypatch) -> None:
    monkeypatch.setattr(hw.shutil, "which", lambda _name: None)

    def _boom(*args, **kwargs):
        raise FileNotFoundError("no tooling")

    monkeypatch.setattr(hw.subprocess, "run", _boom)
    tc = hw.discover_toolchain()  # must not raise
    assert tc.nvcc_version is None
    assert tc.nvcc_path is None


def test_nvidia_smi_output_is_parsed(monkeypatch) -> None:
    class _Proc:
        returncode = 0

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "--query-gpu" in joined:
            return _Proc("NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 595.84\n")
        return _Proc("blah blah CUDA Version: 13.0 blah\n")

    monkeypatch.setattr(hw.subprocess, "run", _fake_run)

    info = hw.discover_hardware()
    assert info.gpu_name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert info.gpu_count == 1
    assert info.gpu_memory_total_mb == 4096
    assert info.driver_version == "595.84"
    # driver-supported CUDA comes from the banner, distinct from any toolkit.
    assert info.driver_cuda_version == "13.0"


def test_nvcc_version_and_path_parsed(monkeypatch) -> None:
    class _Proc:
        returncode = 0

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(hw.shutil, "which", lambda _name: "/usr/bin/nvcc")

    def _fake_run(cmd, **kwargs):
        return _Proc(
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Cuda compilation tools, release 12.4, V12.4.131\n"
        )

    monkeypatch.setattr(hw.subprocess, "run", _fake_run)
    # torch not installed in the test env -> torch_cuda_version None; force to isolate nvcc.
    monkeypatch.setattr(hw, "_detect_torch_cuda_version", lambda: "13.0")

    tc = hw.discover_toolchain()
    assert tc.nvcc_version == "12.4.131"
    assert tc.nvcc_path == "/usr/bin/nvcc"
    # The toolkit (12.4.131) and torch's CUDA (13.0) are distinct fields.
    assert tc.torch_cuda_version == "13.0"
    assert tc.nvcc_version != tc.torch_cuda_version
