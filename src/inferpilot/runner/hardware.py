"""One-time hardware + toolchain discovery (best-effort, non-fatal).

Captures a single snapshot at run start. This is NOT continuous GPU polling — it
runs once. Every probe is wrapped so that missing tooling (no ``nvidia-smi``, no
``nvcc``, no ``/proc``) degrades to ``None`` fields rather than aborting the
benchmark.

CUDA is recorded as three DISTINCT facts, never conflated:
* ``HardwareInfo.driver_cuda_version`` — max CUDA the driver supports (nvidia-smi banner),
* ``ToolchainInfo.torch_cuda_version`` — CUDA PyTorch was built against,
* ``ToolchainInfo.nvcc_version`` — the local nvcc toolkit that JIT builds would use.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from importlib import metadata as _md
from typing import Optional

from ..environment import HardwareInfo, ToolchainInfo


def _run(cmd: list[str], timeout: float = 10.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _query_gpu(fields: str) -> Optional[list[list[str]]]:
    out = _run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    if out is None:
        return None
    rows = [
        [c.strip() for c in line.split(",")]
        for line in out.splitlines()
        if line.strip()
    ]
    return rows or None


def _detect_driver_cuda_version() -> Optional[str]:
    """Max CUDA supported by the driver, from the nvidia-smi banner."""
    out = _run(["nvidia-smi"])
    if out:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if match:
            return match.group(1)
    return None


def _detect_torch_cuda_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        if getattr(torch, "version", None) and torch.version.cuda:
            return str(torch.version.cuda)
    except Exception:
        pass
    return None


def _detect_nvcc() -> tuple[Optional[str], Optional[str]]:
    """Return (nvcc_version, nvcc_path); either may be None."""
    path = shutil.which("nvcc")
    if not path:
        return None, None
    out = _run([path, "--version"])
    version = None
    if out:
        # e.g. "Cuda compilation tools, release 12.4, V12.4.131"
        match = re.search(r"release\s+[0-9.]+,\s*V([0-9.]+)", out)
        if match:
            version = match.group(1)
    return version, path


def _detect_flashinfer_version() -> Optional[str]:
    try:
        return _md.version("flashinfer")
    except _md.PackageNotFoundError:
        pass
    except Exception:
        pass
    try:
        import flashinfer  # type: ignore

        return getattr(flashinfer, "__version__", None)
    except Exception:
        return None


def _detect_cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _detect_system_ram_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def discover_hardware() -> HardwareInfo:
    """Return a best-effort hardware snapshot; never raises."""
    gpu_name: Optional[str] = None
    gpu_count = 0
    vram_mb: Optional[int] = None
    driver: Optional[str] = None

    try:
        rows = _query_gpu("name,memory.total,driver_version")
        if rows:
            gpu_count = len(rows)
            first = rows[0]
            gpu_name = first[0] or None
            try:
                vram_mb = int(float(first[1]))
            except (ValueError, IndexError):
                vram_mb = None
            driver = first[2] if len(first) > 2 and first[2] else None
    except Exception:
        pass

    driver_cuda = None
    try:
        driver_cuda = _detect_driver_cuda_version()
    except Exception:
        pass

    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_memory_total_mb=vram_mb,
        driver_version=driver,
        driver_cuda_version=driver_cuda,
        cpu_model=_detect_cpu_model(),
        cpu_count=os.cpu_count(),
        system_ram_mb=_detect_system_ram_mb(),
    )


def discover_toolchain() -> ToolchainInfo:
    """Return a best-effort CUDA/build toolchain snapshot; never raises."""
    try:
        nvcc_version, nvcc_path = _detect_nvcc()
    except Exception:
        nvcc_version, nvcc_path = None, None
    return ToolchainInfo(
        torch_cuda_version=_detect_torch_cuda_version(),
        nvcc_version=nvcc_version,
        nvcc_path=nvcc_path,
        flashinfer_version=_detect_flashinfer_version(),
    )
