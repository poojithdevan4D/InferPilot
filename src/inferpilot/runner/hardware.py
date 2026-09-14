"""One-time hardware discovery (best-effort, non-fatal).

Captures a single snapshot of GPU / CPU / RAM at run start. This is NOT
continuous GPU polling — it runs once. Every probe is wrapped so that missing
tooling (no ``nvidia-smi``, no ``/proc``) degrades to ``None`` fields rather than
aborting the benchmark.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional

from ..environment import HardwareInfo


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


def _detect_cuda_version() -> Optional[str]:
    # Prefer the toolkit version torch was built against, if torch is installed.
    try:
        import torch  # type: ignore

        if getattr(torch, "version", None) and torch.version.cuda:
            return str(torch.version.cuda)
    except Exception:
        pass
    # Fall back to the driver's reported CUDA version from nvidia-smi's banner.
    out = _run(["nvidia-smi"])
    if out:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        if match:
            return match.group(1)
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
        # Any unexpected shape from nvidia-smi must not abort the benchmark.
        pass

    cuda = None
    try:
        cuda = _detect_cuda_version()
    except Exception:
        pass

    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_memory_total_mb=vram_mb,
        cuda_version=cuda,
        driver_version=driver,
        cpu_model=_detect_cpu_model(),
        cpu_count=os.cpu_count(),
        system_ram_mb=_detect_system_ram_mb(),
    )
