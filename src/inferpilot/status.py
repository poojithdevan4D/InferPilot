"""Experiment lifecycle status and failure records.

InferPilot must faithfully record *why* an experiment did not produce metrics.
An out-of-memory experiment is a first-class outcome (the RTX 3050 has only 4 GB),
not an exception to be swallowed. These records are what let the core loop
reject a configuration on real evidence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import Field

from ._base import SchemaModel


class ExperimentStatus(str, Enum):
    """Terminal or in-flight status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OOM = "oom"
    TIMEOUT = "timeout"

    @property
    def is_terminal(self) -> bool:
        return self not in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING)

    @property
    def is_success(self) -> bool:
        return self is ExperimentStatus.COMPLETED


class FailureRecord(SchemaModel):
    """Structured description of an experiment failure.

    Present on a result whenever ``status`` is not ``COMPLETED``. ``error_type``
    is a short machine-friendly tag (e.g. ``"CudaOOM"``); ``message`` is the
    human-readable summary; ``traceback`` is optional raw detail.
    """

    status: ExperimentStatus = Field(
        description="The failing status this record explains (e.g. oom, failed, timeout).",
    )
    error_type: str = Field(
        description="Short machine-friendly error tag, e.g. 'CudaOOM' or 'EngineStartupError'.",
    )
    message: str = Field(description="Human-readable failure summary.")
    traceback: Optional[str] = Field(
        default=None, description="Optional raw traceback or engine log excerpt."
    )
    occurred_at: Optional[datetime] = Field(
        default=None, description="When the failure was observed (timezone-aware recommended)."
    )
