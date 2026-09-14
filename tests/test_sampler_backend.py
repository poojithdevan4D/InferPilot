"""Sampler backend env mapping, non-mutating child env, provenance round-trips,
and clear failure on unsupported schema versions."""

from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

from inferpilot import (
    EngineConfig,
    EnvironmentMetadata,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ToolchainInfo,
    WorkloadSpec,
)
from inferpilot.runner.server import ManagedServer, build_server_env, find_free_port

_SAMPLER_VAR = "VLLM_USE_FLASHINFER_SAMPLER"


def _engine(backend: str) -> EngineConfig:
    return EngineConfig(model="org/model", sampler_backend=backend)


def _config(backend: str = "pytorch") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="exp-s",
        name="s",
        engine=_engine(backend),
        workload=WorkloadSpec(
            name="w", num_requests=1, prompt_tokens=8, output_tokens=8, max_concurrency=1
        ),
    )


# --- env mapping ------------------------------------------------------------ #


def test_build_server_env_mapping() -> None:
    assert build_server_env(_engine("auto")) == {}
    assert build_server_env(_engine("pytorch")) == {_SAMPLER_VAR: "0"}
    assert build_server_env(_engine("flashinfer")) == {_SAMPLER_VAR: "1"}


def test_managed_server_applies_env_without_mutating_parent(tmp_path) -> None:
    # Precondition: the var is not already set in this process.
    assert _SAMPLER_VAR not in os.environ
    parent_snapshot = dict(os.environ)

    cmd = [
        sys.executable,
        "-c",
        f"import os; print('VAL=' + os.environ.get('{_SAMPLER_VAR}', 'UNSET'))",
    ]
    server = ManagedServer(
        cmd,
        host="127.0.0.1",
        port=find_free_port(),
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        env={_SAMPLER_VAR: "0"},
    )
    server.start()
    server._process.wait(timeout=10)
    server.stop()

    assert "VAL=0" in (tmp_path / "out.log").read_text()  # child saw the override
    assert _SAMPLER_VAR not in os.environ  # parent never mutated
    assert dict(os.environ) == parent_snapshot


# --- provenance round trips ------------------------------------------------- #


def test_sampler_backend_survives_config_roundtrip() -> None:
    cfg = _config("pytorch")
    restored = ExperimentConfig.model_validate_json(cfg.model_dump_json())
    assert restored.engine.sampler_backend == "pytorch"
    assert restored == cfg


def test_effective_sampler_and_overrides_survive_result_roundtrip() -> None:
    env = EnvironmentMetadata(
        effective_sampler_backend="pytorch",
        runtime_overrides={_SAMPLER_VAR: "0"},
    )
    result = ExperimentResult(
        config=_config("pytorch"),
        environment=env,
        status=ExperimentStatus.OOM,
        failure={"status": "oom", "error_type": "CudaOOM", "message": "x"},
    )
    restored = ExperimentResult.model_validate_json(result.model_dump_json())
    assert restored.environment.effective_sampler_backend == "pytorch"
    assert restored.environment.runtime_overrides == {_SAMPLER_VAR: "0"}
    assert restored == result


def test_cuda_runtime_and_toolkit_stay_separate() -> None:
    env = EnvironmentMetadata(
        toolchain=ToolchainInfo(
            torch_cuda_version="13.0", nvcc_version="12.4.131", nvcc_path="/usr/bin/nvcc"
        ),
    )
    env.hardware.driver_cuda_version = "13.0"
    restored = EnvironmentMetadata.model_validate_json(env.model_dump_json())
    # Three distinct CUDA facts survive as separate fields.
    assert restored.hardware.driver_cuda_version == "13.0"
    assert restored.toolchain.torch_cuda_version == "13.0"
    assert restored.toolchain.nvcc_version == "12.4.131"
    assert restored.toolchain.nvcc_version != restored.toolchain.torch_cuda_version


# --- schema version gating -------------------------------------------------- #


def test_old_schema_version_fails_clearly_config() -> None:
    raw = _config().model_dump()
    raw["schema_version"] = "0.2.0"
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        ExperimentConfig.model_validate(raw)


def test_old_schema_version_fails_clearly_result() -> None:
    result = ExperimentResult(
        config=_config(),
        environment=EnvironmentMetadata(),
        status=ExperimentStatus.OOM,
        failure={"status": "oom", "error_type": "CudaOOM", "message": "x"},
    )
    raw = result.model_dump()
    raw["schema_version"] = "0.2.0"
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        ExperimentResult.model_validate(raw)
