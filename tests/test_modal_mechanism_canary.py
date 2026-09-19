"""Static safety checks for the paid Modal mechanism-canary launcher."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "fp8-instrumentation-pilot"
    / "modal_canary.py"
)


def test_modal_canary_is_parseable_and_digest_pinned() -> None:
    source = SCRIPT.read_text()
    ast.parse(source)
    assert "sha256:725769f8279dd5d50fc366fb9183a687e19ea1669ad4cdb2fa3904c29a3d35c1" in source
    assert 'add_python="3.12"' in source
    assert 'setup_dockerfile_commands=["ENTRYPOINT []"]' in source
    assert 'GPU = "A10G"' in source


def test_modal_canary_cannot_launch_registered_performance_cells() -> None:
    source = SCRIPT.read_text()
    assert "run_semantic_canary" in source
    assert "total_registered_cells" not in source
    assert "paired_blocks" not in source
    assert "require_metric_capabilities=True" in source
    assert '"async-scheduling": False' in source
    assert "enable_prefix_caching=False" in source
    assert "speculative" not in source.lower()
