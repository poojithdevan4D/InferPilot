"""CLI: run one experiment from a config file.

Usage:
    python -m inferpilot.runner <config.json> [--output-dir runs]

This launches a REAL vLLM server (requires vLLM installed + a GPU). Automated
tests do not use this entrypoint; they drive the orchestrator with a fake server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import ExperimentConfig
from .orchestrator import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inferpilot.runner")
    parser.add_argument("config", type=Path, help="Path to an ExperimentConfig JSON file.")
    parser.add_argument("--output-dir", default="runs", help="Base directory for run artifacts.")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    config = ExperimentConfig.model_validate_json(args.config.read_text())
    result = run_experiment(
        config,
        output_dir=args.output_dir,
        ready_timeout_s=args.ready_timeout,
        request_timeout_s=args.request_timeout,
    )
    print(f"status={result.status.value}")
    if result.aggregates is not None:
        agg = result.aggregates
        print(
            f"requests={agg.num_requests} ok={agg.num_successful} failed={agg.num_failed} "
            f"ttft_p50={agg.ttft_p50_ms} tpot_p50={agg.tpot_p50_ms} "
            f"tok/s={agg.throughput_tokens_per_s}"
        )
    if result.failure is not None:
        print(f"failure: {result.failure.error_type}: {result.failure.message}", file=sys.stderr)
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
