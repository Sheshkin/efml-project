#!/usr/bin/env python3
import argparse
from types import SimpleNamespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click runner for all future inference-acceleration experiments."
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory with command YAML configs (default: configs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for logs/tables/plots/results.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop pipeline on first failed experiment.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run optional QAT and TVM experiments after the six core experiments.",
    )
    args = parser.parse_args()

    try:
        from run_experiment import run_future_all
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependency")
        raise SystemExit(
            f"Missing dependency: {missing}. Install project requirements first: "
            "pip install -r requirements.txt"
        ) from exc

    run_future_all(
        SimpleNamespace(
            fail_fast=args.fail_fast,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            include_optional=args.include_optional,
        )
    )
