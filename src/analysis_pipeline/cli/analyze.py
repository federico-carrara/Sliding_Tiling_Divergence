#!/usr/bin/env python3
"""
Multi-method experiment analysis CLI.

Generalized experiment analysis pipeline supporting 2-5 input predictions.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from ..config.settings import load_config_from_args
from ..utils import load_prediction, ensure_4d, remove_padding
from ..core.analysis import run_gradient_analysis_multi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-method experiment analysis (compare 2-5 predictions)"
    )

    # Required arguments
    parser.add_argument(
        "--model_name",
        required=True,
        choices=["usplit", "microsplit", "HDN"],
        help="Model name",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--predictions",
        required=True,
        type=str,
        help="Comma-separated list of prediction files (.tiff/.pkl)",
    )
    parser.add_argument(
        "--method_names",
        required=True,
        type=str,
        help="Comma-separated method names, e.g., 'OG,SW,Method3'",
    )

    # Optional arguments
    parser.add_argument(
        "--save_dir", required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--inner_tile_size",
        type=parse_comma_separated,
        default=[32],
        help="Inner tile size(s)",
    )
    parser.add_argument(
        "--bins", type=int, default=100, help="Number of histogram bins"
    )
    parser.add_argument(
        "--channel", type=int, default=0, help="Channel to analyze (0-indexed)"
    )
    parser.add_argument(
        "--padding",
        type=parse_comma_separated,
        default=[48],
        help="Padding to remove from predictions",
    )

    # Analysis flags
    parser.add_argument(
        "--gradient_analysis",
        action="store_true",
        default=True,
        help="Run gradient analysis",
    )
    parser.add_argument(
        "--qualitative_analysis",
        action="store_true",
        default=True,
        help="Run qualitative analysis",
    )
    parser.add_argument(
        "--skip_gradient_analysis",
        action="store_true",
        help="Skip gradient analysis",
    )

    return parser.parse_args()


def parse_comma_separated(value: str) -> list:
    """
    Parse comma-separated integers.

    Supports formats:
    - Single value: "32" → [32]
    - 2D tile: "32,32" → [32, 32]
    - 3D tile: "4,32,32" → [4, 32, 32]
    - Per-method: "32;64;32" → [[32], [64], [32]]
    - Per-method 3D: "4,32,32;5,32,32" → [[4,32,32], [5,32,32]]
    """
    # Check if there are semicolons (per-method specification)
    if ";" in value:
        methods = value.split(";")
        return [
            [int(v) for v in method.split(",")]
            for method in methods
        ]

    # Single tile specification for all methods
    try:
        result = [int(v) for v in value.split(",")]
        # If single value, keep as [32], otherwise keep as is [4,32,32]
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid comma-separated values: {value}. "
            "Use format: '32' or '32,32' or '4,32,32' or '32;64;32'"
        )


def main():
    """Main entry point for analysis CLI."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config_from_args(args)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"MULTI-METHOD ANALYSIS PIPELINE")
    print(f"{'=' * 60}")
    print(f"Methods: {', '.join(config.method_names)}")
    print(f"Predictions: {config.predictions}")
    print(f"Dataset: {config.dataset}")
    print(f"{'=' * 60}\n")

    # Create output directory
    config.save_dir.mkdir(parents=True, exist_ok=True)

    # Load all predictions
    print("Loading predictions...")
    predictions_list = []
    for pred_file, method_name, pad in zip(
        config.predictions, config.method_names, config.padding
    ):
        print(f"  Loading {method_name}: {pred_file}")
        try:
            pred = load_prediction(pred_file)

            # Handle 6D arrays (squeeze first dimension)
            if len(pred.shape) == 6:
                pred = np.squeeze(pred, axis=0)

            pred = ensure_4d(pred)
            print(f"    Padding to remove: {pad}")
            pred = remove_padding(pred, pad)

            predictions_list.append(pred)
            print(f"    Final shape: {pred.shape}")
        except Exception as e:
            print(f"❌ Error loading {pred_file}: {e}")
            sys.exit(1)

    # Run gradient analysis
    if config.run_gradient_analysis:
        print("\n" + "-" * 60)

        if config.gradient.channel == 2:
            # Analyze each channel separately
            for channel in range(config.gradient.channel):
                print(f"\nAnalyzing Channel {channel}...")
                run_gradient_analysis_multi(
                    predictions_list=predictions_list,
                    method_names=config.method_names,
                    save_dir=config.save_dir / f"Channel_{channel}",
                    inner_tile_size=config.gradient.inner_tile_size,
                    bins=config.gradient.bins,
                    channel=channel,
                    border=0,
                )
        else:
            run_gradient_analysis_multi(
                predictions_list=predictions_list,
                method_names=config.method_names,
                save_dir=config.save_dir,
                inner_tile_size=config.gradient.inner_tile_size,
                bins=config.gradient.bins,
                channel=config.gradient.channel,
                border=0,
            )

    print(f"\n✅ Analysis complete! Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
