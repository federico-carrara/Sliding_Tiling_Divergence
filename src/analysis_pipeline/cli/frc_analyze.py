#!/usr/bin/env python3
"""Single-method FRC-metric CLI.

Loads one (prediction, ground-truth) pair, computes a per-image FRC curve
for each image, aggregates per-frequency mean + 95% CI, and pickles the
report to ``save_dir/<method_name>_frc_report.pkl``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..config.analysis import load_frc_single_config_from_args
from ..frc.analysis import run_frc_analysis
from ..utils import ensure_4d, load_prediction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-method FRC CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fourier Ring Correlation against ground truth for a single "
            "method on its image set."
        )
    )

    parser.add_argument(
        "--model_name",
        required=True,
        choices=["usplit", "microsplit", "HDN"],
        help="Model name (used in the report header).",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--prediction",
        required=True,
        type=str,
        help="Prediction file (.tiff/.pkl).",
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        type=str,
        help="Ground-truth file (.tiff/.pkl) paired with --prediction.",
    )
    parser.add_argument(
        "--method_name",
        required=True,
        type=str,
        help="Display name for the method (used in logs and the pickle filename).",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory for the pickled FRCMethodReport.",
    )

    parser.add_argument(
        "--channel", type=int, default=0, help="Channel index to analyze (0-based)."
    )
    parser.add_argument(
        "--no_window",
        dest="apply_window",
        action="store_false",
        default=True,
        help=(
            "Disable the 2-D Hamming window before FFT. Off by default; "
            "enable only for sanity tests (real images need windowing)."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the FRC metric pipeline for a single method from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on configuration or
        loading error).
    """
    args = parse_args()

    try:
        config = load_frc_single_config_from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    prediction_file = config.predictions[0]
    ground_truth_file = config.ground_truths[0]
    method_name = config.method_names[0]

    print("=" * 60)
    print("FRC METRIC PIPELINE (single method)")
    print("=" * 60)
    print(f"Method:         {method_name}")
    print(f"Dataset:        {config.dataset}")
    print(f"Prediction:     {prediction_file}")
    print(f"Ground truth:   {ground_truth_file}")
    print(f"Save dir:       {config.save_dir}")
    print(f"Apply window:   {config.frc.apply_window}")
    print(f"Channel:        {config.frc.channel}")
    print("=" * 60)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    def _load(path: str, label: str) -> np.ndarray:
        print(f"  {label}: {path}")
        arr = load_prediction(path)
        if arr.ndim == 6:
            arr = np.squeeze(arr, axis=0)
        arr = ensure_4d(arr)
        print(f"    shape: {arr.shape}")
        return arr

    print("\nLoading prediction and ground truth...")
    try:
        prediction = _load(prediction_file, f"{method_name} (pred)")
        ground_truth = _load(ground_truth_file, f"{method_name} (GT)")
    except Exception as e:
        print(f"Error loading {method_name}: {e}", file=sys.stderr)
        return 1

    print("\nRunning FRC analysis...")
    run_frc_analysis(
        predictions=prediction,
        ground_truths=ground_truth,
        save_dir=config.save_dir,
        method_name=method_name,
        channel=config.frc.channel,
        apply_window=config.frc.apply_window,
    )

    print(f"\nDone. Results saved to: {config.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
