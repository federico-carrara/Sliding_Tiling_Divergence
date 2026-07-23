#!/usr/bin/env python3
"""Multi-method FRC-metric CLI.

Loads 2-5 (prediction, ground-truth) pairs, computes a per-image FRC curve
for each, aggregates per-frequency mean + 95% CI per method, and serializes
the report as JSON to ``save_dir/frc_report.json``. A headline plot is written
to ``save_dir/frc_curves.png``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from analysis_pipeline.config.analysis import load_frc_config_from_args
from analysis_pipeline.frc.comparison import run_frc_analysis_multi
from analysis_pipeline.frc.plotting import plot_frc_curves
from analysis_pipeline.utils import ensure_4d, load_prediction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the FRC CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fourier Ring Correlation against ground truth, for 2-5 methods "
            "evaluated on the same image set."
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
        "--predictions",
        required=True,
        type=str,
        help="Comma-separated prediction files (.tiff/.pkl).",
    )
    parser.add_argument(
        "--ground_truths",
        required=True,
        type=str,
        help=(
            "Comma-separated ground-truth files (.tiff/.pkl), in 1:1 order "
            "with --predictions."
        ),
    )
    parser.add_argument(
        "--method_names",
        required=True,
        type=str,
        help="Comma-separated method names, e.g. 'OG,SW'.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory for the JSON FRCMultiMethodReport and headline plot.",
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
    parser.add_argument(
        "--tile_inner_sizes",
        type=str,
        default=None,
        help=(
            "Optional per-method inner tile sizes for harmonic verticals on "
            "the headline plot, e.g. '32,none'. Use 'none' for methods "
            "without a fixed seam grid (SWiTi). Plot-decoration only — does "
            "not affect numerics."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run the FRC metric pipeline from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on configuration or
        loading error).
    """
    args = parse_args()

    try:
        config = load_frc_config_from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("FRC METRIC PIPELINE")
    print("=" * 60)
    print(f"Methods:        {', '.join(config.method_names)}")
    print(f"Dataset:        {config.dataset}")
    print(f"Predictions:    {config.predictions}")
    print(f"Ground truths:  {config.ground_truths}")
    print(f"Save dir:       {config.save_dir}")
    print(f"Apply window:   {config.frc.apply_window}")
    print(f"Channel:        {config.frc.channel}")
    if config.tile_inner_sizes is not None:
        print(f"Tile inner S:   {config.tile_inner_sizes}")
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

    print("\nLoading predictions and ground truths...")
    predictions_list: list[np.ndarray] = []
    ground_truths_list: list[np.ndarray] = []
    for pred_file, gt_file, method_name in zip(
        config.predictions,
        config.ground_truths,
        config.method_names,
        strict=True,
    ):
        try:
            predictions_list.append(_load(pred_file, f"{method_name} (pred)"))
            ground_truths_list.append(_load(gt_file, f"{method_name} (GT)"))
        except Exception as e:
            print(f"Error loading {method_name}: {e}", file=sys.stderr)
            return 1

    print("\nRunning FRC analysis...")
    report = run_frc_analysis_multi(
        predictions_list=predictions_list,
        ground_truths_list=ground_truths_list,
        method_names=config.method_names,
        save_dir=config.save_dir,
        channel=config.frc.channel,
        apply_window=config.frc.apply_window,
    )

    tile_inner_sizes_map = None
    if config.tile_inner_sizes is not None:
        tile_inner_sizes_map = dict(
            zip(config.method_names, config.tile_inner_sizes, strict=True)
        )

    plot_path = config.save_dir / "frc_curves.png"
    plot_frc_curves(
        list(report.methods.values()), tile_inner_sizes_map, save_path=plot_path
    )
    print(f"\nHeadline plot saved to: {plot_path}")

    print(f"\nDone. Results saved to: {config.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
