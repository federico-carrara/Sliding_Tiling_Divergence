#!/usr/bin/env python3
"""Single-method per-tile-metric CLI.

Loads one prediction file, runs the per-tile two-sample test on its image
set, and serializes the resulting ``MethodReport`` as JSON to
``save_dir/<method_name>_per_tile_report.json``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from analysis_pipeline.config.analysis import load_gradient_test_single_config_from_args
from analysis_pipeline.gradient_test.analysis import run_gradient_analysis
from analysis_pipeline.utils import ensure_4d, load_prediction


def parse_comma_separated_ints(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints.

    Supports ``"32"`` → ``[32]`` and ``"4,32,32"`` → ``[4, 32, 32]``.

    Parameters
    ----------
    value : str
        Comma-separated string of integers.

    Returns
    -------
    list of int
        Parsed integers in input order.

    Raises
    ------
    argparse.ArgumentTypeError
        If any token cannot be parsed as an integer.
    """
    try:
        return [int(v) for v in value.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated integers: {value!r} ({e})"
        ) from e


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the single-method per-tile CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Per-tile two-sample test for stitching artifacts. Runs the "
            "metric on a single prediction set for one method."
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
        "--method_name",
        required=True,
        type=str,
        help="Display name for the method (used in logs and the report filename).",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to drop the JSON MethodReport.",
    )

    # TiledPatching geometry (per spatial axis, in image-pixel units).
    parser.add_argument(
        "--tile_size",
        required=True,
        type=parse_comma_separated_ints,
        help="TiledPatching tile size per spatial axis, e.g. '64,64' (2D) or '16,64,64' (3D).",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        type=parse_comma_separated_ints,
        help="TiledPatching overlap per spatial axis, e.g. '32,32'.",
    )

    # Statistical test.
    parser.add_argument(
        "--statistic",
        choices=["kl", "js", "ks", "wasserstein", "mean_abs_ratio"],
        default="kl",
        help="Two-sample discrepancy statistic.",
    )
    parser.add_argument("--strip_width", type=int, default=4, help="Control strip half-width N.")
    parser.add_argument("--block_size", type=int, default=3, help="Block size B for permutation.")
    parser.add_argument(
        "--n_permutations", type=int, default=1000, help="Number of permutations R."
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Rejection threshold.")
    parser.add_argument(
        "--num_bins_per_tile",
        type=int,
        default=32,
        help="Bins for histogram-based statistics (KL, JS).",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--no_pool_z_with_xy",
        dest="pool_z_with_xy",
        action="store_false",
        default=True,
        help=(
            "Reserved for v2: when set, run separate xy and z tests in 3D. "
            "Currently emits a warning and behaves as if pooling were on."
        ),
    )
    parser.add_argument(
        "--channel", type=int, default=0, help="Channel index to analyze (0-based)."
    )

    return parser.parse_args()


def main() -> int:
    """Run the per-tile metric pipeline for a single method from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on configuration or loading error).
    """
    args = parse_args()

    try:
        config = load_gradient_test_single_config_from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    prediction_file = config.predictions[0]
    method_name = config.method_names[0]

    print("=" * 60)
    print("PER-TILE METRIC PIPELINE (single method)")
    print("=" * 60)
    print(f"Method:      {method_name}")
    print(f"Dataset:     {config.dataset}")
    print(f"Prediction:  {prediction_file}")
    print(f"Save dir:    {config.save_dir}")
    print(f"Tile size:   {config.gradient_test.tile_size}")
    print(f"Overlap:     {config.gradient_test.overlap}")
    print(f"Statistic:   {config.gradient_test.statistic}")
    print(f"R:           {config.gradient_test.n_permutations}")
    print("=" * 60)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading prediction...")
    print(f"  {method_name}: {prediction_file}")
    try:
        pred = load_prediction(prediction_file)
        if pred.ndim == 6:
            pred = np.squeeze(pred, axis=0)
        pred = ensure_4d(pred)
        print(f"    shape: {pred.shape}")
    except Exception as e:
        print(f"Error loading {prediction_file}: {e}", file=sys.stderr)
        return 1

    print("\nRunning per-tile analysis...")
    run_gradient_analysis(
        predictions=pred,
        save_dir=config.save_dir,
        tile_size=config.gradient_test.tile_size,
        overlap=config.gradient_test.overlap,
        method_name=method_name,
        statistic=config.gradient_test.statistic,
        strip_width=config.gradient_test.strip_width,
        block_size=config.gradient_test.block_size,
        n_permutations=config.gradient_test.n_permutations,
        alpha=config.gradient_test.alpha,
        num_bins_per_tile=config.gradient_test.num_bins_per_tile,
        random_seed=config.gradient_test.random_seed,
        pool_z_with_xy=config.gradient_test.pool_z_with_xy,
        channel=config.gradient_test.channel,
    )

    print(f"\nDone. Results saved to: {config.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
