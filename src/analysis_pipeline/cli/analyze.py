#!/usr/bin/env python3
"""Multi-method per-tile-metric CLI.

Loads 2-5 predictions, runs the per-tile two-sample test on each, and prints
a per-method summary. The structured ``MultiMethodReport`` is pickled to
``save_dir/per_tile_report.pkl`` so downstream notebooks can load it
directly.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..config.settings import load_config_from_args
from ..core.analysis import run_gradient_analysis_multi
from ..utils import ensure_4d, load_prediction


def parse_comma_separated_ints(value: str) -> list[int]:
    """Parse a CSV of ints. Supports ``32`` → ``[32]`` and ``4,32,32`` → ``[4, 32, 32]``."""
    try:
        return [int(v) for v in value.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid comma-separated integers: {value!r} ({e})"
        ) from e


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-tile two-sample test for stitching artifacts. Compares 2-5 "
            "predictions of the same image set."
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
        "--method_names",
        required=True,
        type=str,
        help="Comma-separated method names, e.g. 'OG,SW,Method3'.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to drop the pickled MultiMethodReport.",
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
    args = parse_args()

    try:
        config = load_config_from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("PER-TILE METRIC PIPELINE")
    print("=" * 60)
    print(f"Methods:     {', '.join(config.method_names)}")
    print(f"Dataset:     {config.dataset}")
    print(f"Predictions: {config.predictions}")
    print(f"Save dir:    {config.save_dir}")
    print(f"Tile size:   {config.per_tile.tile_size}")
    print(f"Overlap:     {config.per_tile.overlap}")
    print(f"Statistic:   {config.per_tile.statistic}")
    print(f"R:           {config.per_tile.n_permutations}")
    print("=" * 60)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading predictions...")
    predictions_list: list[np.ndarray] = []
    for pred_file, method_name in zip(
        config.predictions, config.method_names, strict=True
    ):
        print(f"  {method_name}: {pred_file}")
        try:
            pred = load_prediction(pred_file)
            if pred.ndim == 6:
                pred = np.squeeze(pred, axis=0)
            pred = ensure_4d(pred)
            print(f"    shape: {pred.shape}")
            predictions_list.append(pred)
        except Exception as e:
            print(f"Error loading {pred_file}: {e}", file=sys.stderr)
            return 1

    print("\nRunning per-tile analysis...")
    run_gradient_analysis_multi(
        predictions_list=predictions_list,
        method_names=config.method_names,
        save_dir=config.save_dir,
        tile_size=config.per_tile.tile_size,
        overlap=config.per_tile.overlap,
        statistic=config.per_tile.statistic,
        strip_width=config.per_tile.strip_width,
        block_size=config.per_tile.block_size,
        n_permutations=config.per_tile.n_permutations,
        alpha=config.per_tile.alpha,
        num_bins_per_tile=config.per_tile.num_bins_per_tile,
        random_seed=config.per_tile.random_seed,
        pool_z_with_xy=config.per_tile.pool_z_with_xy,
        channel=config.per_tile.channel,
    )

    print(f"\nDone. Results saved to: {config.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
