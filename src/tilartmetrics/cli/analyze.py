#!/usr/bin/env python3
"""Gradient (per-tile permutation) metric CLI, for one method.

Runs the reference-free per-tile two-sample test over every image of a single
prediction set and writes the aggregated :class:`MethodReport` plus a flat
``summary.csv``.

Input format
------------
``--predictions`` points at a single ``.npz`` archive whose **keys are image
names** and whose arrays squeeze to channel-first ``(C, H, W)`` (2-D) or
``(C, D, H, W)`` (3-D). The spatial dimensionality is inferred from the number
of ``--tile_size`` entries. Images may differ in size (each is tested
independently and merged into one report).

The test is reference-free, so there is no ground-truth argument. To test a
ground truth as a seam-free null baseline, run the command again pointing
``--predictions`` at the ground-truth ``.npz`` (with ``--method_name GT``).

Outputs (under ``--output_dir``)
--------------------------------
- ``gradient_test_config.json`` — the resolved :class:`GradientTestConfig`.
- ``{method_name}_gradient_report.json`` — the full per-method report.
- ``{method_name}_summary.csv`` — one row per (image, channel).

Example::

    analyze-experiment \\
        --predictions preds.npz --method_name inner_tiling \\
        --tile_size 64,64 --overlap 32,32 --statistic js \\
        --output_dir results/gradient_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tilartmetrics.config import GradientTestConfig
from tilartmetrics.gradient_test.analysis import run_gradient_analysis_dataset
from tilartmetrics.utils import iter_npz_images, read_image_names


def parse_comma_separated_ints(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints.

    Supports ``"32"`` -> ``[32]`` and ``"4,32,32"`` -> ``[4, 32, 32]``.

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
    """Parse command-line arguments for the gradient-test CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Reference-free per-tile two-sample test for stitching artifacts, "
            "run over one prediction set (a single .npz keyed by image name)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / output.
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction .npz archive (keys = image names, channel-first arrays).",
    )
    parser.add_argument(
        "--method_name",
        required=True,
        help="Display name for the method (used in logs and the report filename).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory for the report, config, and summary.csv.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name; stamped into the report and summary rows.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of images for a quick trial (default: all).",
    )

    # TiledPatching geometry (per spatial axis, image-pixel units).
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
        default="js",
        help="Two-sample discrepancy statistic.",
    )
    parser.add_argument(
        "--channels",
        type=parse_comma_separated_ints,
        default=None,
        help="Channel indices to test, e.g. '0,1' (default: all channels).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Rejection threshold.")
    parser.add_argument(
        "--n_permutations", type=int, default=1000, help="Permutations per tile."
    )
    parser.add_argument("--random_seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--strip_width",
        type=int,
        default=2,
        help="Half-width N of the control strip around each seam.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=3,
        help="Contiguous-block size B for the permutation engine.",
    )
    parser.add_argument(
        "--num_bins_per_tile",
        type=int,
        default=32,
        help="Histogram bins for binned statistics (KL, JS).",
    )
    parser.add_argument(
        "--no_normalize_per_axis",
        dest="normalize_per_axis",
        action="store_false",
        default=True,
        help="Disable per-axis mean/std normalization of gradients before pooling.",
    )
    parser.add_argument(
        "--no_balance_axis_counts",
        dest="balance_axis_counts",
        action="store_false",
        default=True,
        help="Disable per-axis count balancing (equal blocks per axis) within tiles.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the gradient-test pipeline for one method from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on configuration or I/O error).
    """
    args = parse_args()

    try:
        config = GradientTestConfig(
            tile_size=args.tile_size,
            overlap=args.overlap,
            statistic=args.statistic,
            strip_width=args.strip_width,
            block_size=args.block_size,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
            num_bins_per_tile=args.num_bins_per_tile,
            random_seed=args.random_seed,
            normalize_per_axis=args.normalize_per_axis,
            balance_axis_counts=args.balance_axis_counts,
            channels=args.channels,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    n_spatial = len(config.tile_size)

    print("=" * 60)
    print("GRADIENT (PER-TILE) METRIC")
    print("=" * 60)
    print(f"Method:       {args.method_name}")
    print(f"Dataset:      {args.dataset}")
    print(f"Predictions:  {args.predictions}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Tile size:    {config.tile_size}")
    print(f"Overlap:      {config.overlap}")
    print(f"Statistic:    {config.statistic}")
    print(f"R:            {config.n_permutations}")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gradient_test_config.json").write_text(
        config.model_dump_json(indent=2)
    )

    try:
        image_names = read_image_names(args.predictions, args.max_images)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading {args.predictions}: {e}", file=sys.stderr)
        return 1
    if not image_names:
        print(f"No images found in {args.predictions}", file=sys.stderr)
        return 1

    print(f"\n=== {args.method_name}: {len(image_names)} images ===")
    image_iter = iter_npz_images(args.predictions, image_names, n_spatial)
    report = run_gradient_analysis_dataset(
        image_iter,
        tile_size=config.tile_size,
        overlap=config.overlap,
        method_name=args.method_name,
        dataset=args.dataset,
        channels=config.channels,
        statistic=config.statistic,
        strip_width=config.strip_width,
        block_size=config.block_size,
        n_permutations=config.n_permutations,
        alpha=config.alpha,
        num_bins_per_tile=config.num_bins_per_tile,
        random_seed=config.random_seed,
        normalize_per_axis=config.normalize_per_axis,
        balance_axis_counts=config.balance_axis_counts,
    )
    report.save(args.output_dir / f"{args.method_name}_gradient_report.json")

    df = pd.DataFrame.from_records(report.to_records())
    df.to_csv(args.output_dir / f"{args.method_name}_summary.csv", index=False)

    print(f"\nSaved report + {len(df)}-row summary to {args.output_dir}")
    if not df.empty:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
