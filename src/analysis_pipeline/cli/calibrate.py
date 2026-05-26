#!/usr/bin/env python3
"""Block-size calibration CLI.

Scans candidate ``block_size`` values on a known-artifact-free reference
image, measures ``frac_rejected`` for each, and recommends the smallest
``B`` that controls Type I error at ``alpha + tolerance``.

The recommended ``B`` should then be passed to
``analyze-experiment --block_size <B>`` on the test set.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from ..gradient_test.calibration import calibrate_block_size
from ..utils import ensure_4d, load_prediction


def _parse_csv_ints(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints.

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
    """Parse command-line arguments for the block-size calibration CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the per-tile permutation test's block_size on a known-"
            "artifact-free reference. Recommends the smallest B with "
            "frac_rejected <= alpha + tolerance."
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        type=str,
        help="Comma-separated reference files (.tiff/.pkl). Use GT or non-tiled predictions from the same modality.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to write calibration_report.pkl.",
    )
    parser.add_argument(
        "--tile_size",
        required=True,
        type=_parse_csv_ints,
        help="TiledPatching tile size per spatial axis (must match test-time geometry).",
    )
    parser.add_argument(
        "--overlap",
        required=True,
        type=_parse_csv_ints,
        help="TiledPatching overlap per spatial axis (must match test-time geometry).",
    )

    parser.add_argument(
        "--strip_width", type=int, default=4, help="Control strip half-width N."
    )
    parser.add_argument(
        "--statistic",
        choices=["kl", "js", "ks", "wasserstein", "mean_abs_ratio"],
        default="kl",
        help="Two-sample statistic (must match test-time choice).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Target Type I error rate."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Slack on the selection rule: smallest B with frac_rejected <= alpha + tolerance.",
    )
    parser.add_argument(
        "--block_sizes",
        type=_parse_csv_ints,
        default=[1, 2, 4, 8, 16],
        help="Candidate B values (CSV). Default: 1,2,4,8,16 (geometric doubling).",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="RNG seeds to average over. >1 only useful on small references.",
    )
    parser.add_argument(
        "--base_seed", type=int, default=0, help="First RNG seed."
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
        help="Permutations per tile (must match test-time setting).",
    )
    parser.add_argument(
        "--num_bins_per_tile",
        type=int,
        default=32,
        help="Histogram bins for binned statistics (KL, JS).",
    )
    parser.add_argument(
        "--channel", type=int, default=0, help="Channel index to analyze."
    )

    return parser.parse_args()


def _load_reference_images(paths: list[str], channel: int) -> list[np.ndarray]:
    """Load reference files and return a flat list of single-channel slices.

    Each reference path is loaded, normalised to channel-first
    ``(N, C, H, W)`` (or ``(N, C, D, H, W)`` for 3D), then split into
    per-sample, single-channel slices.

    Parameters
    ----------
    paths : list of str
        Reference file paths (``.tiff``/``.pkl``).
    channel : int
        Channel index to extract from each loaded array.

    Returns
    -------
    list of np.ndarray
        Single-channel slices of shape ``(H, W)`` or ``(D, H, W)``.

    Raises
    ------
    ValueError
        If ``channel`` is out of range for any loaded file.
    """
    images: list[np.ndarray] = []
    for path in paths:
        print(f"  loading {path}")
        arr = load_prediction(path)
        if arr.ndim == 6:
            arr = np.squeeze(arr, axis=0)
        arr = ensure_4d(arr)  # (N, C, H, W) or (N, C, D, H, W)
        if not (0 <= channel < arr.shape[1]):
            raise ValueError(
                f"{path}: channel={channel} out of range for C={arr.shape[1]}"
            )
        print(f"    shape: {arr.shape}")
        for n in range(arr.shape[0]):
            images.append(arr[n, channel])
    return images


def _print_summary(report, save_dir: Path) -> None:
    """Print a human-readable summary of the calibration report.

    Parameters
    ----------
    report : CalibrationReport
        The calibration report to summarise.
    save_dir : pathlib.Path
        Directory the report was pickled to (referenced in the output).
    """
    bar = "=" * 72
    threshold = report.alpha + report.tolerance
    print()
    print(bar)
    print(
        f"BLOCK-SIZE CALIBRATION SUMMARY  (alpha={report.alpha}, "
        f"tolerance={report.tolerance}, threshold={threshold:.3f})"
    )
    print(bar)
    print(f"{'B':>4s}  {'frac_rejected':>14s}  {'sd':>8s}  {'n_tiles':>9s}  status")
    print("-" * 72)
    for c in report.candidates:
        status = "PASS" if c.frac_rejected_mean <= threshold else "FAIL"
        sd_str = f"{c.frac_rejected_sd:.4f}"
        if c.frac_rejected_sd == 0.0:
            sd_str += "*"  # marker for single-sample
        print(
            f"{c.block_size:>4d}  {c.frac_rejected_mean:>14.4f}  "
            f"{sd_str:>8s}  {c.n_tiles_total:>9d}  {status}"
        )
    print(bar)
    if report.recommended_block_size is None:
        print(
            "NO candidate B satisfies the selection rule. Consider raising the "
            f"ceiling above max({list(report.config_summary['candidate_block_sizes'])})."
        )
    else:
        print(f"Recommended block_size: {report.recommended_block_size}")
        print(
            f"Use this with: analyze-experiment --block_size "
            f"{report.recommended_block_size} ..."
        )
    if any(c.frac_rejected_sd == 0.0 for c in report.candidates):
        print(
            "  (* sd=0 indicates a single sample — increase --n_seeds or use a larger reference "
            "for a real noise estimate.)"
        )
    print(f"\nReport pickled to: {save_dir / 'calibration_report.pkl'}")


def main() -> int:
    """Run the block-size calibration pipeline from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on loading or calibration error).
    """
    args = parse_args()

    print("=" * 72)
    print("BLOCK-SIZE CALIBRATION")
    print("=" * 72)
    print(
        "NOTE: use a reference image from the same modality and through the "
        "same patching scheme as the test set. A synthetic flat field will "
        "report B=1 is fine — which is correct for white noise but does not "
        "generalize to real microscopy data with spatial correlation."
    )
    print("-" * 72)
    print(f"Reference:   {args.reference}")
    print(f"Tile size:   {args.tile_size}")
    print(f"Overlap:     {args.overlap}")
    print(f"Statistic:   {args.statistic}")
    print(f"Block grid:  {args.block_sizes}")
    print(f"alpha={args.alpha}  tolerance={args.tolerance}  n_seeds={args.n_seeds}")
    print("-" * 72)

    paths = [p.strip() for p in args.reference.split(",") if p.strip()]
    try:
        images = _load_reference_images(paths, args.channel)
    except Exception as e:
        print(f"Error loading reference: {e}", file=sys.stderr)
        return 1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nScanning candidate block sizes...")
    try:
        report = calibrate_block_size(
            images,
            tile_size=args.tile_size,
            overlap=args.overlap,
            strip_width=args.strip_width,
            statistic=args.statistic,
            n_permutations=args.n_permutations,
            num_bins_per_tile=args.num_bins_per_tile,
            alpha=args.alpha,
            tolerance=args.tolerance,
            candidate_block_sizes=args.block_sizes,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            verbose=True,
        )
    except ValueError as e:
        print(f"Calibration error: {e}", file=sys.stderr)
        return 1

    with open(save_dir / "calibration_report.pkl", "wb") as f:
        pickle.dump(report, f)

    _print_summary(report, save_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
