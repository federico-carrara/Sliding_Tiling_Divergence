#!/usr/bin/env python3
"""Fourier Ring Correlation (FRC) metric CLI, for one method.

Computes the reference-based 2-D FRC of a prediction set against its ground
truth, aggregates per-frequency mean + 95% CI across the images, and writes the
report, a flat ``summary.csv``, and per-channel FRC-curve plots.

Input format
------------
Both ``--predictions`` and ``--ground_truth`` point at ``.npz`` archives whose
**keys are image names** and whose arrays squeeze to channel-first ``(C, H, W)``
(2-D) or ``(C, D, H, W)`` (3-D). Each prediction is paired with the ground-truth
array under the **same key** (an image whose key is missing from the ground-truth
archive is an error). FRC is a 2-D metric: with ``--ndim 3`` every z-slice is
scored as its own 2-D image (``{name}_z{d:03d}``).

Outputs (under ``--output_dir``)
--------------------------------
- ``{method_name}_frc_report.json`` — the full per-method report.
- ``{method_name}_summary.csv`` — one row per (image, channel).
- ``{method_name}_frc_curves_ch{c}.pdf`` — mean curve + 95% CI band per channel.

Example::

    frc-experiment \\
        --predictions preds.npz --ground_truth gt.npz \\
        --method_name inner_tiling --ndim 2 \\
        --step 64 --output_dir results/frc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tilartmetrics.frc.analysis import run_frc_analysis_dataset  # noqa: E402
from tilartmetrics.frc.plotting import plot_frc_curves, shared_ylim  # noqa: E402
from tilartmetrics.utils import ensure_channel_first, read_image_names  # noqa: E402


def parse_comma_separated_ints(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints (e.g. ``"0,1"``).

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
    """Parse command-line arguments for the FRC CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fourier Ring Correlation against ground truth for one method "
            "(prediction and ground-truth .npz archives keyed by image name)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Prediction .npz archive (keys = image names, channel-first arrays).",
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        type=Path,
        help="Ground-truth .npz archive, keyed by the same image names as --predictions.",
    )
    parser.add_argument(
        "--method_name",
        required=True,
        help="Display name for the method (used in logs and output filenames).",
    )
    parser.add_argument(
        "--ndim",
        required=True,
        type=int,
        choices=[2, 3],
        help="Spatial dimensionality; 3-D volumes are scored per z-slice.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory for the report, summary.csv, and FRC-curve plots.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name; stamped into the report, summary, and plot titles.",
    )
    parser.add_argument(
        "--channels",
        type=parse_comma_separated_ints,
        default=None,
        help="Channel indices to analyse, e.g. '0,1' (default: all channels).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Seam interval in pixels (e.g. tile_size - overlap). If given, dashed "
            "harmonic verticals k/step are drawn on the FRC-curve plots."
        ),
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap the number of images for a quick trial (default: all).",
    )
    parser.add_argument(
        "--no_window",
        dest="apply_window",
        action="store_false",
        default=True,
        help="Disable the 2-D Hamming window before FFT (real images need it).",
    )
    parser.add_argument(
        "--font_family",
        default=None,
        help="Font family for the plots (default: matplotlib's default).",
    )

    return parser.parse_args()


def iter_frc_pairs(
    pred_npz: Path,
    gt_npz: Path,
    image_names: list[str],
    ndim: int,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Lazily yield ``(image_id, prediction, ground_truth)`` triples.

    Prediction and ground truth are read from their respective ``.npz`` archives
    under the same key and normalised to channel-first layout. For ``ndim == 3``
    each z-slice is emitted as its own 2-D image (``{name}_z{d:03d}``), since FRC
    is a 2-D metric.

    Parameters
    ----------
    pred_npz, gt_npz : pathlib.Path
        Prediction / ground-truth ``.npz`` archives keyed by image name.
    image_names : list of str
        Prediction keys to process, in order.
    ndim : int
        Spatial dimensionality (2 or 3).

    Yields
    ------
    tuple of (str, np.ndarray, np.ndarray)
        ``(image_id, pred_2d, gt_2d)`` with both arrays channel-first ``(C, H, W)``.

    Raises
    ------
    KeyError
        If a prediction key is absent from the ground-truth archive.
    """
    with np.load(pred_npz, allow_pickle=True) as preds, np.load(
        gt_npz, allow_pickle=True
    ) as gts:
        gt_keys = set(gts.files)
        for name in image_names:
            if name not in gt_keys:
                raise KeyError(
                    f"image '{name}' is missing from ground-truth archive {gt_npz}"
                )
            pred = ensure_channel_first(preds[name], ndim)
            gt = ensure_channel_first(gts[name], ndim)
            if ndim == 2:
                yield name, pred, gt
            else:
                for d in range(pred.shape[1]):
                    yield f"{name}_z{d:03d}", pred[:, d], gt[:, d]


def main() -> int:
    """Run the FRC pipeline for one method from CLI arguments.

    Returns
    -------
    int
        Process exit code (``0`` on success, ``1`` on I/O or configuration error).
    """
    args = parse_args()

    print("=" * 60)
    print("FRC METRIC")
    print("=" * 60)
    print(f"Method:        {args.method_name}")
    print(f"Dataset:       {args.dataset}")
    print(f"Predictions:   {args.predictions}")
    print(f"Ground truth:  {args.ground_truth}")
    print(f"Output dir:    {args.output_dir}")
    print(f"ndim:          {args.ndim}")
    print(f"Apply window:  {args.apply_window}")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_names = read_image_names(args.predictions, args.max_images)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading {args.predictions}: {e}", file=sys.stderr)
        return 1
    if not image_names:
        print(f"No images found in {args.predictions}", file=sys.stderr)
        return 1

    print(f"\n=== {args.method_name}: {len(image_names)} images ===")
    try:
        pairs = iter_frc_pairs(
            args.predictions, args.ground_truth, image_names, args.ndim
        )
        report = run_frc_analysis_dataset(
            pairs,
            method_name=args.method_name,
            dataset=args.dataset,
            channels=args.channels,
            apply_window=args.apply_window,
        )
    except (KeyError, ValueError, FileNotFoundError, OSError) as e:
        print(f"Error running FRC analysis: {e}", file=sys.stderr)
        return 1

    report.save(args.output_dir / f"{args.method_name}_frc_report.json")

    df = pd.DataFrame.from_records(report.to_records())
    df.to_csv(args.output_dir / f"{args.method_name}_summary.csv", index=False)

    # Per-channel FRC-curve plots with a shared vertical scale.
    channels_plotted = sorted(report.mean_frc)
    ylim = shared_ylim([report], channels_plotted)
    steps = {args.method_name: args.step}
    for c in channels_plotted:
        fig = plot_frc_curves(
            [report],
            steps,
            save_path=args.output_dir / f"{args.method_name}_frc_curves_ch{c}.pdf",
            channel=c,
            ylim=ylim,
            font_family=args.font_family,
        )
        plt.close(fig)

    print(
        f"\nSaved report + {len(df)}-row summary + {len(channels_plotted)} plot(s) "
        f"to {args.output_dir}"
    )
    if not df.empty:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
