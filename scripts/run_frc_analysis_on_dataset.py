"""Run the FRC metric on every image of a dataset, per method.

Experimental driver, argparse-based so it can be launched from a SLURM job. The
FRC counterpart of ``scripts/run_gradient_test_on_dataset.py``: same data layout,
same method->subdir map, same report/summary outputs — swap ``gradient_test`` for
``frc`` and you are done.

Unlike the gradient test, FRC is a *reference* metric: every prediction is scored
against its matching ground truth, so the GT is mandatory (there is no seam-free
"null" source to append). FRC is also 2-D only; a 3-D volume ``(C, D, H, W)`` is
scored per z-slice, each z-plane contributing an extra image ``{name}_z{d:03d}``.

Data layout (see ``playground.ipynb``): for a dataset ``D`` each method stores a
single ``predictions.npz`` whose keys are image names and whose arrays squeeze to
channel-first ``(C, H, W)`` (2-D) or ``(C, D, H, W)`` (3-D); the matching ground
truth lives at ``{data_root}/{D}/targets/test/{image_name}.tif``. All images of a
dataset must share the same spatial size — FRC pools a per-bin mean curve + 95%
CI across images, which requires a common frequency grid (a clear error is raised
otherwise).

Outputs (under ``{output_root}``):
- ``{method}_frc_report.json`` — the full per-method report (nested
  image -> channel -> FRC curve), loadable via ``FRCMethodReport.load``.
- ``summary.csv`` — one row per (method, image, channel) from ``to_records()``.

Example::

    python scripts/run_frc_analysis_on_dataset.py \\
        --dataset PaviaATN --predictions_subdir predictions_MMSE64 \\
        --methods inner_tiling SWITi
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import tifffile as tiff

from analysis_pipeline.frc.aggregation import FRCMethodReport
from analysis_pipeline.frc.analysis import run_frc_analysis_dataset


METHODS_TO_SUBDIR = {
    "inner_tiling": "inner_tiling",
    "SWITi": "sw_inner_tiling",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run the FRC metric on all images of a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data location.
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. PaviaATN.")
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("/project/careamics/switi/results"),
        help="Root holding {dataset}/{predictions_subdir}/{method}/predictions.npz.",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path("/project/careamics/switi/data"),
        help="Root holding {dataset}/targets/test/{image}.tif ground truths.",
    )
    p.add_argument(
        "--predictions_subdir",
        default="predictions_MMSE64",
        help="Predictions folder under the dataset (e.g. predictions_MMSE64).",
    )
    p.add_argument(
        "--methods",
        required=True,
        type=str,
        nargs="+",
        choices=["inner_tiling", "SWITi"],
        help="Space-separated list of method names.",
    )
    # FRC geometry / parameters.
    p.add_argument(
        "--ndim",
        type=int,
        default=2,
        choices=[2, 3],
        help="Spatial dimensionality; 3-D volumes are scored per z-slice.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to analyse (default: all channels).",
    )
    p.add_argument(
        "--no_window",
        dest="apply_window",
        action="store_false",
        default=True,
        help="Disable the 2-D Hamming window (sanity tests only).",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/frc"),
        help="Reports + summary.csv are written under {output_root}.",
    )
    return p.parse_args()


def _ensure_channel_first(arr: np.ndarray, n_spatial: int) -> np.ndarray:
    """Squeeze to channel-first ``(C, *spatial)`` for the given spatial ndim.

    ``n_spatial`` is 2 for 2-D ``(C, H, W)`` or 3 for 3-D ``(C, D, H, W)``.
    A bare spatial array with no channel axis (``(H, W)`` / ``(D, H, W)``) is
    promoted to a single channel.
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim == n_spatial:
        arr = arr[np.newaxis, ...]
    if arr.ndim != n_spatial + 1:
        raise ValueError(
            f"expected {n_spatial + 1}-D channel-first array after squeeze "
            f"(n_spatial={n_spatial}), got shape {arr.shape}"
        )
    return arr


def read_image_names(npz_path: Path, max_images: int | None) -> list[str]:
    """Return the image names in a ``predictions.npz`` (reads the archive index,
    not the arrays); optionally capped to the first ``max_images``."""
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def _gt_filename(name: str) -> str:
    """Map a prediction image name to its ground-truth filename.

    The ``predictions.npz`` keys mirror the *input* image names (``input_img_*``),
    while the ground truths are stored as ``target_img_*.tif``; translate the
    ``input`` prefix to ``target`` so the two line up.
    """
    return f"{name.replace('input', 'target', 1)}.tif"


def iter_frc_pairs(
    npz_path: Path,
    target_dir: Path,
    image_names: list[str],
    n_spatial: int,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Lazily yield ``(image_id, prediction, ground_truth)`` 2-D slices.

    Each ``prediction`` / ``ground_truth`` is channel-first ``(C, H, W)``.
    ``.npz`` archives decompress each array only on access, so only one image is
    held in memory at a time. For ``n_spatial == 3`` each volume ``(C, D, H, W)``
    is expanded into ``D`` per-z-slice images with ids ``f"{name}_z{d:03d}"``.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            pred = _ensure_channel_first(data[name], n_spatial)
            gt = _ensure_channel_first(
                tiff.imread(target_dir / _gt_filename(name)), n_spatial
            )
            if n_spatial == 2:
                yield name, pred, gt
            else:  # 3-D: score each z-plane as its own 2-D image.
                depth = pred.shape[1]
                for d in range(depth):
                    yield f"{name}_z{d:03d}", pred[:, d], gt[:, d]


def main() -> None:
    args = parse_args()
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.results_root / args.dataset / args.predictions_subdir
    target_dir = args.data_root / args.dataset / "targets" / "test"

    # Image names shared across methods (cheap: reads the archive directory).
    first_subdir = METHODS_TO_SUBDIR[args.methods[0]]
    image_names = read_image_names(
        pred_root / first_subdir / "predictions.npz", args.max_images
    )

    reports: dict[str, FRCMethodReport] = {}
    for method_name in args.methods:
        print(f"\n=== {args.dataset} / {method_name}: {len(image_names)} images ===")
        pairs = iter_frc_pairs(
            pred_root / METHODS_TO_SUBDIR[method_name] / "predictions.npz",
            target_dir,
            image_names,
            args.ndim,
        )
        report = run_frc_analysis_dataset(
            pairs,
            method_name=method_name,
            dataset=args.dataset,
            channels=args.channels,
            apply_window=args.apply_window,
        )
        report.save(out_dir / f"{method_name}_frc_report.json")
        reports[method_name] = report

    records = [row for report in reports.values() for row in report.to_records()]
    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(reports)} reports + {len(df)}-row summary to {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
