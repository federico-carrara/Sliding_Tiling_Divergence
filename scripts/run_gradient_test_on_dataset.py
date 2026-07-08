"""Run the gradient permutation test on every image of a dataset, per method.

Experimental driver, argparse-based so it can be launched from a SLURM job.

Data layout (see ``playground.ipynb``): for a dataset ``D`` each method stores a
single ``predictions.npz`` whose keys are image names and whose arrays squeeze to
``(C, H, W)``; the matching ground truth lives at
``{data_root}/{D}/targets/test/{image_name}.tif``. Images within a dataset may
differ in size, so we test them one at a time (``N=1``) and merge the per-image
reports into one :class:`MethodReport` per method.

Outputs (under ``{output_root}/{dataset}``):
- ``{method}_gradient_report.json`` — the full per-method report (nested
  image -> channel -> tiles), loadable via ``MethodReport.load``.
- ``summary.csv`` — one row per (method, image, channel) from ``to_records()``.

Example::

    python scripts/run_gradient_test_on_dataset.py \\
        --dataset PaviaATN --predictions_subdir predictions_MMSE64 \\
        --tile_size 64,64 --overlap 32,32 --statistic js
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import tifffile as tiff

from analysis_pipeline.gradient_test.aggregation import MethodReport
from analysis_pipeline.gradient_test.analysis import run_gradient_analysis_dataset


METHODS_TO_SUBDIR = {
    "inner_tiling": "inner_tiling",
    "SWITi": "SWITi",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run the gradient permutation test on all images of a dataset.",
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
        default="predictions",
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
    p.add_argument(
        "--no_gt",
        dest="include_gt",
        action="store_false",
        default=True,
        help="Skip the ground truth (by default GT is tested as a seam-free null).",
    )
    # Gradient-test geometry / parameters.
    p.add_argument(
        "--tile_size",
        required=True,
        type=int,
        nargs="+",
        help="TiledPatching tile size per spatial axis.",
    )
    p.add_argument(
        "--overlap",
        required=True,
        type=int,
        nargs="+",
        help="TiledPatching overlap per spatial axis.",
    )
    p.add_argument(
        "--statistic",
        default="js",
        choices=["kl", "js", "ks", "wasserstein", "mean_abs_ratio"],
        help="Two-sample discrepancy statistic.",
    )
    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to test (default: all channels).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Rejection threshold.")
    p.add_argument(
        "--n_permutations", type=int, default=1000, help="Permutations per tile."
    )
    p.add_argument("--random_seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cap images per method for quick trials (default: all).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/gradient_test"),
        help="Reports + summary.csv are written under {output_root}/{dataset}.",
    )
    return p.parse_args()


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    """Squeeze to ``(C, H, W)``, promoting a bare ``(H, W)`` to a single channel."""
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected (C, H, W) after squeeze, got shape {arr.shape}")
    return arr


def read_image_names(npz_path: Path, max_images: int | None) -> list[str]:
    """Return the image names in a ``predictions.npz`` (reads the archive index,
    not the arrays); optionally capped to the first ``max_images``."""
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def iter_prediction_images(
    npz_path: Path, image_names: list[str]
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, H, W))`` from a ``predictions.npz``.

    ``.npz`` archives decompress each array only on access, so this keeps just
    one image in memory at a time.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            yield name, _ensure_chw(data[name])


def iter_gt_images(
    target_dir: Path, image_names: list[str]
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, H, W))`` ground truths, one file at a time."""
    for name in image_names:
        yield name, _ensure_chw(tiff.imread(target_dir / f"{name}.tif"))


def main() -> None:
    args = parse_args()
    out_dir = args.output_root / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = args.results_root / args.dataset / args.predictions_subdir

    # Image names shared across methods/GT (cheap: reads the archive directory).
    first_subdir = METHODS_TO_SUBDIR[args.methods[0]]
    image_names = read_image_names(
        pred_root / first_subdir / "predictions.npz", args.max_images
    )

    # (method_name, lazy image iterator) sources, GT appended as a null reference.
    sources: list[tuple[str, Iterator[tuple[str, np.ndarray]]]] = [
        (
            name, 
            iter_prediction_images(
                pred_root / METHODS_TO_SUBDIR[name] / "predictions.npz", image_names
            )
        )
        for name  in args.methods
    ]
    if args.include_gt:
        target_dir = args.data_root / args.dataset / "targets" / "test"
        sources.append(("GT", iter_gt_images(target_dir, image_names)))

    reports: dict[str, MethodReport] = {}
    for method_name, image_iter in sources:
        print(f"\n=== {args.dataset} / {method_name}: {len(image_names)} images ===")
        report = run_gradient_analysis_dataset(
            image_iter,
            tile_size=args.tile_size,
            overlap=args.overlap,
            method_name=method_name,
            dataset=args.dataset,
            channels=args.channels,
            statistic=args.statistic,
            alpha=args.alpha,
            n_permutations=args.n_permutations,
            random_seed=args.random_seed,
        )
        report.save(out_dir / f"{method_name}_gradient_report.json")
        reports[method_name] = report

    records = [row for report in reports.values() for row in report.to_records()]
    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(reports)} reports + {len(df)}-row summary to {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
