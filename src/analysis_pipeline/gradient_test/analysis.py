"""Public orchestrator for the per-tile stitching-artifact metric.

Loops images → channel slice → :func:`per_image_tile_scan` and returns a
:class:`MethodReport`. This is the primary public API: one method, a set of
predictions. Multi-method comparison lives in :mod:`.comparison`.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .aggregation import MethodReport, aggregate_method
from .per_tile import per_image_tile_scan


def run_gradient_analysis(
    predictions: np.ndarray,
    save_dir: Optional[Path],
    tile_size: list[int],
    overlap: list[int],
    *,
    method_name: str = "method",
    statistic: str = "kl",
    strip_width: int = 4,
    block_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    num_bins_per_tile: int = 32,
    random_seed: int = 0,
    pool_z_with_xy: bool = True,
    channel: int = 0,
) -> MethodReport:
    """Run the per-tile metric on a set of predictions for a single method.

    ``predictions`` is channel-first: ``(N, C, H, W)`` for 2-D or
    ``(N, C, D, H, W)`` for 3-D. ``tile_size`` and ``overlap`` are per
    spatial axis and must match the TiledPatching configuration used to
    produce the predictions.

    If ``save_dir`` is not None, the report is pickled to
    ``save_dir / f"{method_name}_per_tile_report.pkl"``.

    Parameters
    ----------
    predictions : np.ndarray
        Channel-first prediction array for one method.
    save_dir : pathlib.Path, optional
        Directory for the pickled report (created if missing); pass ``None``
        to skip writing.
    tile_size : list of int
        Per-spatial-axis tile size.
    overlap : list of int
        Per-spatial-axis overlap.
    method_name : str, default="method"
        Display name used in logs and the pickle filename.
    statistic : str, default="kl"
        Two-sample discrepancy statistic name.
    strip_width : int, default=4
        Control-strip half-width ``N``.
    block_size : int, default=3
        Contiguous-block size ``B`` for permutation.
    n_permutations : int, default=1000
        Number of permutations ``R`` per tile.
    alpha : float, default=0.05
        Rejection threshold for the per-tile test.
    num_bins_per_tile : int, default=32
        Histogram bin count for binned statistics (KL, JS).
    random_seed : int, default=0
        RNG seed.
    pool_z_with_xy : bool, default=True
        If False (reserved for v2), run separate xy and z tests in 3D.
    channel : int, default=0
        Channel index to analyse.

    Returns
    -------
    MethodReport
        Aggregated per-method report.

    Raises
    ------
    ValueError
        If shapes or per-axis specs are inconsistent with the predictions.
    """
    is_2d = predictions.ndim == 4  # (N, C, H, W)
    n_spatial = 2 if is_2d else 3

    if predictions.ndim != n_spatial + 2:
        raise ValueError(
            f"{method_name}: expected ndim={n_spatial + 2} (N,C,...); "
            f"got {predictions.ndim}"
        )
    if len(tile_size) != n_spatial:
        raise ValueError(
            f"{method_name}: tile_size has {len(tile_size)} entries, "
            f"expected {n_spatial}"
        )
    if len(overlap) != n_spatial:
        raise ValueError(
            f"{method_name}: overlap has {len(overlap)} entries, "
            f"expected {n_spatial}"
        )
    if not (0 <= channel < predictions.shape[1]):
        raise ValueError(
            f"{method_name}: channel={channel} out of range for "
            f"C={predictions.shape[1]}"
        )

    if not is_2d and not pool_z_with_xy:
        warnings.warn(
            "pool_z_with_xy=False is reserved for a future revision; the v1 "
            "implementation always pools all spatial axes. Result reflects "
            "pool_z_with_xy=True.",
            stacklevel=2,
        )

    rng = np.random.default_rng(random_seed)

    print(f"  [{method_name}] {predictions.shape[0]} images × channel {channel}")

    image_reports = []
    for n in range(predictions.shape[0]):
        image = predictions[n, channel]
        ir = per_image_tile_scan(
            image,
            tile_size=tile_size,
            overlap=overlap,
            strip_width=strip_width,
            block_size=block_size,
            n_permutations=n_permutations,
            statistic=statistic,
            alpha=alpha,
            num_bins_per_tile=num_bins_per_tile,
            rng=rng,
        )
        image_reports.append(ir)
        print(
            f"    image {n}: median_T={ir.median_T:.4g} "
            f"frac_rejected={ir.frac_rejected:.3f}"
        )

    method_report = aggregate_method(image_reports)
    print(
        f"  -> {method_name}: mean_median_T={method_report.mean_median_T:.4g} "
        f"mean_frac_rejected={method_report.mean_frac_rejected:.3f}"
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{method_name}_per_tile_report.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(method_report, f)
        print(f"\nReport pickled to: {out_path}")

    return method_report
