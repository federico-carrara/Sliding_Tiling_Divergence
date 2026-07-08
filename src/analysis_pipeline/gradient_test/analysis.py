"""Public orchestrator for the per-tile stitching-artifact metric.

Loops images → channel slice → :func:`per_image_tile_scan` and returns a
:class:`MethodReport`. This is the primary public API: one method, a set of
predictions. Multi-method comparison lives in :mod:`.comparison`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from analysis_pipeline.gradient_test.aggregation import (
    ChannelReport,
    ImageReport,
    MethodReport,
    aggregate_image,
    aggregate_method,
)
from analysis_pipeline.gradient_test.per_tile import per_image_tile_scan
from analysis_pipeline.gradient_test.statistics import StatisticName


def run_gradient_analysis(
    predictions: np.ndarray,
    tile_size: list[int],
    overlap: list[int],
    *,
    channels: Optional[Sequence[int]] = None,
    image_ids: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    save_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    statistic: StatisticName = "js",
    strip_width: int = 4,
    block_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    num_bins_per_tile: int = 32,
    random_seed: int = 0,
    pool_z_with_xy: bool = True,
) -> MethodReport:
    """Run the per-tile metric on a set of predictions for a single method.

    ``predictions`` is channel-first: ``(N, C, H, W)`` for 2-D or
    ``(N, C, D, H, W)`` for 3-D. ``tile_size`` and ``overlap`` are per
    spatial axis and must match the TiledPatching configuration used to
    produce the predictions.

    If ``save_dir`` is not None, the report is serialized as JSON to
    ``save_dir / f"{method_name}_per_tile_report.json"``.

    Parameters
    ----------
    predictions : np.ndarray
        Channel-first prediction array for one method.
    tile_size : list of int
        Per-spatial-axis tile size.
    overlap : list of int
        Per-spatial-axis overlap.
    channels : sequence of int, optional
        Channel indices to analyse. If ``None`` (default), every channel is
        analysed. Each image's report then holds one ``ChannelReport`` per
        requested channel.
    image_ids : sequence of str, optional
        Identifier for each image, one per ``N``; used as the keys of the
        returned ``images`` dict. Defaults to ``"0" .. "N-1"``.
    dataset : str, optional
        Name of the dataset the predictions were drawn from; stamped onto the
        report for downstream analysis.
    save_dir : pathlib.Path, optional
        Directory for the JSON report (created if missing); pass ``None``
        to skip writing. Default is ``None``.
    method_name : str, default="method"
        Display name used in logs and the report filename.
    statistic : StatisticName, default="js"
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
    n_channels = predictions.shape[1]
    if channels is None:
        channels = list(range(n_channels))
    else:
        channels = list(channels)
        if not channels:
            raise ValueError(
                f"{method_name}: channels must be a non-empty sequence or None"
            )
        for c in channels:
            if not (0 <= c < n_channels):
                raise ValueError(
                    f"{method_name}: channel={c} out of range for "
                    f"C={n_channels}"
                )

    n_images = predictions.shape[0]
    if image_ids is None:
        image_ids = [str(n) for n in range(n_images)]
    elif len(image_ids) != n_images:
        raise ValueError(
            f"{method_name}: image_ids has {len(image_ids)} entries, "
            f"expected {n_images} (one per image)"
        )

    if not is_2d and not pool_z_with_xy:
        warnings.warn(
            "pool_z_with_xy=False is reserved for a future revision; the v1 "
            "implementation always pools all spatial axes. Result reflects "
            "pool_z_with_xy=True.",
            stacklevel=2,
        )

    rng = np.random.default_rng(random_seed)

    print(f"  [{method_name}] {n_images} images × channels {channels}")

    images: dict[str, ImageReport] = {}
    for n, image_id in enumerate(image_ids):
        channel_reports: dict[int, ChannelReport] = {}
        for c in channels:
            ch = per_image_tile_scan(
                predictions[n, c],
                tile_size=tile_size,
                overlap=overlap,
                strip_width=strip_width,
                block_size=block_size,
                n_permutations=n_permutations,
                statistic=statistic,
                alpha=alpha,
                num_bins_per_tile=num_bins_per_tile,
                rng=rng,
                channel=c,
            )
            channel_reports[c] = ch
        images[image_id] = aggregate_image(image_id, channel_reports, alpha)
        summary = "  ".join(
            f"c{c}: median_T={channel_reports[c].median_T:.4g} "
            f"frac_rejected={channel_reports[c].frac_rejected:.3f}"
            for c in channels
        )
        print(f"    image {image_id}: {summary}")

    method_report = aggregate_method(images, method_name, dataset)
    for c in channels:
        print(
            f"  -> {method_name} [c{c}]: "
            f"mean_median_T={method_report.mean_median_T[c]:.4g} "
            f"mean_frac_rejected={method_report.mean_frac_rejected[c]:.3f}"
        )

    if save_dir is not None:
        out_path = method_report.save(
            Path(save_dir) / f"{method_name}_per_tile_report.json"
        )
        print(f"\nReport written to: {out_path}")

    return method_report
