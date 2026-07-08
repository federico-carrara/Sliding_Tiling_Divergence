"""Public orchestrator for the per-tile stitching-artifact metric.

:func:`run_gradient_analysis` is the single-image primitive: it takes one
channel-first prediction ``(C, ...)`` and returns an :class:`ImageReport`.
:func:`run_gradient_analysis_dataset` wraps it over an (optionally lazy)
iterable of images and returns a :class:`MethodReport`. Multi-method comparison
lives in :mod:`.comparison`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

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
    prediction: np.ndarray,
    tile_size: list[int],
    overlap: list[int],
    *,
    image_id: str = "0",
    channels: Optional[Sequence[int]] = None,
    statistic: StatisticName = "js",
    strip_width: int = 4,
    block_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    num_bins_per_tile: int = 32,
    random_seed: int = 0,
    pool_z_with_xy: bool = True,
    verbose: bool = True,
) -> ImageReport:
    """Run the per-tile metric on a single channel-first image.

    ``prediction`` is channel-first with no batch axis: ``(C, H, W)`` for 2-D or
    ``(C, D, H, W)`` for 3-D. ``tile_size`` and ``overlap`` are per spatial axis
    and must match the TiledPatching configuration used to produce the
    prediction. Each requested channel is tested independently and the results
    are grouped into one :class:`ImageReport`.

    Parameters
    ----------
    prediction : np.ndarray
        Channel-first prediction for one image: ``(C, H, W)`` or ``(C, D, H, W)``.
    tile_size : list of int
        Per-spatial-axis tile size.
    overlap : list of int
        Per-spatial-axis overlap.
    image_id : str, default="0"
        Identifier stamped onto the returned :class:`ImageReport`.
    channels : sequence of int, optional
        Channel indices to analyse. If ``None`` (default), every channel is
        analysed; the report then holds one ``ChannelReport`` per channel.
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
        RNG seed. A fresh generator is created per call, so an image's result is
        independent of any other image (reproducible regardless of run order).
    pool_z_with_xy : bool, default=True
        If False (reserved for v2), run separate xy and z tests in 3D.
    verbose : bool, default=True
        Print a one-line per-channel summary for this image.

    Returns
    -------
    ImageReport
        Per-image report grouping the requested channels.

    Raises
    ------
    ValueError
        If shapes or per-axis specs are inconsistent with the prediction.
    """
    is_2d = prediction.ndim == 3  # (C, H, W)
    n_spatial = 2 if is_2d else 3

    if prediction.ndim != n_spatial + 1:
        raise ValueError(
            f"{image_id}: expected ndim={n_spatial + 1} (C,...); "
            f"got {prediction.ndim}"
        )
    if len(tile_size) != n_spatial:
        raise ValueError(
            f"{image_id}: tile_size has {len(tile_size)} entries, "
            f"expected {n_spatial}"
        )
    if len(overlap) != n_spatial:
        raise ValueError(
            f"{image_id}: overlap has {len(overlap)} entries, "
            f"expected {n_spatial}"
        )

    n_channels = prediction.shape[0]
    if channels is None:
        channels = list(range(n_channels))
    else:
        channels = list(channels)
        if not channels:
            raise ValueError(
                f"{image_id}: channels must be a non-empty sequence or None"
            )
        for c in channels:
            if not (0 <= c < n_channels):
                raise ValueError(
                    f"{image_id}: channel={c} out of range for C={n_channels}"
                )

    if not is_2d and not pool_z_with_xy:
        warnings.warn(
            "pool_z_with_xy=False is reserved for a future revision; the v1 "
            "implementation always pools all spatial axes. Result reflects "
            "pool_z_with_xy=True.",
            stacklevel=2,
        )

    rng = np.random.default_rng(random_seed)

    channel_reports: dict[int, ChannelReport] = {}
    for c in channels:
        channel_reports[c] = per_image_tile_scan(
            prediction[c],
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

    image_report = aggregate_image(image_id, channel_reports, alpha)

    if verbose:
        summary = "  ".join(
            f"c{c}: median_T={channel_reports[c].median_T:.4g} "
            f"median_Z={channel_reports[c].median_Z:.3g} "
            f"frac_rejected={channel_reports[c].frac_rejected:.3f}"
            for c in channels
        )
        print(f"    image {image_id}: {summary}")

    return image_report


def run_gradient_analysis_dataset(
    images: Iterable[tuple[str, np.ndarray]],
    tile_size: list[int],
    overlap: list[int],
    *,
    method_name: str = "method",
    dataset: Optional[str] = None,
    channels: Optional[Sequence[int]] = None,
    save_dir: Optional[Path] = None,
    statistic: StatisticName = "js",
    strip_width: int = 4,
    block_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    num_bins_per_tile: int = 32,
    random_seed: int = 0,
    pool_z_with_xy: bool = True,
    verbose: bool = True,
) -> MethodReport:
    """Run :func:`run_gradient_analysis` over a set of images for one method.

    ``images`` is an iterable of ``(image_id, prediction)`` pairs (or a mapping
    ``{image_id: prediction}``). It is consumed lazily, so callers can stream one
    image into memory at a time (e.g. from a ``.npz`` archive) rather than
    materialising the whole dataset. Predictions may differ in spatial size since
    each is tested independently.

    If ``save_dir`` is not None, the report is serialized as JSON to
    ``save_dir / f"{method_name}_per_tile_report.json"``.

    Parameters
    ----------
    images : iterable of (str, np.ndarray), or mapping of str to np.ndarray
        ``(image_id, prediction)`` pairs; each prediction is channel-first
        ``(C, H, W)`` or ``(C, D, H, W)``.
    tile_size, overlap : list of int
        Per-spatial-axis tile size / overlap.
    method_name : str, default="method"
        Display name; also the report filename stem.
    dataset : str, optional
        Dataset name stamped onto the report.
    channels : sequence of int, optional
        Channel indices to analyse (``None`` = all), forwarded to each image.
    save_dir : pathlib.Path, optional
        Directory for the JSON report; ``None`` skips writing.
    statistic, strip_width, block_size, n_permutations, alpha, num_bins_per_tile, random_seed, pool_z_with_xy
        Forwarded per image to :func:`run_gradient_analysis`.
    verbose : bool, default=True
        Print per-image and per-method summaries.

    Returns
    -------
    MethodReport
        Aggregated per-method report keyed by ``image_id``.
    """
    if isinstance(images, Mapping):
        images = images.items()

    if verbose:
        print(f"  [{method_name}] running (channels={channels})")

    image_reports: dict[str, ImageReport] = {}
    for image_id, prediction in images:
        image_reports[image_id] = run_gradient_analysis(
            prediction,
            tile_size=tile_size,
            overlap=overlap,
            image_id=image_id,
            channels=channels,
            statistic=statistic,
            strip_width=strip_width,
            block_size=block_size,
            n_permutations=n_permutations,
            alpha=alpha,
            num_bins_per_tile=num_bins_per_tile,
            random_seed=random_seed,
            pool_z_with_xy=pool_z_with_xy,
            verbose=verbose,
        )

    method_report = aggregate_method(image_reports, method_name, dataset)

    if verbose:
        for c in sorted(method_report.mean_median_T):
            print(
                f"  -> {method_name} [c{c}]: "
                f"mean_median_T={method_report.mean_median_T[c]:.4g} "
                f"mean_median_Z={method_report.mean_median_Z[c]:.3g} "
                f"mean_frac_rejected={method_report.mean_frac_rejected[c]:.3f}"
            )

    if save_dir is not None:
        out_path = method_report.save(
            Path(save_dir) / f"{method_name}_per_tile_report.json"
        )
        if verbose:
            print(f"\nReport written to: {out_path}")

    return method_report
