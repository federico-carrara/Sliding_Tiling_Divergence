"""Per-image tile scan: gradient → tile enumeration → sample → test.

The orchestrator in :mod:`analysis` iterates methods and images and calls
this for each ``(image, channel)`` slice. The function returns a fully
populated :class:`ImageReport` for that one slice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from tqdm import tqdm

from analysis_pipeline.gradient_test.aggregation import (
    ChannelReport,
    TileResult,
    aggregate_channel,
)
from analysis_pipeline.gradient_test.gradient_analysis import compute_gradients
from analysis_pipeline.gradient_test.permutation import permutation_pvalue
from analysis_pipeline.gradient_test.sampling import sample_tile
from analysis_pipeline.gradient_test.statistics import StatisticName, get_statistic
from analysis_pipeline.gradient_test.tiles import enumerate_tiles

# Variance floor for the per-tile Z-score. In flat regions the permutation null
# can collapse (``null_std ≈ 0``), which would make ``Z_obs`` explode; flooring
# the denominator keeps it finite.
Z_VAR_FLOOR = 1e-8


def _validate_step_vs_strip(
    tile_size: Sequence[int], overlap: Sequence[int], strip_width: int
) -> None:
    """Validate that each axis has enough step to fit the full control strip.

    Parameters
    ----------
    tile_size : Sequence[int]
        Per-axis tile size.
    overlap : Sequence[int]
        Per-axis overlap.
    strip_width : int
        Half-width ``N`` of the control strip.

    Raises
    ------
    ValueError
        If on any axis ``tile_size - overlap < 2 * strip_width + 2``.
    """
    for i, (ts, ov) in enumerate(zip(tile_size, overlap, strict=True)):
        step = ts - ov
        if step < 2 * strip_width + 2:
            raise ValueError(
                f"axis {i}: step = tile_size - overlap = {step} but strip_width="
                f"{strip_width} requires step >= 2*N + 2 = {2 * strip_width + 2}. "
                "Lower --strip_width or use a smaller overlap."
            )


def per_image_tile_scan(
    image: np.ndarray,
    *,
    tile_size: Sequence[int],
    overlap: Sequence[int],
    strip_width: int,
    block_size: int,
    n_permutations: int,
    statistic: StatisticName,
    alpha: float,
    num_bins_per_tile: int,
    rng: np.random.Generator,
    channel: int = 0,
) -> ChannelReport:
    """Run the per-tile test on one single-channel image slice.

    The caller is responsible for slicing the per-method ``(N, C, ...)``
    prediction down to the single-image, single-channel array we operate on.

    Parameters
    ----------
    image : np.ndarray
        2-D ``(H, W)`` or 3-D ``(D, H, W)`` image — no batch or channel axes.
    tile_size : Sequence[int]
        TiledPatching tile size per spatial axis.
    overlap : Sequence[int]
        TiledPatching overlap per spatial axis.
    strip_width : int
        Half-width ``N`` of the control strip around each seam.
    block_size : int
        Contiguous-block size ``B`` for permutation.
    n_permutations : int
        Number of permutations ``R`` per tile.
    statistic : str
        Name of the two-sample discrepancy statistic.
    alpha : float
        Rejection threshold for ``frac_rejected`` aggregation.
    num_bins_per_tile : int
        Histogram bin count for binned statistics (KL, JS).
    rng : numpy.random.Generator
        Random generator for the permutation engine.
    channel : int, default=0
        Channel index this slice was taken from; stamped onto the result.

    Returns
    -------
    ChannelReport
        Per-tile results aggregated into channel-level scalars.

    Raises
    ------
    ValueError
        If ``image`` is not 2-D or 3-D, or the lengths of ``tile_size`` and
        ``overlap`` do not match ``image.ndim``.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (H,W) or 3-D (D,H,W); got ndim={image.ndim}"
        )
    if len(tile_size) != image.ndim or len(overlap) != image.ndim:
        raise ValueError(
            "tile_size and overlap must have one entry per spatial axis "
            f"(image.ndim={image.ndim}, tile_size={list(tile_size)}, "
            f"overlap={list(overlap)})"
        )

    _validate_step_vs_strip(tile_size, overlap, strip_width)

    gradients = compute_gradients(image)
    tiles_list = enumerate_tiles(image.shape, tile_size, overlap)

    stat_spec = get_statistic(statistic)
    stat_kwargs: dict = {}
    if stat_spec.vec_kind == "binned":
        stat_kwargs["num_bins"] = num_bins_per_tile

    tile_results: list[TileResult] = []

    for tile in tqdm(tiles_list, desc="Running tests for tiles", unit="tile"):
        if tile.n_seams < 2:
            tile_results.append(
                TileResult(
                    coord=tile.coord,
                    n_seams=tile.n_seams,
                    T_obs=float("nan"),
                    p=float("nan"),
                    n_seam_samples=0,
                    n_control_samples=0,
                )
            )
            continue

        sample = sample_tile(gradients, tile, strip_width)

        T_obs, p, T_null = permutation_pvalue(
            sample.seam_slices,
            sample.control_slices,
            stat_spec=stat_spec,
            block_size=block_size,
            n_permutations=n_permutations,
            rng=rng,
            stat_kwargs=stat_kwargs,
        )

        # Calibrate T_obs against the tile's own permutation null so the score
        # is comparable across tiles and images. An empty null (degenerate tile)
        # leaves the calibrated fields NaN, matching the skipped-tile convention.
        if T_null.size:
            null_mean = float(np.mean(T_null))
            null_std = float(np.std(T_null))
            Z_obs = (T_obs - null_mean) / max(null_std, Z_VAR_FLOOR)
        else:
            null_mean = float("nan")
            null_std = float("nan")
            Z_obs = float("nan")

        tile_results.append(
            TileResult(
                coord=tile.coord,
                n_seams=tile.n_seams,
                T_obs=float(T_obs),
                p=float(p),
                null_mean=null_mean,
                null_std=null_std,
                Z_obs=float(Z_obs),
                n_seam_samples=int(sample.seam_sample.size),
                n_control_samples=int(sample.control_sample.size),
            )
        )

    return aggregate_channel(tile_results, alpha, channel)
