"""Per-image tile scan: gradient → tile enumeration → sample → test.

The orchestrator in :mod:`analysis` iterates methods and images and calls
this for each ``(image, channel)`` slice. The function returns a fully
populated :class:`ImageReport` for that one slice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .aggregation import ImageReport, TileResult, aggregate_image
from .gradient_analysis import compute_gradients
from .permutation import permutation_pvalue
from .sampling import sample_tile
from .statistics import get_statistic
from .tiles import enumerate_tiles


def _validate_step_vs_strip(
    tile_size: Sequence[int], overlap: Sequence[int], strip_width: int
) -> None:
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
    statistic: str,
    alpha: float,
    num_bins_per_tile: int,
    rng: np.random.Generator,
) -> ImageReport:
    """Run the per-tile test on one single-channel image slice.

    ``image`` is 2-D ``(H, W)`` or 3-D ``(D, H, W)`` — no batch or channel
    axes. ``tile_size`` and ``overlap`` are per spatial axis. The caller is
    responsible for slicing the per-method ``(N, C, ...)`` prediction down
    to the single-image, single-channel array we operate on.
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

    for tile in tiles_list:
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

        T_obs, p, _ = permutation_pvalue(
            sample.seam_slices,
            sample.control_slices,
            stat_spec=stat_spec,
            block_size=block_size,
            n_permutations=n_permutations,
            rng=rng,
            stat_kwargs=stat_kwargs,
        )

        tile_results.append(
            TileResult(
                coord=tile.coord,
                n_seams=tile.n_seams,
                T_obs=float(T_obs),
                p=float(p),
                n_seam_samples=int(sample.seam_sample.size),
                n_control_samples=int(sample.control_sample.size),
            )
        )

    return aggregate_image(tile_results, alpha)
