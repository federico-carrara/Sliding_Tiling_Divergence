"""Kept-region (a.k.a. "tile") enumeration for the per-tile metric.

A "tile" in this metric is one kept region of the TiledPatching grid — the
slab between two consecutive seams along each spatial axis (or between the
image edge and the first/last seam at the ends). Each tile owns the up-to-2
seams that bound it along each axis: interior tiles own one seam per face,
boundary tiles own one fewer per touched edge.

The data classes carry the geometry the sampler and permutation engine
need: per-axis pixel ranges and the list of owned seams (axis, pixel
position, finite-difference gradient index).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .seams import compute_seam_positions


@dataclass(frozen=True)
class Seam:
    """One stitching seam owned by a tile along a specific spatial axis.

    Attributes
    ----------
    axis : int
        Spatial-axis index (0=y,1=x in 2D; 0=z,1=y,2=x in 3D).
    pixel : int
        Image-pixel position of the seam along ``axis``.
    grad_idx : int
        Across-seam finite-difference index (``pixel - 1``).
    """

    axis: int
    pixel: int
    grad_idx: int


@dataclass(frozen=True)
class Tile:
    """One kept region and the seams it owns.

    Attributes
    ----------
    coord : tuple of int
        Multi-index in the kept-region grid.
    ranges : tuple of (int, int)
        Per-axis ``(lo, hi)`` pixel ranges.
    seams : tuple of Seam
        Owned seams.
    """

    coord: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...]
    seams: tuple[Seam, ...]

    @property
    def n_seams(self) -> int:
        """Return the number of owned seams.

        Returns
        -------
        int
            Number of seams in ``seams``.
        """
        return len(self.seams)

    @property
    def n_axes(self) -> int:
        """Return the number of spatial axes for this tile.

        Returns
        -------
        int
            Number of entries in ``ranges``.
        """
        return len(self.ranges)


def enumerate_tiles(
    image_shape: Sequence[int],
    tile_size: Sequence[int],
    overlap: Sequence[int],
) -> list[Tile]:
    """Enumerate kept-region tiles for a single image's spatial shape.

    ``image_shape`` is the spatial shape only — ``(H, W)`` for 2D or
    ``(D, H, W)`` for 3D; channel and batch axes belong upstream. Tiles
    along an axis with zero seams (axis fits in a single tile) have no
    owned seams from that axis; callers may filter on ``Tile.n_seams``.

    Parameters
    ----------
    image_shape : Sequence[int]
        Spatial shape of the image, ``(H, W)`` in 2D or ``(D, H, W)`` in 3D.
    tile_size : Sequence[int]
        TiledPatching tile size per spatial axis (image-pixel units).
    overlap : Sequence[int]
        TiledPatching overlap per spatial axis (image-pixel units).

    Returns
    -------
    list of Tile
        Tiles in C-order over the per-axis kept-region grid.

    Raises
    ------
    ValueError
        If ``image_shape``, ``tile_size`` and ``overlap`` do not all have
        the same length.
    """
    if not (len(image_shape) == len(tile_size) == len(overlap)):
        raise ValueError(
            "image_shape, tile_size, overlap must have the same length "
            f"(got {len(image_shape)}, {len(tile_size)}, {len(overlap)})"
        )

    seams_per_axis: list[np.ndarray] = []
    boundaries_per_axis: list[np.ndarray] = []
    for sz, ts, ov in zip(image_shape, tile_size, overlap, strict=True):
        seams = compute_seam_positions(sz, ts, ov)
        boundaries = np.concatenate([[0], seams, [sz]]).astype(int)
        seams_per_axis.append(seams)
        boundaries_per_axis.append(boundaries)

    n_regions = tuple(len(b) - 1 for b in boundaries_per_axis)

    tiles: list[Tile] = []
    for coord in np.ndindex(*n_regions):
        ranges = tuple(
            (
                int(boundaries_per_axis[a][coord[a]]),
                int(boundaries_per_axis[a][coord[a] + 1])
            )
            for a in range(len(coord))
        )
        owned: list[Seam] = []
        for a, i in enumerate(coord):
            seams_a = seams_per_axis[a]
            if seams_a.size == 0:
                continue
            if i > 0:
                px = int(seams_a[i - 1])
                owned.append(Seam(axis=a, pixel=px, grad_idx=px - 1))
            if i < n_regions[a] - 1:
                px = int(seams_a[i])
                owned.append(Seam(axis=a, pixel=px, grad_idx=px - 1))

        tiles.append(Tile(coord=tuple(coord), ranges=ranges, seams=tuple(owned)))

    return tiles
