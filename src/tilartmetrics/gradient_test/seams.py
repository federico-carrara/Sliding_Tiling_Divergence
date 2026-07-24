"""Seam locator for CAREamics's ``TiledPatching`` strategy.

Pure functions that derive stitching-seam positions from the
``(tile_size, overlap, axis_size)`` parameters of a tiled prediction. No
runtime CAREamics dependency — the closed form is taken from
``careamics/dataset/patching/tiled_patching.py::_compute_1d_coords``.

Closed form, per axis:

- ``step = tile_size - overlap``, ``M = overlap // 2``
- single-tile fallback when ``axis_size <= tile_size`` (no seams)
- otherwise ``N = ceil((axis_size - overlap) / step)`` tiles and seams at
  ``{k * step + M : k = 1, ..., N - 1}`` in image-pixel coordinates.
"""

from __future__ import annotations

import math
import warnings

import numpy as np


def compute_seam_positions(
    axis_size: int, tile_size: int, overlap: int
) -> np.ndarray:
    """Compute image-pixel positions of stitching seams along one axis.

    A seam is the boundary between two adjacent kept regions in the stitched
    image — equivalently, the first pixel of tile ``k`` (``k >= 1``) in the
    output. Returns an empty array (and emits a warning) when the axis fits
    in a single tile.

    Parameters
    ----------
    axis_size : int
        Size of the image along the axis (image-pixel units).
    tile_size : int
        TiledPatching tile size along the axis.
    overlap : int
        TiledPatching overlap along the axis.

    Returns
    -------
    np.ndarray
        1-D integer array of seam positions in image-pixel coordinates; empty
        if ``axis_size <= tile_size``.
    """
    if axis_size <= tile_size:
        warnings.warn(
            f"axis_size={axis_size} <= tile_size={tile_size}: the patching "
            "strategy emits a single tile with no seams; skipping this axis.",
            stacklevel=2,
        )
        return np.array([], dtype=int)

    step = tile_size - overlap
    M = overlap // 2
    N = math.ceil((axis_size - overlap) / step)
    return np.array([k * step + M for k in range(1, N)], dtype=int)


def assert_shape_consistent(
    axis_size: int, tile_size: int, overlap: int, axis_label: str
) -> None:
    """Assert ``axis_size`` matches a clean TiledPatching tiling.

    The clean regime has ``(axis_size - tile_size)`` divisible by
    ``step = tile_size - overlap`` — equivalently
    ``axis_size == (N - 1) * step + tile_size``. Irregular last-tile tilings
    are rejected; the metric is only meaningful when the tiling produced the
    prediction at hand.

    Single-tile axes (``axis_size <= tile_size``) are not checked here; they
    are handled by :func:`compute_seam_positions` via warn-and-skip.

    Parameters
    ----------
    axis_size : int
        Size of the image along the axis (image-pixel units).
    tile_size : int
        TiledPatching tile size along the axis.
    overlap : int
        TiledPatching overlap along the axis.
    axis_label : str
        Human-readable axis label used in the assertion message.

    Raises
    ------
    AssertionError
        If ``axis_size`` does not match the clean tiling formula.
    """
    if axis_size <= tile_size:
        return
    step = tile_size - overlap
    N = math.ceil((axis_size - overlap) / step)
    expected = (N - 1) * step + tile_size
    assert axis_size == expected, (
        f"axis {axis_label!r}: axis_size={axis_size} is inconsistent with a "
        f"clean TiledPatching tiling (tile_size={tile_size}, "
        f"overlap={overlap}); expected axis_size={expected}."
    )


def pixel_positions_to_grad_indices(positions: np.ndarray) -> np.ndarray:
    """Convert image-pixel positions to gradient-array indices.

    Finite-difference convention:
    ``grad[..., p, ...] = imgs[..., p + 1, ...] - imgs[..., p, ...]``.
    A seam at pixel ``j`` (the step between pixel ``j - 1`` and pixel
    ``j``) therefore sits at gradient index ``j - 1``.

    Parameters
    ----------
    positions : np.ndarray
        Image-pixel positions (integer array of any shape).

    Returns
    -------
    np.ndarray
        Gradient-array indices (``positions - 1``).
    """
    return positions - 1
