"""Per-tile seam / control gradient sampling.

For a tile with one or more owned seams, this module builds:

- ``seam_slices``: for each owned seam, the 1-D array of across-seam
  finite-difference gradients lying *on* the seam line, sliced to the tile's
  range along all parallel axes.
- ``control_slices``: ``2 * strip_width`` strips per seam — ``strip_width``
  on each side of the seam, gradient-index offsets ``{-N, …, -1, +1, …, +N}``
  along the seam's across-seam axis. Each strip is sliced to the same
  parallel range as the seam line.

Slices are kept separate (not pre-concatenated) so the block-permutation
engine in :mod:`permutation` can partition each slice independently — blocks
must not span slice boundaries, otherwise the test would carry spurious
"coherence" between unrelated seams.

The orchestrator is expected to have validated ``step >= 2*strip_width + 1``
along every axis before calling here, so all strip indices are guaranteed to
land inside the gradient arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tilartmetrics.gradient_test.tiles import Tile


@dataclass
class TileGradientSample:
    """Per-tile seam and control gradient samples, kept as per-slice 1-D arrays.

    Attributes
    ----------
    seam_slices : list of np.ndarray
        One 1-D array per owned seam, of across-seam gradients along the seam line.
    control_slices : list of np.ndarray
        ``2 * strip_width`` 1-D arrays per owned seam, one per control-strip offset.
    seam_axes : list of int
        Spatial-axis index of each entry in ``seam_slices`` (parallel list).
    control_axes : list of int
        Spatial-axis index of each entry in ``control_slices`` (parallel list).
    """

    seam_slices: list[np.ndarray]
    control_slices: list[np.ndarray]
    seam_axes: list[int]
    control_axes: list[int]

    @property
    def seam_sample(self) -> np.ndarray:
        """Concatenate all seam slices into a single 1-D array.

        Returns
        -------
        np.ndarray
            Concatenated seam values, or an empty array if no slices were collected.
        """
        if not self.seam_slices:
            return np.array([], dtype=np.float64)
        return np.concatenate(self.seam_slices)

    @property
    def control_sample(self) -> np.ndarray:
        """Concatenate all control slices into a single 1-D array.

        Returns
        -------
        np.ndarray
            Concatenated control values, or an empty array if no slices were collected.
        """
        if not self.control_slices:
            return np.array([], dtype=np.float64)
        return np.concatenate(self.control_slices)


def _slice_along(
    gradient: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    axis: int,
    grad_idx: int,
) -> np.ndarray:
    """Slice a per-axis gradient at a fixed across-axis index and the tile ranges.

    ``gradient`` is a per-axis gradient array whose shape matches the image
    except that axis ``axis`` is shorter by 1 (finite differences along that
    axis). ``ranges`` are image-pixel ranges per spatial axis; we apply them
    verbatim to every axis except ``axis``, where we instead pick the single
    gradient index. The result is flattened to 1-D.

    Parameters
    ----------
    gradient : np.ndarray
        Finite-difference gradient array along ``axis``.
    ranges : tuple of (int, int)
        Per-axis ``(lo, hi)`` image-pixel ranges of the tile.
    axis : int
        Spatial axis index to fix at ``grad_idx``.
    grad_idx : int
        Gradient-array index along ``axis``.

    Returns
    -------
    np.ndarray
        Contiguous 1-D array of sliced gradient values.
    """
    sl: list[slice | int] = []
    for a in range(gradient.ndim):
        if a == axis:
            sl.append(int(grad_idx))
        else:
            lo, hi = ranges[a]
            sl.append(slice(int(lo), int(hi)))
    return np.ascontiguousarray(gradient[tuple(sl)]).ravel()


def group_by_axis(
    slices: list[np.ndarray], axes: list[int]
) -> dict[int, list[np.ndarray]]:
    """Group parallel ``(slice, axis)`` lists into a per-axis dict.

    Parameters
    ----------
    slices : list of np.ndarray
        1-D gradient slices (seam or control).
    axes : list of int
        Spatial-axis index of each entry in ``slices`` (parallel list).

    Returns
    -------
    dict of int to list of np.ndarray
        Mapping ``axis -> list of slices`` belonging to that axis.
    """
    out: dict[int, list[np.ndarray]] = {}
    for s, a in zip(slices, axes, strict=True):
        out.setdefault(a, []).append(s)
    return out


def sample_tile(
    gradients: tuple[np.ndarray, ...],
    tile: Tile,
    strip_width: int,
) -> TileGradientSample:
    """Build the per-tile seam/control sample.

    For each owned seam, collect the 1-D across-seam gradient slice lying on
    the seam line plus ``2 * strip_width`` parallel control strips at offsets
    ``{-N, ..., -1, +1, ..., +N}`` along the seam's across-axis.

    Pre-condition: along every owned-seam axis, ``step >= 2*strip_width + 1``
    so that all ``2 * strip_width`` strip offsets land inside the gradient.
    The orchestrator enforces this.

    Parameters
    ----------
    gradients : tuple of np.ndarray
        Per-axis finite-difference gradient arrays. For 2D this is
        ``(g_y, g_x)`` with shapes ``(H-1, W)`` and ``(H, W-1)``; for 3D
        ``(g_z, g_y, g_x)`` with the analogous shapes.
    tile : Tile
        Tile whose owned seams drive the sampling.
    strip_width : int
        Half-width ``N`` of the control strip around each seam.

    Returns
    -------
    TileGradientSample
        Container with the per-slice seam and control arrays.
    """
    seam_slices: list[np.ndarray] = []
    control_slices: list[np.ndarray] = []
    seam_axes: list[int] = []
    control_axes: list[int] = []

    for seam in tile.seams:
        a = seam.axis
        g = gradients[a]
        g_idx = seam.grad_idx

        seam_slices.append(_slice_along(g, tile.ranges, a, g_idx))
        seam_axes.append(a)

        for offset in range(-strip_width, strip_width + 1):
            if offset == 0:
                continue
            control_slices.append(
                _slice_along(g, tile.ranges, a, g_idx + offset)
            )
            control_axes.append(a)

    return TileGradientSample(
        seam_slices=seam_slices,
        control_slices=control_slices,
        seam_axes=seam_axes,
        control_axes=control_axes,
    )
