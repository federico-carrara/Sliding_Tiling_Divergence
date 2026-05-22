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

Per-axis labelling is preserved so the anisotropy diagnostic can split the
control sample by seam axis.

The orchestrator is expected to have validated ``step >= 2*strip_width + 2``
along every axis before calling here, so all strip indices are guaranteed to
land inside the gradient arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tiles import Tile


@dataclass
class TileSample:
    """Per-tile seam and control samples, kept as per-slice 1-D arrays."""

    seam_slices: list[np.ndarray]
    seam_axes: list[int]
    control_slices: list[np.ndarray]
    control_axes: list[int]

    @property
    def seam_sample(self) -> np.ndarray:
        if not self.seam_slices:
            return np.array([], dtype=np.float64)
        return np.concatenate(self.seam_slices)

    @property
    def control_sample(self) -> np.ndarray:
        if not self.control_slices:
            return np.array([], dtype=np.float64)
        return np.concatenate(self.control_slices)

    def per_axis_control(self, axis: int) -> np.ndarray:
        arrs = [s for s, a in zip(self.control_slices, self.control_axes) if a == axis]
        if not arrs:
            return np.array([], dtype=np.float64)
        return np.concatenate(arrs)


def _slice_along(
    g: np.ndarray,
    ranges: tuple[tuple[int, int], ...],
    axis: int,
    grad_idx: int,
) -> np.ndarray:
    """Return ``g`` with ``axis`` fixed to ``grad_idx`` and the other axes
    sliced to ``ranges``. The result is flattened to 1-D.

    ``g`` is a per-axis gradient array whose shape matches the image except
    that axis ``axis`` is shorter by 1 (finite differences along that axis).
    ``ranges`` are image-pixel ranges per spatial axis; we apply them
    verbatim to every axis except ``axis``, where we instead pick the single
    gradient index.
    """
    sl: list[slice | int] = []
    for a in range(g.ndim):
        if a == axis:
            sl.append(int(grad_idx))
        else:
            lo, hi = ranges[a]
            sl.append(slice(int(lo), int(hi)))
    return np.ascontiguousarray(g[tuple(sl)]).ravel()


def sample_tile(
    gradients: tuple[np.ndarray, ...],
    tile: Tile,
    strip_width: int,
) -> TileSample:
    """Build the per-tile seam/control sample.

    ``gradients[a]`` is the finite-difference gradient along spatial axis
    ``a``, with the differenced axis one element shorter than the image. For
    2D this is ``(g_y, g_x)`` with shapes ``(H-1, W)``, ``(H, W-1)``; for 3D
    ``(g_z, g_y, g_x)`` with the analogous shapes.

    Pre-condition: along every owned-seam axis, ``step >= 2*strip_width + 2``
    so that all ``2 * strip_width`` strip offsets land inside the gradient.
    The orchestrator enforces this.
    """
    seam_slices: list[np.ndarray] = []
    seam_axes: list[int] = []
    control_slices: list[np.ndarray] = []
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

    return TileSample(
        seam_slices=seam_slices,
        seam_axes=seam_axes,
        control_slices=control_slices,
        control_axes=control_axes,
    )
