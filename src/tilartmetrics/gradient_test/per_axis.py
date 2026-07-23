"""Per-axis normalization and count balancing for the per-tile permutation test.

Two anisotropy corrections applied before the seam-vs-control statistic, so that
gradients from different spatial axes can be pooled into a single test:

- **Per-axis mean/std normalization** removes the cross-axis *scale* difference
  (e.g. Z gradients being ~3x larger than X/Y). Statistics are computed per axis
  over *all* of that axis's samples with seam and control pooled together — never
  separately, which would erase the seam-vs-control signal the test measures. The
  scope is the whole image/channel (all tiles), so :class:`AxisMoments` accumulates
  running moments across a first pass before normalization in a second pass.

- **Per-axis count balancing** removes the cross-axis *sample-count* imbalance. In a
  thin-Z kept region a Z-seam line spans the large Y-X face while X/Y-seam lines span
  the thin Z face, so Z contributes far more samples and would dominate the pooled
  histogram. :func:`balance_axis_blocks` subsamples, at ``block_size``-block
  granularity, so each present axis contributes an equal number of blocks.

Balancing pools blocks from different axes into one permutation null, which is only
exchangeable under H0 when the axes share a common scale — so it is only valid
together with normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis_pipeline.gradient_test.permutation import _split_into_blocks

# Floor on the per-axis std, mirroring per_tile.Z_VAR_FLOOR: a flat axis
# (std ~ 0) would otherwise blow up the normalized values.
STD_FLOOR = 1e-8


@dataclass
class AxisMoments:
    """Running per-axis first/second moments over pooled seam+control samples.

    Accumulating ``n``, ``sum`` and ``sum of squares`` (rather than storing all
    values) gives exact per-axis mean/std with negligible memory.

    Attributes
    ----------
    n : np.ndarray
        Per-axis sample count.
    s : np.ndarray
        Per-axis sum of values.
    s2 : np.ndarray
        Per-axis sum of squared values.
    """

    n: np.ndarray
    s: np.ndarray
    s2: np.ndarray

    @classmethod
    def zeros(cls, n_axes: int) -> "AxisMoments":
        """Create a zero-initialised accumulator for ``n_axes`` spatial axes.

        Parameters
        ----------
        n_axes : int
            Number of spatial axes (2 or 3).

        Returns
        -------
        AxisMoments
            Accumulator with all counts/sums zeroed.
        """
        return cls(
            n=np.zeros(n_axes, dtype=np.int64),
            s=np.zeros(n_axes, dtype=np.float64),
            s2=np.zeros(n_axes, dtype=np.float64),
        )

    def update(self, axis: int, values: np.ndarray) -> None:
        """Fold one axis's slice of values into the accumulator.

        Parameters
        ----------
        axis : int
            Spatial-axis index the values belong to.
        values : np.ndarray
            1-D gradient values (seam or control) for that axis.
        """
        v = values.astype(np.float64, copy=False)
        self.n[axis] += v.size
        self.s[axis] += v.sum()
        self.s2[axis] += np.square(v).sum()

    def finalize(self, std_floor: float = STD_FLOOR) -> dict[int, tuple[float, float]]:
        """Compute per-axis ``(mean, std)`` for axes with any samples.

        Parameters
        ----------
        std_floor : float, default=STD_FLOOR
            Lower bound applied to each axis's std to avoid division by ~0.

        Returns
        -------
        dict of int to (float, float)
            Mapping ``axis -> (mean, std)``; axes with zero samples are omitted.
        """
        stats: dict[int, tuple[float, float]] = {}
        for a in range(len(self.n)):
            if self.n[a] == 0:
                continue
            mean = self.s[a] / self.n[a]
            var = self.s2[a] / self.n[a] - mean * mean
            std = float(np.sqrt(max(var, 0.0)))
            stats[a] = (float(mean), max(std, std_floor))
        return stats


def normalize_slices(
    slices: list[np.ndarray],
    axes: list[int],
    stats: dict[int, tuple[float, float]],
) -> list[np.ndarray]:
    """Standardize each slice by its axis's ``(mean, std)``.

    Parameters
    ----------
    slices : list of np.ndarray
        1-D gradient slices (seam or control).
    axes : list of int
        Spatial-axis index of each entry in ``slices`` (parallel list).
    stats : dict of int to (float, float)
        Per-axis ``(mean, std)`` from :meth:`AxisMoments.finalize`. An axis absent
        from ``stats`` is left unchanged (identity).

    Returns
    -------
    list of np.ndarray
        Normalized slices, in the same order as ``slices``.
    """
    out: list[np.ndarray] = []
    for s, a in zip(slices, axes, strict=True):
        mean, std = stats.get(a, (0.0, 1.0))
        out.append((s - mean) / std)
    return out


def balance_axis_blocks(
    slices_by_axis: dict[int, list[np.ndarray]],
    *,
    block_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Subsample to an equal number of ``block_size`` blocks per axis.

    Each axis's slices are split into contiguous blocks (reusing the permutation
    engine's splitter, so blocks match exactly what the test would build). The axis
    with the fewest blocks sets the target ``m``; every other axis keeps a random
    ``m`` of its blocks. An axis already at ``m`` keeps all its blocks *without*
    drawing from ``rng``, so a no-op balancing (e.g. equal-count 2-D tiles or
    single-axis tiles) leaves the rng stream untouched.

    Returns the selected blocks as a flat list; passing them as the seam (or
    control) argument to :func:`permutation_pvalue` re-emits exactly one block per
    entry, so the block partition is identical to the unbalanced one minus the
    dropped blocks.

    Parameters
    ----------
    slices_by_axis : dict of int to list of np.ndarray
        Per-axis 1-D slices (from :func:`sampling.group_by_axis`).
    block_size : int
        Contiguous-block size ``B`` (same value used by the permutation engine).
    rng : numpy.random.Generator
        Generator used to choose which blocks to keep.

    Returns
    -------
    list of np.ndarray
        Selected blocks (each length ``<= block_size``), flattened across axes.
    """
    blocks_by_axis = {
        a: _split_into_blocks(sl, block_size) for a, sl in slices_by_axis.items()
    }
    blocks_by_axis = {a: b for a, b in blocks_by_axis.items() if b}
    if not blocks_by_axis:
        return []

    m = min(len(b) for b in blocks_by_axis.values())

    selected: list[np.ndarray] = []
    for a in sorted(blocks_by_axis):
        blocks = blocks_by_axis[a]
        if len(blocks) == m:
            selected.extend(blocks)  # no-op: do not perturb the rng stream
        else:
            idx = rng.choice(len(blocks), size=m, replace=False)
            selected.extend(blocks[i] for i in idx)
    return selected
