"""Per-image and per-method aggregation of per-tile results.

Per-tile we record ``T_obs`` and ``p``; per-image we summarize with the
median ``T`` and the fraction of tiles rejecting at ``alpha`` (NaN tiles —
those with insufficient seams — excluded from both); per-method we average
the per-image scalars across the test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TileResult:
    """Outcome of the permutation test for one kept-region tile.

    Attributes
    ----------
    coord : tuple of int
        Multi-index in the kept-region grid.
    n_seams : int
        Number of seams owned by the tile.
    T_obs : float
        Observed statistic value (``nan`` for skipped tiles).
    p : float
        Phipson–Smyth p-value (``nan`` for skipped tiles).
    n_seam_samples : int
        Total number of seam samples used in the test.
    n_control_samples : int
        Total number of control samples used in the test.
    """

    coord: tuple[int, ...]
    n_seams: int
    T_obs: float
    p: float
    n_seam_samples: int
    n_control_samples: int


@dataclass
class ImageReport:
    """Per-image roll-up of tile-level scores.

    Attributes
    ----------
    tiles : list of TileResult
        Per-tile outcomes (including skipped tiles).
    median_T : float
        Median of valid ``T_obs`` across tiles (``nan`` if none are valid).
    frac_rejected : float
        Fraction of valid tiles with ``p < alpha`` (``nan`` if none are valid).
    """

    tiles: list[TileResult]
    median_T: float
    frac_rejected: float


@dataclass
class MethodReport:
    """Per-method roll-up across all images for that method.

    Attributes
    ----------
    images : list of ImageReport
        Per-image reports for this method.
    mean_median_T : float
        Mean of valid per-image ``median_T`` values.
    mean_frac_rejected : float
        Mean of valid per-image ``frac_rejected`` values.
    """

    images: list[ImageReport] = field(default_factory=list)
    mean_median_T: float = float("nan")
    mean_frac_rejected: float = float("nan")


@dataclass
class MultiMethodReport:
    """Top-level result of a multi-method per-tile run.

    Attributes
    ----------
    methods : dict of str to MethodReport
        Per-method reports keyed by method name.
    config_summary : dict, optional
        Snapshot of the run configuration.
    """

    methods: dict[str, MethodReport] = field(default_factory=dict)
    config_summary: Optional[dict] = None


def aggregate_image(
    tiles: list[TileResult],
    alpha: float,
) -> ImageReport:
    """Aggregate per-tile outcomes into an :class:`ImageReport`.

    Tiles with ``NaN`` ``T_obs`` or ``p`` (insufficient seams) are excluded
    from the median and rejection-rate computations.

    Parameters
    ----------
    tiles : list of TileResult
        Per-tile outcomes for the image.
    alpha : float
        Rejection threshold.

    Returns
    -------
    ImageReport
        Per-image roll-up.
    """
    valid_T = np.array(
        [t.T_obs for t in tiles if not np.isnan(t.T_obs)], dtype=np.float64
    )
    valid_p = np.array(
        [t.p for t in tiles if not np.isnan(t.p)], dtype=np.float64
    )
    median_T = float(np.median(valid_T)) if valid_T.size else float("nan")
    frac_rejected = (
        float(np.mean(valid_p < alpha)) if valid_p.size else float("nan")
    )
    return ImageReport(
        tiles=tiles,
        median_T=median_T,
        frac_rejected=frac_rejected,
    )


def aggregate_method(images: list[ImageReport]) -> MethodReport:
    """Aggregate per-image reports into a per-method :class:`MethodReport`.

    Per-image scalars that are ``NaN`` are excluded from the mean.

    Parameters
    ----------
    images : list of ImageReport
        Per-image reports for the method.

    Returns
    -------
    MethodReport
        Per-method roll-up; an empty report if ``images`` is empty.
    """
    if not images:
        return MethodReport()
    medians = np.array(
        [im.median_T for im in images if not np.isnan(im.median_T)],
        dtype=np.float64,
    )
    fracs = np.array(
        [im.frac_rejected for im in images if not np.isnan(im.frac_rejected)],
        dtype=np.float64,
    )
    return MethodReport(
        images=images,
        mean_median_T=float(np.mean(medians)) if medians.size else float("nan"),
        mean_frac_rejected=float(np.mean(fracs)) if fracs.size else float("nan"),
    )
