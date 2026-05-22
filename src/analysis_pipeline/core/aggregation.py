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
    """Outcome of the permutation test for one kept-region tile."""

    coord: tuple[int, ...]
    n_seams: int
    T_obs: float
    p: float
    n_seam_samples: int
    n_control_samples: int


@dataclass
class ImageReport:
    """Per-image roll-up of tile-level scores."""

    tiles: list[TileResult]
    median_T: float
    frac_rejected: float


@dataclass
class MethodReport:
    """Per-method roll-up across all images for that method."""

    images: list[ImageReport] = field(default_factory=list)
    mean_median_T: float = float("nan")
    mean_frac_rejected: float = float("nan")


@dataclass
class MultiMethodReport:
    """Top-level result of a multi-method per-tile run."""

    methods: dict[str, MethodReport] = field(default_factory=dict)
    config_summary: Optional[dict] = None


def aggregate_image(
    tiles: list[TileResult],
    alpha: float,
) -> ImageReport:
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
