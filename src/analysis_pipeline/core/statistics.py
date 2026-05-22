"""Two-sample discrepancy statistics used by the per-tile permutation test.

Each statistic accepts two 1-D ``numpy`` samples ``(seam, control)`` and
returns a scalar score. A registry exposes per-statistic metadata so the
permutation engine can dispatch a vectorized fast path when one exists
(``vec_kind``: ``"binned"`` for KL/JS, ``"ks"`` for KS, ``"abs_ratio"`` for
the Pan-et-al. ratio, ``"scalar"`` for everything else).

Binning convention for KL/JS: per-tile joint bin edges built from
``np.concatenate([seam, control])``. The same edges are used for the
observed statistic and all permutations within that tile, so per-tile
comparability is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np

EPS = 1e-12


def _joint_bin_edges(
    seam: np.ndarray, control: np.ndarray, num_bins: int
) -> np.ndarray:
    return np.histogram_bin_edges(
        np.concatenate([seam, control]), bins=num_bins
    )


def _hist(sample: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.histogram(sample, bins=bin_edges)[0].astype(np.float64)


def kl(
    seam: np.ndarray,
    control: np.ndarray,
    *,
    num_bins: int = 32,
    bin_edges: Optional[np.ndarray] = None,
) -> float:
    """``KL(seam ‖ control)`` on histograms with joint per-tile binning."""
    if bin_edges is None:
        bin_edges = _joint_bin_edges(seam, control, num_bins)
    p = _hist(seam, bin_edges)
    q = _hist(control, bin_edges)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    return float(np.sum(p * np.log((p + EPS) / (q + EPS))))


def js(
    seam: np.ndarray,
    control: np.ndarray,
    *,
    num_bins: int = 32,
    bin_edges: Optional[np.ndarray] = None,
) -> float:
    """Jensen–Shannon divergence (natural log) on joint-binned histograms."""
    if bin_edges is None:
        bin_edges = _joint_bin_edges(seam, control, num_bins)
    p = _hist(seam, bin_edges)
    q = _hist(control, bin_edges)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + EPS) / (m + EPS)))
    kl_qm = np.sum(q * np.log((q + EPS) / (m + EPS)))
    return float(0.5 * (kl_pm + kl_qm))


def ks(seam: np.ndarray, control: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (D only, no p-value)."""
    seam_s = np.sort(seam)
    control_s = np.sort(control)
    n_s, n_c = seam_s.size, control_s.size
    if n_s == 0 or n_c == 0:
        return 0.0
    combined = np.concatenate([seam_s, control_s])
    grid = np.sort(np.unique(combined))
    cdf_s = np.searchsorted(seam_s, grid, side="right") / n_s
    cdf_c = np.searchsorted(control_s, grid, side="right") / n_c
    return float(np.max(np.abs(cdf_s - cdf_c)))


def wasserstein(seam: np.ndarray, control: np.ndarray) -> float:
    """1-D Wasserstein-1 (earth mover's) distance via ``scipy``."""
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(seam, control))


def mean_abs_ratio(seam: np.ndarray, control: np.ndarray) -> float:
    """``mean(|seam|) / mean(|control|)`` — Pan et al. (2004) inter/intra ratio."""
    m_s = float(np.mean(np.abs(seam))) if seam.size else 0.0
    m_c = float(np.mean(np.abs(control))) if control.size else 0.0
    return m_s / (m_c + EPS)


VecKind = Literal["binned", "ks", "abs_ratio", "scalar"]


@dataclass(frozen=True)
class StatisticSpec:
    name: str
    fn: Callable[..., float]
    vec_kind: VecKind
    default_kwargs: dict = field(default_factory=dict)


STATISTICS: dict[str, StatisticSpec] = {
    "kl": StatisticSpec("kl", kl, "binned", {"num_bins": 32}),
    "js": StatisticSpec("js", js, "binned", {"num_bins": 32}),
    "ks": StatisticSpec("ks", ks, "ks"),
    "mean_abs_ratio": StatisticSpec(
        "mean_abs_ratio", mean_abs_ratio, "abs_ratio"
    ),
    "wasserstein": StatisticSpec("wasserstein", wasserstein, "scalar"),
}


def get_statistic(name: str) -> StatisticSpec:
    if name not in STATISTICS:
        raise ValueError(
            f"unknown statistic {name!r}; available: {sorted(STATISTICS)}"
        )
    return STATISTICS[name]
