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
    """Build joint per-tile bin edges over the concatenated samples.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.
    num_bins : int
        Number of bins.

    Returns
    -------
    np.ndarray
        ``(num_bins + 1,)`` array of bin edges.
    """
    return np.histogram_bin_edges(
        np.concatenate([seam, control]), bins=num_bins
    )


def _hist(sample: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute a float64 histogram for ``sample`` over ``bin_edges``.

    Parameters
    ----------
    sample : np.ndarray
        1-D array of values.
    bin_edges : np.ndarray
        Bin edges.

    Returns
    -------
    np.ndarray
        Float64 histogram counts.
    """
    return np.histogram(sample, bins=bin_edges)[0].astype(np.float64)


def kl(
    seam: np.ndarray,
    control: np.ndarray,
    *,
    num_bins: int = 32,
    bin_edges: Optional[np.ndarray] = None,
) -> float:
    """Compute ``KL(seam || control)`` on histograms with joint per-tile binning.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.
    num_bins : int, default=32
        Number of bins (used only if ``bin_edges`` is None).
    bin_edges : np.ndarray, optional
        Precomputed bin edges. If None, joint edges are built on the fly.

    Returns
    -------
    float
        KL divergence value.
    """
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
    """Compute the Jensen–Shannon divergence on joint-binned histograms.

    Uses the natural-log convention.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.
    num_bins : int, default=32
        Number of bins (used only if ``bin_edges`` is None).
    bin_edges : np.ndarray, optional
        Precomputed bin edges. If None, joint edges are built on the fly.

    Returns
    -------
    float
        Jensen–Shannon divergence value.
    """
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
    """Compute the two-sample Kolmogorov–Smirnov statistic ``D``.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.

    Returns
    -------
    float
        KS statistic ``D`` (no p-value); ``0.0`` if either sample is empty.
    """
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
    """Compute the 1-D Wasserstein-1 (earth mover's) distance.

    Thin wrapper around :func:`scipy.stats.wasserstein_distance`.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.

    Returns
    -------
    float
        Wasserstein-1 distance.
    """
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(seam, control))


def mean_abs_ratio(seam: np.ndarray, control: np.ndarray) -> float:
    """Compute the Pan et al. (2004) inter/intra ratio ``mean(|s|) / mean(|c|)``.

    Parameters
    ----------
    seam : np.ndarray
        Seam-side 1-D samples.
    control : np.ndarray
        Control-side 1-D samples.

    Returns
    -------
    float
        Ratio of mean absolute seam to mean absolute control.
    """
    m_s = float(np.mean(np.abs(seam))) if seam.size else 0.0
    m_c = float(np.mean(np.abs(control))) if control.size else 0.0
    return m_s / (m_c + EPS)


VecKind = Literal["binned", "ks", "abs_ratio", "scalar"]


@dataclass(frozen=True)
class StatisticSpec:
    """Registry entry describing a two-sample statistic.

    Attributes
    ----------
    name : str
        Statistic name used by the registry.
    fn : Callable[..., float]
        Callable returning a scalar discrepancy from ``(seam, control)``.
    vec_kind : {"binned", "ks", "abs_ratio", "scalar"}
        Vectorization kind dispatched by the permutation engine.
    default_kwargs : dict
        Default keyword arguments forwarded to ``fn``.
    """

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
    """Look up a registered :class:`StatisticSpec` by name.

    Parameters
    ----------
    name : str
        Statistic name (e.g. ``"kl"``, ``"js"``, ``"ks"``, ``"wasserstein"``,
        ``"mean_abs_ratio"``).

    Returns
    -------
    StatisticSpec
        Registered specification for the requested statistic.

    Raises
    ------
    ValueError
        If ``name`` is not in :data:`STATISTICS`.
    """
    if name not in STATISTICS:
        raise ValueError(
            f"unknown statistic {name!r}; available: {sorted(STATISTICS)}"
        )
    return STATISTICS[name]
