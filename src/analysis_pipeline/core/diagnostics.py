"""Anisotropy diagnostic for direction pooling.

The per-tile metric pools control gradients across the spatial axes of each
tile (horizontal seams contribute ``g_y`` samples, vertical seams contribute
``g_x``, etc.) under the assumption that the underlying gradient statistics
are approximately isotropic. This diagnostic spot-checks that assumption by
running a two-sample KS test, on a small random subset of tiles, between the
first two control-axis samples available within each tile.

It is intentionally coarse: analytical KS p-values are anti-conservative
under local pixel dependence, so we use them only to flag a clear failure
(median p well below ``alpha`` and most tested tiles rejecting), not as a
calibrated estimate of anisotropy.
"""

from __future__ import annotations

import warnings

import numpy as np

from .aggregation import AnisotropyReport
from .sampling import TileSample


def anisotropy_diagnostic(
    tile_samples: list[TileSample],
    rng: np.random.Generator,
    *,
    n_tested: int,
    alpha: float = 0.05,
) -> AnisotropyReport:
    """Return a coarse direction-pooling sanity check.

    Picks ``n_tested`` tiles uniformly without replacement and runs
    ``scipy.stats.ks_2samp`` on the per-axis control samples of the first
    two axes that carry data. Tiles with fewer than two such axes are
    skipped.
    """
    if n_tested <= 0 or not tile_samples:
        return AnisotropyReport(
            n_tested=0, median_p=float("nan"), n_significant=0, alpha=alpha
        )

    from scipy.stats import ks_2samp

    n_sample = min(n_tested, len(tile_samples))
    idx = rng.choice(len(tile_samples), size=n_sample, replace=False)

    pvals: list[float] = []
    for i in idx:
        ts = tile_samples[int(i)]
        axes_with_data = sorted(set(ts.control_axes))
        if len(axes_with_data) < 2:
            continue
        a, b = axes_with_data[0], axes_with_data[1]
        ctrl_a = ts.per_axis_control(a)
        ctrl_b = ts.per_axis_control(b)
        if ctrl_a.size < 2 or ctrl_b.size < 2:
            continue
        _, p = ks_2samp(ctrl_a, ctrl_b)
        pvals.append(float(p))

    if not pvals:
        return AnisotropyReport(
            n_tested=0, median_p=float("nan"), n_significant=0, alpha=alpha
        )

    arr = np.array(pvals, dtype=np.float64)
    n_sig = int(np.sum(arr < alpha))
    median_p = float(np.median(arr))

    if n_sig / len(pvals) > 0.2:
        warnings.warn(
            f"Anisotropy diagnostic: {n_sig}/{len(pvals)} sampled tiles "
            f"reject the equality of axis-0 vs axis-1 control distributions "
            f"at alpha={alpha}. Direction pooling may be inappropriate for "
            "this dataset.",
            stacklevel=2,
        )

    return AnisotropyReport(
        n_tested=len(pvals),
        median_p=median_p,
        n_significant=n_sig,
        alpha=alpha,
    )
