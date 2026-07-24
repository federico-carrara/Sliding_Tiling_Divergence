"""Curve-to-scalar reductions for FRC.

Hosts :func:`frc_resolution` — the standard threshold-crossing resolution
readout — plus :func:`frc_to_scalar`, the *seam-artifact* scalar, which stays
DEFERRED: the choice between

    - dip depth at expected harmonics ``k/S``, or
    - FRC-AUC (integral from 0 to Nyquist), or
    - some third alternative

depends on what the actual averaged curves look like on the test datasets.
See ``agents_artifacts/FRC_metric.md`` §3.7 for that design discussion.
"""

from __future__ import annotations

import numpy as np

FRC_THRESHOLD_1_7 = 1.0 / 7.0
"""Conventional FRC/FSC resolution threshold (``1/7 ≈ 0.143``).

The frequency at which an FRC curve falls below this value is quoted as the
resolution (van Heel & Schatz 2005 for cryo-EM FSC; Nieuwenhuizen et al. 2013,
Nat. Methods, for SMLM FRC; Koho et al. 2019, Nat. Commun. — the MIPLIB paper
this package's ``frc.py`` credits).

Caveat for this codebase: the 1/7 value is derived for FRC between two
*independent, equally-noisy half-datasets* of the same specimen. Here we
correlate a prediction against a (clean) ground truth, which is a *fidelity*
curve rather than a two-noisy-halves curve, so the crossing frequency should be
read as a conventional, consistent cutoff for **ranking methods on the same
data** — not quoted as a physical instrument resolution.
"""


def frc_resolution(
    freqs: np.ndarray,
    frc: np.ndarray,
    threshold: float = FRC_THRESHOLD_1_7,
) -> float:
    """Frequency at which an FRC curve first drops below ``threshold``.

    The curve is scanned from low to high frequency (the DC bin is skipped, and
    ``NaN`` rings are ignored); the first bin strictly below ``threshold`` marks
    the crossing, which is then refined by linear interpolation between that bin
    and its predecessor. Taking the *first* crossing is the usual convention —
    real curves are noisy and may re-cross, so this is the conservative readout.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency-bin centres in cycles/pixel, shape ``(n_bins,)``.
    frc : np.ndarray
        Matching per-bin FRC values, same shape as ``freqs``.
    threshold : float, default=``1/7``
        Correlation value defining the crossing (see
        :data:`FRC_THRESHOLD_1_7`).

    Returns
    -------
    float
        Crossing frequency in cycles/pixel. ``nan`` when the curve never falls
        below ``threshold`` within the measured band (resolution beyond the
        sampled range), when it is already below at the first non-DC bin
        (degenerate — no correlated band), or when there are too few valid bins.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    frc = np.asarray(frc, dtype=np.float64)
    if freqs.shape != frc.shape or frc.size < 2:
        return float("nan")

    # Skip the DC bin (bin 0 is the image mean, not structure) and NaN rings.
    idx = np.arange(1, frc.size)
    idx = idx[~np.isnan(frc[idx])]
    if idx.size < 2:
        return float("nan")
    f, c = freqs[idx], frc[idx]

    below = np.flatnonzero(c < threshold)
    if below.size == 0:
        return float("nan")  # never crosses within the measured band
    j = int(below[0])
    if j == 0:
        return float("nan")  # already below at the first non-DC bin

    f_lo, c_lo = f[j - 1], c[j - 1]
    f_hi, c_hi = f[j], c[j]
    if c_lo == c_hi:  # flat segment; no unique crossing
        return float(f_hi)
    t = (c_lo - threshold) / (c_lo - c_hi)  # in [0, 1] since c_lo >= thr > c_hi
    return float(f_lo + t * (f_hi - f_lo))


def frc_resolution_period(
    freqs: np.ndarray,
    frc: np.ndarray,
    threshold: float = FRC_THRESHOLD_1_7,
) -> float:
    """Resolution as a period in pixels — ``1 / frc_resolution(...)``.

    Parameters
    ----------
    freqs, frc, threshold
        Forwarded to :func:`frc_resolution`.

    Returns
    -------
    float
        Smallest resolved feature period in pixels (``nan`` when the crossing
        frequency is ``nan`` or zero). Multiply by the pixel size to quote a
        physical length.
    """
    f_c = frc_resolution(freqs, frc, threshold)
    if not np.isfinite(f_c) or f_c <= 0.0:
        return float("nan")
    return float(1.0 / f_c)

