"""Curve-to-scalar reduction for FRC — DEFERRED placeholder.

The choice between

    - dip depth at expected harmonics ``k/S``, or
    - FRC-AUC (integral from 0 to Nyquist), or
    - some third alternative

depends on what the actual averaged curves look like on the test datasets.
See ``agents_artifacts/FRC_metric.md`` §3.7 for the design discussion. The
function below raises until the empirical question is settled; everything
upstream (per-image curves, per-method aggregation, headline plot) works
without it.
"""

from __future__ import annotations

import numpy as np


def frc_to_scalar(curve: np.ndarray, freqs: np.ndarray, S: int) -> float:
    """Reduce an FRC curve to a single scalar — DEFERRED.

    Parameters
    ----------
    curve : np.ndarray
        Per-bin FRC values.
    freqs : np.ndarray
        Matching frequency-bin centres in cycles/pixel.
    S : int
        Inner-tile size in pixels.

    Returns
    -------
    float
        Never returned — the call always raises.

    Raises
    ------
    NotImplementedError
        Always. The reduction is intentionally deferred until inspection
        of real FRC curves on the workshop datasets indicates which
        formulation is appropriate.
    """
    raise NotImplementedError(
        "FRC curve-to-scalar reduction is deferred until empirical evidence "
        "indicates dip-depth-at-harmonics vs. FRC-AUC vs. an alternative. "
        "See agents_artifacts/FRC_metric.md §3.7."
    )
