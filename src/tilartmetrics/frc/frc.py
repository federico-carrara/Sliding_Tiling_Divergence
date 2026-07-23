"""Per-image Fourier Ring Correlation against a matching ground truth.

For a ``(prediction, ground_truth)`` pair of the same field of view, compute

    FRC(r) = Σ_{k ∈ ring(r)} F_P(k) * conj(F_G(k))
           / sqrt(Σ_{k ∈ ring(r)} |F_P(k)|² * Σ_{k ∈ ring(r)} |F_G(k)|²)

where the rings are integer-radius shells in centered 2-D Fourier space.
The output is a 1-D curve over radial frequency, x-axis in cycles/pixel
(``0`` to ``0.5`` Nyquist).

Implementation credit: the radial-binning approach mirrors MIPLIB
(``miplib/data/iterators/fourier_ring_iterators.py``, Koho et al. 2019,
Nat. Commun. 10:3103). The standard two-image variant in numpy is short
enough that we re-implement to avoid taking MIPLIB's heavy install
footprint (C extensions, jpype1/JDK, SimpleITK, ...).
"""

from __future__ import annotations

import numpy as np

from tilartmetrics.frc.aggregation import FRCChannelResult
from tilartmetrics.frc.windowing import apply_hamming_window_2d


def per_image_frc(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    apply_window: bool = True,
    channel: int = 0,
) -> FRCChannelResult:
    """Compute the FRC curve of ``prediction`` against ``ground_truth``.

    Parameters
    ----------
    prediction : np.ndarray
        Prediction image, shape ``(H, W)``.
    ground_truth : np.ndarray
        Ground-truth image, same shape as ``prediction``.
    apply_window : bool, default=True
        If True (default), multiply both images by a 2-D Hamming window
        before the FFT. Disable only for sanity tests where periodic-image
        assumptions hold (e.g. white noise against itself).
    channel : int, default=0
        Channel index this slice was taken from; stamped onto the result.

    Returns
    -------
    FRCChannelResult
        Per-bin FRC values, matching frequency-bin centres in cycles/pixel,
        and the input image shape.

    Raises
    ------
    ValueError
        If the two images are not 2-D or do not share the same shape.
    """
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError(
            f"prediction and ground_truth must be 2-D; got "
            f"ndim={prediction.ndim} and {ground_truth.ndim}"
        )
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape} vs "
            f"ground_truth {ground_truth.shape}"
        )

    p = apply_hamming_window_2d(prediction) if apply_window else prediction.astype(
        np.float64, copy=False
    )
    g = apply_hamming_window_2d(ground_truth) if apply_window else ground_truth.astype(
        np.float64, copy=False
    )

    f_p = np.fft.fftshift(np.fft.fft2(p))
    f_g = np.fft.fftshift(np.fft.fft2(g))

    h, w = p.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.rint(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(np.int64)

    n = min(h, w)
    r_max = n // 2
    valid = r <= r_max
    r_flat = r[valid]

    cross_re = (f_p[valid] * np.conj(f_g[valid])).real
    p_mag2 = (f_p[valid].real ** 2 + f_p[valid].imag ** 2)
    g_mag2 = (f_g[valid].real ** 2 + f_g[valid].imag ** 2)
    # For real-valued input images F is Hermitian-symmetric, so each ring
    # contains every (k, -k) pair and the imaginary part of the cross sum
    # cancels exactly. Take .real and skip the imag bincount.

    minlength = r_max + 1
    numerator = np.bincount(r_flat, weights=cross_re, minlength=minlength)
    p_energy = np.bincount(r_flat, weights=p_mag2, minlength=minlength)
    g_energy = np.bincount(r_flat, weights=g_mag2, minlength=minlength)

    denom = np.sqrt(p_energy * g_energy)
    with np.errstate(divide="ignore", invalid="ignore"):
        frc = np.where(denom > 0, numerator / denom, np.nan)

    freqs = np.arange(minlength, dtype=np.float64) / n

    return FRCChannelResult(
        channel=channel,
        freqs=freqs,
        frc=frc,
        image_shape=(h, w),
    )
