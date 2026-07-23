"""Smooth tapers applied before the FFT to suppress boundary leakage.

A finite-size image fed to ``np.fft.fft2`` is treated as one period of an
infinite periodic signal. Real images do not wrap cleanly, so the discrete
spectrum is contaminated by a cross of high-frequency energy along the
``u`` and ``v`` axes. That cross lands exactly where seam harmonics
``(k/S, 0)`` and ``(0, k/S)`` are expected, so windowing is mandatory for
this use case.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import hamming


def apply_hamming_window_2d(img: np.ndarray) -> np.ndarray:
    """Apply a 2-D separable Hamming window to a single image.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Windowed image, same shape and dtype-promoted to float.
    """
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image; got ndim={img.ndim}")
    h, w = img.shape
    window = np.outer(hamming(h), hamming(w))
    return img.astype(np.float64, copy=False) * window
