"""Finite-difference gradient helpers.

Single-image gradient computation for 2-D ``(H, W)`` or 3-D ``(D, H, W)``
arrays. The output has one entry per spatial axis, with the differenced
axis one element shorter than the input.

The old class-based ``GradientUtils*`` hierarchy (seam-aligned sampling,
shared bin edges, Wiener helpers) belonged to the previous global-KL
pipeline and is gone. Per-tile sampling now lives in
:mod:`analysis_pipeline.core.sampling`.
"""

from __future__ import annotations

import numpy as np


def compute_gradients_2d(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(g_y, g_x)`` for a 2-D image ``(H, W)``.

    Shapes: ``g_y`` is ``(H-1, W)``; ``g_x`` is ``(H, W-1)``.
    """
    if img.ndim != 2:
        raise ValueError(f"expected 2-D (H, W); got ndim={img.ndim}")
    g_y = img[1:, :] - img[:-1, :]
    g_x = img[:, 1:] - img[:, :-1]
    return g_y, g_x


def compute_gradients_3d(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(g_z, g_y, g_x)`` for a 3-D image ``(D, H, W)``."""
    if img.ndim != 3:
        raise ValueError(f"expected 3-D (D, H, W); got ndim={img.ndim}")
    g_z = img[1:, :, :] - img[:-1, :, :]
    g_y = img[:, 1:, :] - img[:, :-1, :]
    g_x = img[:, :, 1:] - img[:, :, :-1]
    return g_z, g_y, g_x


def compute_gradients(img: np.ndarray) -> tuple[np.ndarray, ...]:
    """Dispatch to the 2-D or 3-D variant based on ``img.ndim``."""
    if img.ndim == 2:
        return compute_gradients_2d(img)
    if img.ndim == 3:
        return compute_gradients_3d(img)
    raise ValueError(
        f"image must be 2-D (H,W) or 3-D (D,H,W); got ndim={img.ndim}"
    )
