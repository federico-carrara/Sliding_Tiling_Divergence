"""Per-image and per-method aggregation of FRC curves.

Per image we record a 1-D FRC curve and the matching frequency-bin centres.
Per method we report the per-bin mean curve across the dataset and a 95%
confidence interval (``± 1.96 · SE``) on that mean. The Fisher-z transform
mentioned in the handout (§3.5) is left for later; default raw-CI is
typically indistinguishable in plots when FRC stays well away from ±1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FRCImageResult:
    """Per-image FRC curve and matching frequency-bin centres.

    Attributes
    ----------
    freqs : np.ndarray
        Frequency-bin centres in cycles/pixel, shape ``(n_bins,)``.
    frc : np.ndarray
        FRC values in ``[-1, 1]``, shape ``(n_bins,)``. ``NaN`` where a ring
        is empty (only possible for highly anisotropic images at the
        outermost rings).
    image_shape : tuple of int
        Original image shape ``(H, W)``.
    """

    freqs: np.ndarray
    frc: np.ndarray
    image_shape: tuple[int, int]


@dataclass
class FRCMethodReport:
    """Per-method roll-up of FRC curves across all images for that method.

    Attributes
    ----------
    images : list of FRCImageResult
        Per-image FRC curves for this method.
    freqs : np.ndarray
        Shared frequency-bin centres in cycles/pixel.
    mean_frc : np.ndarray
        Per-bin mean across images (``np.nanmean``).
    ci95_lo : np.ndarray
        Per-bin lower 95% CI bound (``mean - 1.96 * SE``).
    ci95_hi : np.ndarray
        Per-bin upper 95% CI bound (``mean + 1.96 * SE``).
    n_images : int
        Number of images aggregated.
    """

    images: list[FRCImageResult]
    freqs: np.ndarray
    mean_frc: np.ndarray
    ci95_lo: np.ndarray
    ci95_hi: np.ndarray
    n_images: int


@dataclass
class FRCMultiMethodReport:
    """Top-level result of a multi-method FRC run.

    Attributes
    ----------
    methods : dict of str to FRCMethodReport
        Per-method reports keyed by method name.
    config_summary : dict, optional
        Snapshot of the run configuration.
    """

    methods: dict[str, FRCMethodReport] = field(default_factory=dict)
    config_summary: Optional[dict] = None


def aggregate_method(images: list[FRCImageResult]) -> FRCMethodReport:
    """Aggregate per-image FRC curves into a :class:`FRCMethodReport`.

    Assumes all images share the same frequency grid (handout §3.5: all
    images in a dataset have the same size). Validated at function entry.

    Parameters
    ----------
    images : list of FRCImageResult
        Per-image FRC curves for one method.

    Returns
    -------
    FRCMethodReport
        Per-method roll-up. If ``images`` is empty, returns a report with
        empty arrays and ``n_images=0``.

    Raises
    ------
    ValueError
        If image curves do not share the same frequency-grid shape.
    """
    if not images:
        empty = np.array([], dtype=np.float64)
        return FRCMethodReport(
            images=[],
            freqs=empty,
            mean_frc=empty,
            ci95_lo=empty,
            ci95_hi=empty,
            n_images=0,
        )

    freqs = images[0].freqs
    for i, im in enumerate(images[1:], start=1):
        if im.freqs.shape != freqs.shape:
            raise ValueError(
                f"image {i} has freqs shape {im.freqs.shape}; expected "
                f"{freqs.shape}. All images must share the same size."
            )

    stack = np.stack([im.frc for im in images], axis=0)
    mean_frc = np.nanmean(stack, axis=0)
    counts = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.nanstd(stack, axis=0, ddof=1)
        se = np.where(counts > 1, std / np.sqrt(counts), np.nan)
    half_width = 1.96 * se

    return FRCMethodReport(
        images=images,
        freqs=freqs,
        mean_frc=mean_frc,
        ci95_lo=mean_frc - half_width,
        ci95_hi=mean_frc + half_width,
        n_images=len(images),
    )
