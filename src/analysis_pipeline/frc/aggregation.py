"""Per-image and per-method aggregation of FRC curves.

Per ``(image, channel)`` slice we record a 1-D FRC curve and the matching
frequency-bin centres. Channels are grouped under images; per method we report,
**per channel**, the per-bin mean curve across the dataset and a 95% confidence
interval (``± 1.96 · SE``) on that mean. The Fisher-z transform mentioned in the
handout (§3.5) is left for later; default raw-CI is typically indistinguishable
in plots when FRC stays well away from ±1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FRCChannelResult:
    """Per-channel FRC curve for one image slice — the computational leaf.

    Attributes
    ----------
    channel : int
        Channel index this curve was computed on.
    freqs : np.ndarray
        Frequency-bin centres in cycles/pixel, shape ``(n_bins,)``.
    frc : np.ndarray
        FRC values in ``[-1, 1]``, shape ``(n_bins,)``. ``NaN`` where a ring
        is empty (only possible for highly anisotropic images at the
        outermost rings).
    image_shape : tuple of int
        Original image shape ``(H, W)``.
    """

    channel: int
    freqs: np.ndarray
    frc: np.ndarray
    image_shape: tuple[int, int]


@dataclass
class FRCImageReport:
    """Per-image container grouping its channels, plus a pooled mean curve.

    Attributes
    ----------
    image_id : str
        Identifier of the image (e.g. an index or a filename stem).
    channels : dict of int to FRCChannelResult
        Per-channel FRC curves keyed by channel index.
    freqs : np.ndarray
        Shared frequency-bin centres in cycles/pixel (all channels of one image
        share the same grid).
    mean_frc : np.ndarray
        Per-bin mean FRC pooled across the image's channels (``np.nanmean``).
    """

    image_id: str
    channels: dict[int, FRCChannelResult] = field(default_factory=dict)
    freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mean_frc: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclass
class FRCMethodReport:
    """Per-method roll-up of FRC curves across all images for that method.

    Attributes
    ----------
    method_name : str
        Name of the method these results belong to.
    dataset : str, optional
        Name of the dataset the images were drawn from.
    images : dict of str to FRCImageReport
        Per-image reports keyed by ``image_id``.
    freqs : np.ndarray
        Shared frequency-bin centres in cycles/pixel.
    mean_frc : dict of int to np.ndarray
        Per-channel per-bin mean across images (``np.nanmean``), keyed by
        channel index.
    ci95_lo : dict of int to np.ndarray
        Per-channel per-bin lower 95% CI bound (``mean - 1.96 * SE``).
    ci95_hi : dict of int to np.ndarray
        Per-channel per-bin upper 95% CI bound (``mean + 1.96 * SE``).
    n_images : int
        Number of images aggregated.
    """

    method_name: str = "method"
    dataset: Optional[str] = None
    images: dict[str, FRCImageReport] = field(default_factory=dict)
    freqs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    mean_frc: dict[int, np.ndarray] = field(default_factory=dict)
    ci95_lo: dict[int, np.ndarray] = field(default_factory=dict)
    ci95_hi: dict[int, np.ndarray] = field(default_factory=dict)
    n_images: int = 0

    def to_records(self) -> list[dict]:
        """Flatten to one summary record per ``(image_id, channel)``.

        Each row carries the method/dataset identifiers plus scalar summaries of
        the channel's FRC curve: the mean FRC excluding the DC bin and the value
        at the Nyquist bin. Suitable for direct construction of a
        :class:`pandas.DataFrame`.

        Returns
        -------
        list of dict
            One row per channel of every image.
        """
        records: list[dict] = []
        for image_id, image in self.images.items():
            for channel, ch in image.channels.items():
                frc = ch.frc
                body_mean = (
                    float(np.nanmean(frc[1:])) if frc.size > 1 else float("nan")
                )
                nyquist = float(frc[-1]) if frc.size else float("nan")
                records.append(
                    {
                        "dataset": self.dataset,
                        "method_name": self.method_name,
                        "image_id": image_id,
                        "channel": channel,
                        "frc_mean_excl_dc": body_mean,
                        "frc_nyquist": nyquist,
                        "n_bins": int(frc.size),
                    }
                )
        return records


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


def _mean_and_ci95(
    curves: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin ``nanmean`` and 95% CI bounds over a set of FRC curves.

    Parameters
    ----------
    curves : list of np.ndarray
        FRC curves sharing the same frequency-grid shape.

    Returns
    -------
    tuple of np.ndarray
        ``(mean_frc, ci95_lo, ci95_hi)``. The CI half-width is ``1.96 * SE``
        with ``SE`` NaN where fewer than two finite values contribute.
    """
    stack = np.stack(curves, axis=0)
    mean_frc = np.nanmean(stack, axis=0)
    counts = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.nanstd(stack, axis=0, ddof=1)
        se = np.where(counts > 1, std / np.sqrt(counts), np.nan)
    half_width = 1.96 * se
    return mean_frc, mean_frc - half_width, mean_frc + half_width


def aggregate_image(
    image_id: str,
    channels: dict[int, FRCChannelResult],
) -> FRCImageReport:
    """Aggregate per-channel FRC curves into an :class:`FRCImageReport`.

    The image-level ``mean_frc`` pools the channels' curves per bin
    (``np.nanmean``); all channels of one image share the same frequency grid.

    Parameters
    ----------
    image_id : str
        Identifier of the image.
    channels : dict of int to FRCChannelResult
        Per-channel FRC curves keyed by channel index (must be non-empty).

    Returns
    -------
    FRCImageReport
        Per-image container with a pooled mean curve.

    Raises
    ------
    ValueError
        If ``channels`` is empty or its curves do not share a frequency grid.
    """
    if not channels:
        raise ValueError(f"image {image_id!r}: channels must be non-empty")

    curves = list(channels.values())
    freqs = curves[0].freqs
    for ch in curves[1:]:
        if ch.freqs.shape != freqs.shape:
            raise ValueError(
                f"image {image_id!r}: channel {ch.channel} has freqs shape "
                f"{ch.freqs.shape}; expected {freqs.shape}. All channels of an "
                "image must share the same size."
            )

    mean_frc = np.nanmean(np.stack([ch.frc for ch in curves], axis=0), axis=0)
    return FRCImageReport(
        image_id=image_id,
        channels=channels,
        freqs=freqs,
        mean_frc=mean_frc,
    )


def aggregate_method(
    images: dict[str, FRCImageReport],
    method_name: str = "method",
    dataset: Optional[str] = None,
) -> FRCMethodReport:
    """Aggregate per-image reports into a per-method :class:`FRCMethodReport`.

    For each channel index present across the images, the per-image channel
    curves are stacked and reduced to a per-bin mean + 95% CI. Assumes all
    curves share the same frequency grid (handout §3.5: all images in a dataset
    have the same size). Validated at function entry.

    Parameters
    ----------
    images : dict of str to FRCImageReport
        Per-image reports keyed by ``image_id``.
    method_name : str, default="method"
        Name of the method.
    dataset : str, optional
        Name of the dataset the images were drawn from.

    Returns
    -------
    FRCMethodReport
        Per-method roll-up. If ``images`` is empty, returns a report with empty
        arrays/dicts and ``n_images=0``.

    Raises
    ------
    ValueError
        If curves do not share the same frequency-grid shape.
    """
    if not images:
        return FRCMethodReport(method_name=method_name, dataset=dataset)

    # Reference grid from the first channel of the first image.
    first_image = next(iter(images.values()))
    freqs = first_image.freqs

    channel_to_curves: dict[int, list[np.ndarray]] = {}
    for image in images.values():
        for channel, ch in image.channels.items():
            if ch.freqs.shape != freqs.shape:
                raise ValueError(
                    f"image {image.image_id!r} channel {channel} has freqs "
                    f"shape {ch.freqs.shape}; expected {freqs.shape}. All "
                    "images must share the same size."
                )
            channel_to_curves.setdefault(channel, []).append(ch.frc)

    mean_frc: dict[int, np.ndarray] = {}
    ci95_lo: dict[int, np.ndarray] = {}
    ci95_hi: dict[int, np.ndarray] = {}
    for channel, curves in channel_to_curves.items():
        mean_frc[channel], ci95_lo[channel], ci95_hi[channel] = _mean_and_ci95(
            curves
        )

    return FRCMethodReport(
        method_name=method_name,
        dataset=dataset,
        images=images,
        freqs=freqs,
        mean_frc=mean_frc,
        ci95_lo=ci95_lo,
        ci95_hi=ci95_hi,
        n_images=len(images),
    )
