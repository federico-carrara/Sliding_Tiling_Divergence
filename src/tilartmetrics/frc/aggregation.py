"""Per-image and per-method aggregation of FRC curves.

Per ``(image, channel)`` slice we record a 1-D FRC curve and the matching
frequency-bin centres. Channels are grouped under images; per method we report,
**per channel**, the per-bin mean curve across the dataset and a 95% confidence
interval (``± 1.96 · SE``) on that mean. The Fisher-z transform mentioned in the
handout (§3.5) is left for later; default raw-CI is typically indistinguishable
in plots when FRC stays well away from ±1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional, Union

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
)
from typing_extensions import Self

from tilartmetrics.frc.reduction import frc_resolution, frc_resolution_period


def _to_ndarray(value: Any) -> np.ndarray:
    """Coerce a value (e.g. a JSON list) to a float64 array on load."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(_to_ndarray),
    PlainSerializer(lambda a: a.tolist(), return_type=list),
]
"""A 1-D float64 numpy array that serializes to a JSON list and back.

``NaN`` entries survive the round-trip because the base model sets
``ser_json_inf_nan="constants"`` (they are written as the bare ``NaN`` token).
"""


class _FRCReportModel(BaseModel):
    """Base for FRC report models with JSON persistence.

    ``arbitrary_types_allowed`` lets the models carry :class:`numpy.ndarray`
    fields (serialized via :data:`NdArray`); ``ser_json_inf_nan="constants"``
    keeps ``NaN`` values in curves and scalars round-tripping losslessly (the
    default ``"null"`` would emit ``null`` and then fail to reload into a
    ``float``/array field); ``protected_namespaces=()`` silences warnings about
    the ``method_name`` field.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        ser_json_inf_nan="constants",
        protected_namespaces=(),
    )

    def save(self, path: Union[str, Path]) -> Path:
        """Serialize to indented JSON at ``path`` (parents created).

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file (a ``.json`` suffix is conventional).

        Returns
        -------
        pathlib.Path
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Load and validate an instance from a JSON file written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Source JSON file.

        Returns
        -------
        Self
            The validated model instance.
        """
        return cls.model_validate_json(Path(path).read_text())


class FRCChannelResult(_FRCReportModel):
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
    freqs: NdArray
    frc: NdArray
    image_shape: tuple[int, int]


class FRCImageReport(_FRCReportModel):
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
    channels: dict[int, FRCChannelResult] = Field(default_factory=dict)
    freqs: NdArray = Field(default_factory=lambda: np.array([], dtype=np.float64))
    mean_frc: NdArray = Field(default_factory=lambda: np.array([], dtype=np.float64))


class FRCMethodReport(_FRCReportModel):
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
    images: dict[str, FRCImageReport] = Field(default_factory=dict)
    freqs: NdArray = Field(default_factory=lambda: np.array([], dtype=np.float64))
    mean_frc: dict[int, NdArray] = Field(default_factory=dict)
    ci95_lo: dict[int, NdArray] = Field(default_factory=dict)
    ci95_hi: dict[int, NdArray] = Field(default_factory=dict)
    n_images: int = 0

    def to_records(self) -> list[dict]:
        """Flatten to one summary record per ``(image_id, channel)``.

        Each row carries the method/dataset identifiers plus scalar summaries of
        the channel's FRC curve: the mean FRC excluding the DC bin, and the
        resolution readout — the frequency where the curve first falls below the
        conventional ``1/7`` threshold, reported both in cycles/pixel and as the
        matching period in pixels (see
        :func:`tilartmetrics.frc.reduction.frc_resolution`). Suitable for
        direct construction of a :class:`pandas.DataFrame`.

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
                records.append(
                    {
                        "dataset": self.dataset,
                        "method_name": self.method_name,
                        "image_id": image_id,
                        "channel": channel,
                        "frc_mean_excl_dc": body_mean,
                        "frc_res_cyc_per_px": frc_resolution(ch.freqs, frc),
                        "frc_res_period_px": frc_resolution_period(ch.freqs, frc),
                        "n_bins": int(frc.size),
                    }
                )
        return records


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
