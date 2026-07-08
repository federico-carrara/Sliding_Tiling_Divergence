"""Per-image and per-method aggregation of per-tile results.

Per-tile we record ``T_obs`` and ``p``; per-image we summarize with the
median ``T`` and the fraction of tiles rejecting at ``alpha`` (NaN tiles —
those with insufficient seams — excluded from both); per-method we average
the per-image scalars across the test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TileResult:
    """Outcome of the permutation test for one kept-region tile.

    Attributes
    ----------
    coord : tuple of int
        Multi-index in the kept-region grid.
    n_seams : int
        Number of seams owned by the tile.
    T_obs : float
        Observed statistic value (``nan`` for skipped tiles).
    p : float
        Phipson–Smyth p-value (``nan`` for skipped tiles).
    n_seam_samples : int
        Total number of seam samples used in the test.
    n_control_samples : int
        Total number of control samples used in the test.
    """

    coord: tuple[int, ...]
    n_seams: int
    T_obs: float
    p: float
    n_seam_samples: int
    n_control_samples: int


@dataclass
class ChannelReport:
    """Per-channel roll-up of tile-level scores for one image slice.

    This is the computational leaf: the per-tile test operates on a single
    ``(image, channel)`` slice and its outcomes are summarized here.

    Attributes
    ----------
    channel : int
        Channel index this report was computed on.
    tiles : list of TileResult
        Per-tile outcomes (including skipped tiles).
    median_T : float
        Median of valid ``T_obs`` across tiles (``nan`` if none are valid).
    frac_rejected : float
        Fraction of valid tiles with ``p < alpha`` (``nan`` if none are valid).
    """

    channel: int
    tiles: list[TileResult]
    median_T: float
    frac_rejected: float


@dataclass
class ImageReport:
    """Per-image container grouping its channels, plus pooled scalars.

    Attributes
    ----------
    image_id : str
        Identifier of the image (e.g. an index or a filename stem).
    channels : dict of int to ChannelReport
        Per-channel reports keyed by channel index.
    median_T : float
        Median of valid ``T_obs`` pooled across all channels' tiles
        (``nan`` if none are valid).
    frac_rejected : float
        Fraction of valid tiles with ``p < alpha`` pooled across all channels'
        tiles (``nan`` if none are valid).
    """

    image_id: str
    channels: dict[int, ChannelReport] = field(default_factory=dict)
    median_T: float = float("nan")
    frac_rejected: float = float("nan")


@dataclass
class MethodReport:
    """Per-method roll-up across all images for that method.

    Attributes
    ----------
    method_name : str
        Name of the method these results belong to.
    dataset : str, optional
        Name of the dataset the images were drawn from.
    images : dict of str to ImageReport
        Per-image reports keyed by ``image_id``.
    mean_median_T : dict of int to float
        Per-channel mean of valid per-image ``median_T`` values, keyed by
        channel index.
    mean_frac_rejected : dict of int to float
        Per-channel mean of valid per-image ``frac_rejected`` values, keyed by
        channel index.
    """

    method_name: str
    dataset: Optional[str] = None
    images: dict[str, ImageReport] = field(default_factory=dict)
    mean_median_T: dict[int, float] = field(default_factory=dict)
    mean_frac_rejected: dict[int, float] = field(default_factory=dict)

    def to_records(self) -> list[dict]:
        """Flatten to one record per ``(image_id, channel)`` for tabular use.

        Returns
        -------
        list of dict
            One row per channel of every image, each carrying the method and
            dataset identifiers plus the channel-level scalars. Suitable for
            direct construction of a :class:`pandas.DataFrame`.
        """
        records: list[dict] = []
        for image_id, image in self.images.items():
            for channel, ch in image.channels.items():
                n_valid = sum(1 for t in ch.tiles if not np.isnan(t.p))
                records.append(
                    {
                        "dataset": self.dataset,
                        "method_name": self.method_name,
                        "image_id": image_id,
                        "channel": channel,
                        "median_T": ch.median_T,
                        "frac_rejected": ch.frac_rejected,
                        "n_tiles": len(ch.tiles),
                        "n_valid_tiles": n_valid,
                    }
                )
        return records


@dataclass
class MultiMethodReport:
    """Top-level result of a multi-method per-tile run.

    Attributes
    ----------
    methods : dict of str to MethodReport
        Per-method reports keyed by method name.
    config_summary : dict, optional
        Snapshot of the run configuration.
    """

    methods: dict[str, MethodReport] = field(default_factory=dict)
    config_summary: Optional[dict] = None


def _median_and_frac(
    tiles: list[TileResult], alpha: float
) -> tuple[float, float]:
    """Compute ``(median_T, frac_rejected)`` over valid tiles.

    Tiles with ``NaN`` ``T_obs`` or ``p`` (insufficient seams) are excluded.

    Parameters
    ----------
    tiles : list of TileResult
        Per-tile outcomes.
    alpha : float
        Rejection threshold.

    Returns
    -------
    tuple of float
        ``median_T`` (``nan`` if no valid ``T_obs``) and ``frac_rejected``
        (``nan`` if no valid ``p``).
    """
    valid_T = np.array(
        [t.T_obs for t in tiles if not np.isnan(t.T_obs)], dtype=np.float64
    )
    valid_p = np.array(
        [t.p for t in tiles if not np.isnan(t.p)], dtype=np.float64
    )
    median_T = float(np.median(valid_T)) if valid_T.size else float("nan")
    frac_rejected = (
        float(np.mean(valid_p < alpha)) if valid_p.size else float("nan")
    )
    return median_T, frac_rejected


def aggregate_channel(
    tiles: list[TileResult],
    alpha: float,
    channel: int,
) -> ChannelReport:
    """Aggregate per-tile outcomes for one channel into a :class:`ChannelReport`.

    Parameters
    ----------
    tiles : list of TileResult
        Per-tile outcomes for the ``(image, channel)`` slice.
    alpha : float
        Rejection threshold.
    channel : int
        Channel index these tiles were computed on.

    Returns
    -------
    ChannelReport
        Per-channel roll-up.
    """
    median_T, frac_rejected = _median_and_frac(tiles, alpha)
    return ChannelReport(
        channel=channel,
        tiles=tiles,
        median_T=median_T,
        frac_rejected=frac_rejected,
    )


def aggregate_image(
    image_id: str,
    channels: dict[int, ChannelReport],
    alpha: float,
) -> ImageReport:
    """Aggregate per-channel reports into an :class:`ImageReport`.

    The image-level scalars pool the valid tiles of *all* channels into one
    population (tile-level pooling), treating the image as a whole.

    Parameters
    ----------
    image_id : str
        Identifier of the image.
    channels : dict of int to ChannelReport
        Per-channel reports keyed by channel index.
    alpha : float
        Rejection threshold.

    Returns
    -------
    ImageReport
        Per-image container with pooled scalars.
    """
    pooled_tiles = [t for ch in channels.values() for t in ch.tiles]
    median_T, frac_rejected = _median_and_frac(pooled_tiles, alpha)
    return ImageReport(
        image_id=image_id,
        channels=channels,
        median_T=median_T,
        frac_rejected=frac_rejected,
    )


def aggregate_method(
    images: dict[str, ImageReport],
    method_name: str,
    dataset: Optional[str] = None,
) -> MethodReport:
    """Aggregate per-image reports into a per-method :class:`MethodReport`.

    For each channel index present across the images, the per-image channel
    scalars are averaged (``NaN`` values excluded) into a per-channel mean.

    Parameters
    ----------
    images : dict of str to ImageReport
        Per-image reports keyed by ``image_id``.
    method_name : str, default="method"
        Name of the method.
    dataset : str, optional
        Name of the dataset the images were drawn from.

    Returns
    -------
    MethodReport
        Per-method roll-up; an empty report (no images) if ``images`` is empty.
    """
    if not images:
        return MethodReport(method_name=method_name, dataset=dataset)

    channel_medians: dict[int, list[float]] = {}
    channel_fracs: dict[int, list[float]] = {}
    for image in images.values():
        for channel, ch in image.channels.items():
            if not np.isnan(ch.median_T):
                channel_medians.setdefault(channel, []).append(ch.median_T)
            if not np.isnan(ch.frac_rejected):
                channel_fracs.setdefault(channel, []).append(ch.frac_rejected)

    all_channels = {
        c for im in images.values() for c in im.channels
    }
    mean_median_T = {
        c: (
            float(np.mean(channel_medians[c]))
            if channel_medians.get(c)
            else float("nan")
        )
        for c in all_channels
    }
    mean_frac_rejected = {
        c: (
            float(np.mean(channel_fracs[c]))
            if channel_fracs.get(c)
            else float("nan")
        )
        for c in all_channels
    }
    return MethodReport(
        method_name=method_name,
        dataset=dataset,
        images=images,
        mean_median_T=mean_median_T,
        mean_frac_rejected=mean_frac_rejected,
    )
