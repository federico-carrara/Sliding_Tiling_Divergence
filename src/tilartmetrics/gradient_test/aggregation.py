"""Per-image and per-method aggregation of per-tile results.

Per-tile we record ``T_obs`` and ``p``; per-image we summarize with the
median ``T`` and the fraction of tiles rejecting at ``alpha`` (NaN tiles —
those with insufficient seams — excluded from both); per-method we average
the per-image scalars across the test set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self


class _ReportModel(BaseModel):
    """Base for gradient-test report models with JSON persistence.

    ``ser_json_inf_nan="constants"`` makes ``NaN`` scalars serialize to the
    bare ``NaN`` token (readable by both pydantic and the stdlib ``json``
    module) so skipped-tile sentinels round-trip losslessly; the default
    ``"null"`` would emit ``null`` and then fail to reload into a ``float``
    field. ``protected_namespaces=()`` silences warnings about the
    ``method_name`` / ``model_*`` field-name overlap.
    """

    model_config = ConfigDict(
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


class TileResult(_ReportModel):
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
    null_mean : float
        Mean of the tile's permutation null (``nan`` for skipped tiles).
    null_std : float
        Standard deviation of the tile's permutation null (``nan`` for skipped
        tiles).
    Z_obs : float
        Calibrated Z-score ``(T_obs - null_mean) / null_std`` expressing the
        observed statistic in units of the tile's own null spread, making it
        comparable across tiles and images (``nan`` for skipped tiles).
    n_seam_samples : int
        Total number of seam samples used in the test (post axis-count balancing,
        when enabled).
    n_control_samples : int
        Total number of control samples used in the test (post axis-count
        balancing, when enabled).
    """

    coord: tuple[int, ...]
    n_seams: int
    T_obs: float
    p: float
    null_mean: float = float("nan")
    null_std: float = float("nan")
    Z_obs: float = float("nan")
    n_seam_samples: int
    n_control_samples: int


class ChannelReport(_ReportModel):
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
    mean_Z : float
        Mean calibrated Z-score over valid tiles (``nan`` if none are valid).
    median_Z : float
        Median calibrated Z-score over valid tiles (``nan`` if none are valid).
    p90_Z : float
        90th-percentile calibrated Z-score over valid tiles — a worst-case
        summary, since artifacts are often driven by a few bad seams (``nan`` if
        none are valid).
    """

    channel: int
    tiles: list[TileResult]
    median_T: float
    frac_rejected: float
    mean_Z: float = float("nan")
    median_Z: float = float("nan")
    p90_Z: float = float("nan")


class ImageReport(_ReportModel):
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
    mean_Z : float
        Mean calibrated Z-score over valid tiles pooled across all channels
        (``nan`` if none are valid).
    median_Z : float
        Median calibrated Z-score over valid pooled tiles (``nan`` if none are
        valid).
    p90_Z : float
        90th-percentile calibrated Z-score over valid pooled tiles (``nan`` if
        none are valid).
    """

    image_id: str
    channels: dict[int, ChannelReport] = Field(default_factory=dict)
    median_T: float = float("nan")
    frac_rejected: float = float("nan")
    mean_Z: float = float("nan")
    median_Z: float = float("nan")
    p90_Z: float = float("nan")


class MethodReport(_ReportModel):
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
    mean_mean_Z : dict of int to float
        Per-channel mean of valid per-image ``mean_Z`` values, keyed by channel
        index.
    mean_median_Z : dict of int to float
        Per-channel mean of valid per-image ``median_Z`` values, keyed by channel
        index.
    mean_p90_Z : dict of int to float
        Per-channel mean of valid per-image ``p90_Z`` values, keyed by channel
        index.
    """

    method_name: str
    dataset: Optional[str] = None
    images: dict[str, ImageReport] = Field(default_factory=dict)
    mean_median_T: dict[int, float] = Field(default_factory=dict)
    mean_frac_rejected: dict[int, float] = Field(default_factory=dict)
    mean_mean_Z: dict[int, float] = Field(default_factory=dict)
    mean_median_Z: dict[int, float] = Field(default_factory=dict)
    mean_p90_Z: dict[int, float] = Field(default_factory=dict)

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
                        "mean_Z": ch.mean_Z,
                        "median_Z": ch.median_Z,
                        "p90_Z": ch.p90_Z,
                        "n_tiles": len(ch.tiles),
                        "n_valid_tiles": n_valid,
                    }
                )
        return records


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


def _z_summaries(tiles: list[TileResult]) -> tuple[float, float, float]:
    """Compute ``(mean_Z, median_Z, p90_Z)`` over valid tiles.

    Tiles with ``NaN`` ``Z_obs`` (skipped tiles) are excluded.

    Parameters
    ----------
    tiles : list of TileResult
        Per-tile outcomes.

    Returns
    -------
    tuple of float
        ``mean_Z``, ``median_Z`` and ``p90_Z`` (all ``nan`` if no tile has a
        valid ``Z_obs``).
    """
    valid_Z = np.array(
        [t.Z_obs for t in tiles if not np.isnan(t.Z_obs)], dtype=np.float64
    )
    if not valid_Z.size:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(valid_Z)),
        float(np.median(valid_Z)),
        float(np.percentile(valid_Z, 90)),
    )


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
    mean_Z, median_Z, p90_Z = _z_summaries(tiles)
    return ChannelReport(
        channel=channel,
        tiles=tiles,
        median_T=median_T,
        frac_rejected=frac_rejected,
        mean_Z=mean_Z,
        median_Z=median_Z,
        p90_Z=p90_Z,
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
    mean_Z, median_Z, p90_Z = _z_summaries(pooled_tiles)
    return ImageReport(
        image_id=image_id,
        channels=channels,
        median_T=median_T,
        frac_rejected=frac_rejected,
        mean_Z=mean_Z,
        median_Z=median_Z,
        p90_Z=p90_Z,
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
    channel_mean_Z: dict[int, list[float]] = {}
    channel_median_Z: dict[int, list[float]] = {}
    channel_p90_Z: dict[int, list[float]] = {}
    for image in images.values():
        for channel, ch in image.channels.items():
            if not np.isnan(ch.median_T):
                channel_medians.setdefault(channel, []).append(ch.median_T)
            if not np.isnan(ch.frac_rejected):
                channel_fracs.setdefault(channel, []).append(ch.frac_rejected)
            if not np.isnan(ch.mean_Z):
                channel_mean_Z.setdefault(channel, []).append(ch.mean_Z)
            if not np.isnan(ch.median_Z):
                channel_median_Z.setdefault(channel, []).append(ch.median_Z)
            if not np.isnan(ch.p90_Z):
                channel_p90_Z.setdefault(channel, []).append(ch.p90_Z)

    all_channels = {
        c for im in images.values() for c in im.channels
    }

    def _channel_means(per_channel: dict[int, list[float]]) -> dict[int, float]:
        return {
            c: (
                float(np.mean(per_channel[c]))
                if per_channel.get(c)
                else float("nan")
            )
            for c in all_channels
        }

    return MethodReport(
        method_name=method_name,
        dataset=dataset,
        images=images,
        mean_median_T=_channel_means(channel_medians),
        mean_frac_rejected=_channel_means(channel_fracs),
        mean_mean_Z=_channel_means(channel_mean_Z),
        mean_median_Z=_channel_means(channel_median_Z),
        mean_p90_Z=_channel_means(channel_p90_Z),
    )
