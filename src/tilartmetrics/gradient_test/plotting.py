"""Visualisation helpers for the gradient test.

Side-by-side comparison of finite-difference gradients produced by
:func:`tilartmetrics.gradient_test.gradient_analysis.compute_gradients`
across several methods (e.g. inner tiling vs. SWiTi). The figure is laid out
as a grid with one column per method and one row per gradient axis, so that
the same axis is compared horizontally and the contrast is matched per row.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from tilartmetrics.gradient_test.aggregation import (
    ChannelReport,
    ImageReport,
    MethodReport,
    TileResult,
)
from tilartmetrics.gradient_test.seams import compute_seam_positions
from tilartmetrics.gradient_test.tiles import enumerate_tiles


def _default_axis_names(n_axes: int) -> tuple[str, ...]:
    """Return default per-axis labels for a ``n_axes``-gradient stack.

    Two axes are assumed to be ``(y, x)`` and three ``(z, y, x)``, matching the
    output ordering of :func:`compute_gradients`. Any other count falls back to
    generic ``axis-i`` labels.
    """
    if n_axes == 2:
        return ("y-gradient", "x-gradient")
    if n_axes == 3:
        return ("z-gradient", "y-gradient", "x-gradient")
    return tuple(f"axis-{i}-gradient" for i in range(n_axes))


def _crop(grad: NDArray, y_ROI: Optional[slice], x_ROI: Optional[slice]) -> NDArray:
    """Crop the two trailing spatial axes of ``grad`` to the given ROIs."""
    if y_ROI is not None:
        grad = grad[..., y_ROI, :]
    if x_ROI is not None:
        grad = grad[..., :, x_ROI]
    return grad


def _as_slice(roi: Optional[Union[slice, tuple[int, int]]]) -> Optional[slice]:
    """Normalise a ``(start, stop)`` tuple (or ``slice``) into a ``slice``."""
    if roi is None or isinstance(roi, slice):
        return roi
    start, stop = roi
    return slice(start, stop)


def plot_gradient_comparison(
    gradients: Mapping[str, Sequence[NDArray]],
    axis_names: Optional[Sequence[str]] = None,
    y_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    x_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    z_idx: Optional[int] = None,
    cmap: str = "gray",
    contrast_lims: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    symmetric: bool = True,
    percentile: float = 99.0,
    suptitle: Optional[str] = None,
    facecolor: str = "black",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> "plt.Figure":
    """Compare per-axis gradients across methods in a single grid figure.

    The grid has one **column per method** and one **row per gradient axis**.
    For the 2-D inner-tiling vs. SWiTi case this reproduces the canonical
    ``2 x 2`` layout (rows ``= (y, x)``, columns ``= (inner tiling, SWiTi)``),
    but it generalises to any number of methods and gradient axes.

    Contrast is matched per row by default, so the same gradient axis is shown
    on a common scale across methods and the comparison is meaningful.

    Parameters
    ----------
    gradients : Mapping[str, Sequence[NDArray]]
        Mapping from method name to its per-axis gradient stack, i.e. the tuple
        returned by :func:`compute_gradients` (``(g_y, g_x)`` for 2-D images,
        ``(g_z, g_y, g_x)`` for 3-D). Every method must expose the same number
        of axes. Method names become column titles.
    axis_names : Optional[Sequence[str]]
        Row labels, one per gradient axis. Defaults to ``("y-gradient",
        "x-gradient")`` for 2 axes and ``("z-gradient", "y-gradient",
        "x-gradient")`` for 3 axes.
    y_ROI, x_ROI : Optional[Union[slice, tuple[int, int]]]
        Region of interest along the y / x axis, given either as a ``slice`` or
        a ``(start, stop)`` tuple. Applied to every gradient image. Default is
        the full extent.
    z_idx : Optional[int]
        For 3-D gradient stacks ``(D, H, W)``, the z-slice to display. Required
        when the gradient images are 3-D; ignored for 2-D images.
    cmap : str
        Matplotlib colormap name. Default is ``"gray"``.
    contrast_lims : Optional[Sequence[Optional[tuple[float, float]]]]
        Explicit ``(vmin, vmax)`` per row (axis). Use ``None`` for a given row
        to fall back to the automatic contrast. If omitted entirely, contrast
        is derived automatically (see ``symmetric`` / ``percentile``).
    symmetric : bool
        When ``True`` (default) the automatic per-row contrast is symmetric
        around zero (``vmax = -vmin``), which is the natural choice for signed
        gradients. When ``False`` the row min / max are used.
    percentile : float
        Percentile of the absolute (or raw) values used to derive the automatic
        contrast, making it robust to outliers. Default is ``99.0``.
    suptitle : Optional[str]
        Overall figure title. Default is None.
    facecolor : str
        Figure background colour. Default is ``"black"``; titles and labels are
        drawn in a contrasting colour automatically.
    save_path : Optional[Union[str, Path]]
        If given, the figure is saved to this path. Default is None.
    dpi : int
        Resolution for the saved figure. Default is ``150``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle, for downstream customisation or display.

    Raises
    ------
    ValueError
        If ``gradients`` is empty, methods disagree on the number of axes, or
        ``axis_names`` / ``contrast_lims`` lengths do not match the axis count.
    """
    if not gradients:
        raise ValueError("`gradients` must contain at least one method.")

    method_names = list(gradients.keys())
    n_methods = len(method_names)
    n_axes = len(next(iter(gradients.values())))
    for name, grads in gradients.items():
        if len(grads) != n_axes:
            raise ValueError(
                f"All methods must have the same number of gradient axes; "
                f"'{name}' has {len(grads)} but expected {n_axes}."
            )

    if axis_names is None:
        axis_names = _default_axis_names(n_axes)
    elif len(axis_names) != n_axes:
        raise ValueError(
            f"`axis_names` must have {n_axes} entries, got {len(axis_names)}."
        )
    if contrast_lims is not None and len(contrast_lims) != n_axes:
        raise ValueError(
            f"`contrast_lims` must have {n_axes} entries, got {len(contrast_lims)}."
        )

    y_slice = _as_slice(y_ROI)
    x_slice = _as_slice(x_ROI)

    text_color = "white" if facecolor == "black" else "black"

    # crop (and z-slice) every gradient image up front
    cropped: dict[str, list[NDArray]] = {}
    for name, grads in gradients.items():
        imgs = []
        for grad in grads:
            grad = np.asarray(grad)
            if grad.ndim == 3:
                if z_idx is None:
                    raise ValueError(
                        "`z_idx` must be provided for 3-D gradient images."
                    )
                grad = grad[z_idx]
            imgs.append(_crop(grad, y_slice, x_slice))
        cropped[name] = imgs

    # derive per-row (per-axis) contrast limits
    row_vlims: list[tuple[float, float]] = []
    for a in range(n_axes):
        if contrast_lims is not None and contrast_lims[a] is not None:
            row_vlims.append(contrast_lims[a])
            continue
        stacked = np.concatenate(
            [cropped[name][a].ravel() for name in method_names]
        )
        if symmetric:
            vmax = float(np.percentile(np.abs(stacked), percentile))
            row_vlims.append((-vmax, vmax))
        else:
            lo = float(np.percentile(stacked, 100.0 - percentile))
            hi = float(np.percentile(stacked, percentile))
            row_vlims.append((lo, hi))

    fig, axes = plt.subplots(
        n_axes,
        n_methods,
        figsize=(4.5 * n_methods, 4.5 * n_axes),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor(facecolor)
    if suptitle:
        fig.suptitle(suptitle, fontsize=20, color=text_color)

    for col, name in enumerate(method_names):
        for row in range(n_axes):
            ax = axes[row, col]
            vmin, vmax = row_vlims[row]
            ax.imshow(cropped[name][row], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=14, color=text_color)
            if col == 0:
                ax.set_ylabel(axis_names[row], fontsize=13, color=text_color)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    return fig


# --------------------------------------------------------------------------- #
# Report-driven visualisations (per-tile significance)                        #
# --------------------------------------------------------------------------- #


def _build_tile_pvalue_map(
    channel_report: ChannelReport,
    image_shape: Sequence[int],
    tile_size: Sequence[int],
    overlap: Sequence[int],
) -> NDArray:
    """Paint each tile's p-value onto a pixel-space array.

    A :class:`~tilartmetrics.gradient_test.aggregation.TileResult` stores a
    grid ``coord``, not a pixel location. This helper rebuilds the tile geometry
    with :func:`~tilartmetrics.gradient_test.tiles.enumerate_tiles` (which
    yields per-axis ``(lo, hi)`` pixel ranges) and fills every pixel of a tile's
    rectangle with that tile's Phipson–Smyth p-value. Skipped tiles
    (``n_seams < 2``) keep ``NaN`` because their ``p`` is already ``NaN``.

    Parameters
    ----------
    channel_report : ChannelReport
        Per-(image, channel) report whose ``tiles`` carry ``coord`` and ``p``.
    image_shape : Sequence[int]
        Spatial shape of the image, ``(H, W)`` in 2D or ``(D, H, W)`` in 3D —
        the same shape that was analysed (no batch / channel axes).
    tile_size, overlap : Sequence[int]
        TiledPatching tile size / overlap per spatial axis, matching the values
        passed to :func:`run_gradient_analysis`.

    Returns
    -------
    NDArray
        Float array of shape ``image_shape``; each pixel holds the p-value of
        the tile that covers it, or ``NaN`` where no tile applies / the tile was
        skipped.
    """
    tiles = enumerate_tiles(image_shape, tile_size, overlap)
    ranges_by_coord = {t.coord: t.ranges for t in tiles}
    pmap = np.full(tuple(image_shape), np.nan, dtype=np.float64)
    for tr in channel_report.tiles:
        ranges = ranges_by_coord.get(tr.coord)
        if ranges is None:
            continue
        slices = tuple(slice(lo, hi) for (lo, hi) in ranges)
        pmap[slices] = tr.p
    return pmap


def _significance_score(pmap: NDArray, alpha: float) -> np.ma.MaskedArray:
    """Return ``-log10(p)`` as a masked array, hiding non-significant pixels.

    Pixels with ``p >= alpha`` or non-finite ``p`` (skipped / uncovered tiles)
    are masked so the background image shows through.
    """
    pmap = np.asarray(pmap, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = -np.log10(pmap)
    mask = ~np.isfinite(pmap) | (pmap >= alpha)
    return np.ma.masked_array(score, mask=mask)


def _score_vlims(pmaps: Sequence[NDArray], alpha: float) -> tuple[float, float]:
    """Shared ``-log10(p)`` colour limits across one or more p-value maps.

    ``vmin`` is the significance threshold ``-log10(alpha)`` (the bottom of the
    colour bar is exactly ``p = alpha``); ``vmax`` is driven by the smallest
    significant p-value seen. Falls back to ``vmin + 1`` when nothing is
    significant, to keep a valid range.
    """
    vmin = -math.log10(alpha)
    vmax = vmin
    for pmap in pmaps:
        arr = np.asarray(pmap, dtype=np.float64)
        valid = arr[np.isfinite(arr) & (arr < alpha)]
        if valid.size:
            vmax = max(vmax, float(-np.log10(valid.min())))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _select_2d(
    image: NDArray, pmap: NDArray, z_idx: Optional[int]
) -> tuple[NDArray, NDArray]:
    """Reduce a 2D ``(H,W)`` / 3D ``(D,H,W)`` image+pmap pair to 2D for display."""
    if image.ndim == 2:
        return image, pmap
    if image.ndim == 3:
        if z_idx is None:
            raise ValueError("`z_idx` must be provided for 3-D images.")
        return image[z_idx], pmap[z_idx]
    raise ValueError(f"image must be 2-D or 3-D; got ndim={image.ndim}")


def _slice_start(sl: Optional[slice]) -> int:
    """Return the start offset of a ``slice`` (0 for ``None`` / open start)."""
    if sl is None or sl.start is None:
        return 0
    return int(sl.start)


def _draw_seams_on_ax(
    ax: "plt.Axes",
    disp_shape: tuple[int, int],
    full_hw: tuple[int, int],
    tile_size: Sequence[int],
    overlap: Sequence[int],
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Draw thin cyan lines at seam pixel positions on ``ax`` (ROI-aware)."""
    h_full, w_full = full_hw
    y0, x0 = origin
    disp_h, disp_w = disp_shape
    for x in compute_seam_positions(w_full, tile_size[-1], overlap[-1]):
        xx = int(x) - x0
        if 0 <= xx <= disp_w:
            ax.axvline(xx - 0.5, color="cyan", lw=0.6, alpha=0.5)
    for y in compute_seam_positions(h_full, tile_size[-2], overlap[-2]):
        yy = int(y) - y0
        if 0 <= yy <= disp_h:
            ax.axhline(yy - 0.5, color="cyan", lw=0.6, alpha=0.5)


def _strip_ax(ax: "plt.Axes") -> None:
    """Remove ticks and spines from an image axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _gradient_display(
    image2d: NDArray, axis: Optional[int]
) -> tuple[NDArray, bool]:
    """Compute a displayable gradient image aligned to the image grid.

    Uses :func:`numpy.gradient` (central differences, same shape as the input)
    so the result lines up pixel-for-pixel with the image and shares its ROI
    crop / seam overlay.

    Parameters
    ----------
    image2d : NDArray
        2-D image ``(H, W)``.
    axis : Optional[int]
        ``0`` for the y-gradient, ``1`` for the x-gradient, or ``None`` for the
        gradient magnitude ``sqrt(g_y**2 + g_x**2)``.

    Returns
    -------
    grad : NDArray
        The gradient image, same shape as ``image2d``.
    symmetric : bool
        ``True`` for a signed single-axis gradient (use symmetric contrast),
        ``False`` for the non-negative magnitude.
    """
    g_y, g_x = np.gradient(np.asarray(image2d, dtype=np.float64))
    if axis is None:
        return np.hypot(g_y, g_x), False
    if axis == 0:
        return g_y, True
    if axis == 1:
        return g_x, True
    raise ValueError(f"`gradient_axis` must be 0, 1 or None; got {axis}.")


def _draw_overlay_on_ax(
    ax: "plt.Axes",
    image2d: NDArray,
    pmap2d: NDArray,
    *,
    alpha: float,
    cmap: str,
    overlay_cmap: str,
    overlay_alpha: float,
    vlims: tuple[float, float],
    draw_seams: bool = False,
    full_hw: Optional[tuple[int, int]] = None,
    tile_size: Optional[Sequence[int]] = None,
    overlap: Optional[Sequence[int]] = None,
    origin: tuple[int, int] = (0, 0),
) -> "plt.cm.ScalarMappable":
    """Render the grayscale image plus the masked ``-log10(p)`` overlay on ``ax``.

    Shared by :func:`plot_significance_overlay` and
    :func:`plot_significance_overlay_grid`. Returns the overlay mappable so the
    caller can attach a colour bar.
    """
    image2d = np.asarray(image2d)
    ax.imshow(image2d, cmap=cmap)
    score = _significance_score(pmap2d, alpha)
    vmin, vmax = vlims
    mappable = ax.imshow(
        score,
        cmap=overlay_cmap,
        alpha=overlay_alpha,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    if draw_seams and full_hw is not None:
        _draw_seams_on_ax(
            ax, image2d.shape, full_hw, tile_size, overlap, origin
        )
    _strip_ax(ax)
    return mappable


def plot_significance_overlay(
    image_report: ImageReport,
    image: NDArray,
    *,
    channel: int,
    tile_size: Sequence[int],
    overlap: Sequence[int],
    alpha: float = 0.05,
    z_idx: Optional[int] = None,
    y_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    x_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    cmap: str = "gray",
    overlay_cmap: str = "inferno",
    overlay_alpha: float = 0.5,
    draw_seams: bool = False,
    show_gradient: bool = True,
    gradient_axis: Optional[int] = None,
    gradient_cmap: str = "gray",
    gradient_percentile: float = 99.0,
    title: Optional[str] = None,
    facecolor: str = "black",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> "plt.Figure":
    """Overlay significant tiles (``-log10(p)``) on a single input image.

    Tiles with ``p < alpha`` are painted with a ``-log10(p)`` heatmap (brighter
    = more significant) over the grayscale image; non-significant and skipped
    tiles are transparent. The bottom of the colour bar corresponds to
    ``p = alpha``. When ``show_gradient`` is set, the image's gradient is drawn
    in a second panel to the right, on the same ROI / seam grid, so the
    significance pattern can be read against the underlying gradient structure.

    Parameters
    ----------
    image_report : ImageReport
        Per-image report, e.g. ``method_report.images[image_id]``.
    image : NDArray
        The matching single-channel image slice — ``(H, W)`` in 2D or
        ``(D, H, W)`` in 3D — i.e. ``predictions[n, channel]``.
    channel : int
        Which channel of ``image_report`` to overlay (indexes
        ``image_report.channels``).
    tile_size, overlap : Sequence[int]
        TiledPatching tile size / overlap per spatial axis (same as the analysis
        run).
    alpha : float
        Significance threshold for masking. Default ``0.05``.
    z_idx : Optional[int]
        z-slice to display for 3-D images. Required when ``image`` is 3-D.
    y_ROI, x_ROI : Optional[Union[slice, tuple[int, int]]]
        Region of interest along y / x as a ``slice`` or ``(start, stop)``.
    cmap : str
        Colormap for the background image. Default ``"gray"``.
    overlay_cmap : str
        Colormap for the ``-log10(p)`` overlay. Default ``"inferno"``.
    overlay_alpha : float
        Opacity of the overlay. Default ``0.6``.
    draw_seams : bool
        Draw thin lines at seam pixel positions for context. Default ``False``.
    show_gradient : bool
        Add a second panel to the right showing the image gradient. Default
        ``True``.
    gradient_axis : Optional[int]
        Which gradient to display in that panel: ``0`` for the y-gradient,
        ``1`` for the x-gradient, or ``None`` (default) for the gradient
        magnitude ``sqrt(g_y**2 + g_x**2)``.
    gradient_cmap : str
        Colormap for the gradient panel. Default ``"gray"``.
    gradient_percentile : float
        Percentile used to set the gradient panel's contrast robustly to
        outliers (symmetric for a signed axis, ``[0, p]`` for the magnitude).
        Default ``99.0``.
    title : Optional[str]
        Axis title (placed over the overlay panel). Default None.
    facecolor : str
        Figure background colour. Default ``"black"``.
    save_path : Optional[Union[str, Path]]
        If given, save the figure here.
    dpi : int
        Resolution for the saved figure. Default ``150``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle.
    """
    image = np.asarray(image)
    channel_report = image_report.channels[channel]
    pmap = _build_tile_pvalue_map(
        channel_report, image.shape, tile_size, overlap
    )
    img2d, pmap2d = _select_2d(image, pmap, z_idx)
    full_hw = (img2d.shape[0], img2d.shape[1])

    # gradient computed on the full slice (so edge effects sit at true image
    # edges) then cropped to the same ROI as the overlay.
    grad2d = (
        _gradient_display(img2d, gradient_axis) if show_gradient else None
    )

    y_slice = _as_slice(y_ROI)
    x_slice = _as_slice(x_ROI)
    img2d = _crop(img2d, y_slice, x_slice)
    pmap2d = _crop(pmap2d, y_slice, x_slice)
    origin = (_slice_start(y_slice), _slice_start(x_slice))

    vlims = _score_vlims([pmap2d], alpha)
    text_color = "white" if facecolor == "black" else "black"

    ncols = 2 if show_gradient else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(6.5 * ncols, 6.5), squeeze=False, constrained_layout=True
    )
    fig.patch.set_facecolor(facecolor)
    # gradient panel (if any) goes on the left, the overlay on the right.
    ax = axes[0, -1]
    mappable = _draw_overlay_on_ax(
        ax,
        img2d,
        pmap2d,
        alpha=alpha,
        cmap=cmap,
        overlay_cmap=overlay_cmap,
        overlay_alpha=overlay_alpha,
        vlims=vlims,
        draw_seams=draw_seams,
        full_hw=full_hw,
        tile_size=tile_size,
        overlap=overlap,
        origin=origin,
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"-log10(p)  (bottom = α = {alpha})", color=text_color)
    cbar.ax.tick_params(colors=text_color)
    if title:
        ax.set_title(title, fontsize=14, color=text_color)

    # stats box in the top-left corner of the overlay panel
    stats_text = (
        f"mean Z = {channel_report.mean_Z:.2f}\n"
        f"median Z = {channel_report.median_Z:.2f}\n"
        f"frac rejected = {channel_report.frac_rejected:.3f}"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="none"),
    )

    if show_gradient:
        grad_img, symmetric = grad2d
        grad_img = _crop(grad_img, y_slice, x_slice)
        vmax = float(np.percentile(np.abs(grad_img), gradient_percentile))
        gvmin, gvmax = (-vmax, vmax) if symmetric else (0.0, vmax)
        gax = axes[0, 0]
        gmappable = gax.imshow(grad_img, cmap=gradient_cmap, vmin=gvmin, vmax=gvmax)
        if draw_seams:
            _draw_seams_on_ax(
                gax, grad_img.shape, full_hw, tile_size, overlap, origin
            )
        _strip_ax(gax)
        gcbar = fig.colorbar(gmappable, ax=gax, fraction=0.046, pad=0.04)
        gcbar.ax.tick_params(colors=text_color)
        grad_label = (
            "gradient magnitude"
            if gradient_axis is None
            else ("y-gradient", "x-gradient")[gradient_axis]
        )
        gax.set_title(grad_label, fontsize=14, color=text_color)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    return fig


def _collect_tiles(
    report: Union[ChannelReport, ImageReport, MethodReport],
    channel: Optional[int],
) -> list[TileResult]:
    """Gather the per-tile results from any report level, optionally filtered.

    Parameters
    ----------
    report : ChannelReport, ImageReport or MethodReport
        Source of per-tile results. Channels are descended into for image- and
        method-level reports.
    channel : Optional[int]
        If given, keep only the tiles of that channel index; if ``None``, pool
        every channel. Ignored for a :class:`ChannelReport` (already one
        channel).

    Returns
    -------
    list of TileResult
        Pooled tiles.

    Raises
    ------
    TypeError
        If ``report`` is not one of the three report types.
    """
    def channels_of(image: ImageReport) -> list[ChannelReport]:
        if channel is None:
            return list(image.channels.values())
        return [image.channels[channel]]

    if isinstance(report, ChannelReport):
        return list(report.tiles)
    if isinstance(report, ImageReport):
        return [t for ch in channels_of(report) for t in ch.tiles]
    if isinstance(report, MethodReport):
        return [
            t
            for image in report.images.values()
            for ch in channels_of(image)
            for t in ch.tiles
        ]
    raise TypeError(
        "`report` must be a ChannelReport, ImageReport or MethodReport; "
        f"got {type(report).__name__}."
    )


def plot_pvalue_distribution(
    report: Union[ChannelReport, ImageReport, MethodReport],
    *,
    channel: Optional[int] = None,
    alpha: float = 0.05,
    bins: int = 20,
    title: Optional[str] = None,
    facecolor: str = "white",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> "plt.Figure":
    """Histogram of per-tile p-values with the rejection threshold marked.

    Parameters
    ----------
    report : ChannelReport, ImageReport or MethodReport
        Source of per-tile p-values. For an :class:`ImageReport` or
        :class:`MethodReport` the p-values are pooled across images/channels.
    channel : Optional[int]
        Restrict pooling to this channel index. If ``None`` (default), every
        channel is pooled. Ignored for a :class:`ChannelReport`.
    alpha : float
        Rejection threshold, drawn as a vertical line and used for the reported
        rejected fraction. Default ``0.05``.
    bins : int
        Number of histogram bins over ``[0, 1]``. Default ``20``.
    title : Optional[str]
        Axis title. Defaults to a summary of the rejected fraction and count.
    facecolor : str
        Figure background colour. Default ``"white"``.
    save_path : Optional[Union[str, Path]]
        If given, save the figure here.
    dpi : int
        Resolution for the saved figure. Default ``150``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle.

    Raises
    ------
    TypeError
        If ``report`` is not a :class:`ChannelReport`, :class:`ImageReport` or
        :class:`MethodReport`.
    """
    tiles = _collect_tiles(report, channel)

    pvals = np.array(
        [t.p for t in tiles if not np.isnan(t.p)], dtype=np.float64
    )
    frac = float(np.mean(pvals < alpha)) if pvals.size else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    fig.patch.set_facecolor(facecolor)
    if pvals.size:
        ax.hist(
            pvals,
            bins=bins,
            range=(0.0, 1.0),
            color="steelblue",
            edgecolor="black",
        )
    ax.axvline(alpha, color="red", linestyle="--", linewidth=1.5, label=f"α = {alpha}")
    ax.set_xlabel("per-tile p-value")
    ax.set_ylabel("tile count")
    ax.set_title(
        title
        or f"p-value distribution — frac rejected = {frac:.3f}  (n = {pvals.size})"
    )
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    return fig


def _per_tile_texture(
    texture_image: NDArray,
    tiles: Sequence,
    use_gradient_magnitude: bool,
) -> dict[tuple[int, ...], float]:
    """Compute a per-tile "texture" (local variability) scalar.

    Texture is the standard deviation, over each tile's pixel rectangle, of
    either the raw intensity or the gradient magnitude. It quantifies the
    natural local variability against which a seam signal must compete: low
    texture == flat tile.

    Parameters
    ----------
    texture_image : NDArray
        2-D ``(H, W)`` reference image defining the texture. Use a *shared*
        reference (e.g. the ground truth) so the texture axis is common across
        methods being compared.
    tiles : Sequence of Tile
        Tiles from :func:`enumerate_tiles` for ``texture_image.shape``.
    use_gradient_magnitude : bool
        If ``True`` use ``std`` of ``sqrt(g_y**2 + g_x**2)`` over the tile; if
        ``False`` use ``std`` of the raw intensity.

    Returns
    -------
    dict
        Mapping from tile ``coord`` to its texture scalar.
    """
    img = np.asarray(texture_image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"texture_image must be 2-D; got ndim={img.ndim}")
    if use_gradient_magnitude:
        g_y, g_x = np.gradient(img)
        field = np.hypot(g_y, g_x)
    else:
        field = img
    texture: dict[tuple[int, ...], float] = {}
    for t in tiles:
        (y0, y1), (x0, x1) = t.ranges
        patch = field[y0:y1, x0:x1]
        texture[t.coord] = float(patch.std()) if patch.size else float("nan")
    return texture


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    More honest than the normal approximation for the small per-bin counts and
    near-0 / near-1 rejection rates this plot encounters.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def plot_flatness_stratified_rejection(
    reports: Mapping[str, ImageReport],
    texture_image: NDArray,
    *,
    channel: int,
    tile_size: Sequence[int],
    overlap: Sequence[int],
    alpha: float = 0.05,
    n_bins: int = 8,
    bin_mode: str = "quantile",
    use_gradient_magnitude: bool = True,
    log_x: bool = False,
    title: Optional[str] = None,
    facecolor: str = "white",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> "plt.Figure":
    """Plot per-tile rejection rate as a function of local tile texture.

    This is the diagnostic that adjudicates *why* a method rejects in flat
    regions. Tiles are stratified by a **shared** texture scalar (computed once
    from ``texture_image``, e.g. the ground truth, so every method sits on the
    same x-axis), then for each method the fraction of tiles rejecting at
    ``alpha`` is plotted per texture bin with a Wilson confidence band.

    Interpretation
    --------------
    - A well-calibrated null (e.g. GT, which has no seams) should sit near
      ``alpha`` across *all* texture bins — flat **or** textured.
    - If a method's rejection rises specifically in the **low-texture** bins
      while the null stays flat, the flat-region positives are a *real*,
      texture-unmasked signal of that method, not a calibration artifact of the
      narrow gradient distribution.
    - If the null *also* climbs at low texture, the narrow-distribution regime
      genuinely inflates the false-positive rate.

    Parameters
    ----------
    reports : Mapping[str, ImageReport]
        Per-image reports keyed by method name (all over the same geometry as
        ``texture_image``). Typically ``{"GT": ..., "SWITi": ..., "Inner
        Tiling": ...}``.
    channel : int
        Which channel of each report to stratify (indexes ``ImageReport.channels``).
    texture_image : NDArray
        2-D ``(H, W)`` image defining the shared per-tile texture axis.
    tile_size, overlap : Sequence[int]
        TiledPatching tile size / overlap per spatial axis (same as the run).
    alpha : float
        Rejection threshold. Default ``0.05``.
    n_bins : int
        Number of texture bins. Default ``8``.
    bin_mode : {"quantile", "linear"}
        ``"quantile"`` for equal-count bins (robust to skewed texture
        distributions), ``"linear"`` for equal-width bins. Default
        ``"quantile"``.
    use_gradient_magnitude : bool
        Define texture from the gradient magnitude (``True``, default) or the
        raw intensity (``False``).
    log_x : bool
        Use a logarithmic texture axis. Default ``False`` (linear).
    title : Optional[str]
        Axis title. ``None`` (default) summarises the configuration; pass an
        empty string ``""`` to draw no title at all.
    facecolor : str
        Figure background colour. Default ``"white"``.
    save_path : Optional[Union[str, Path]]
        If given, save the figure here.
    dpi : int
        Resolution for the saved figure. Default ``150``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle.

    Raises
    ------
    ValueError
        If ``reports`` is empty, ``bin_mode`` is unknown, or no valid tiles are
        found across the reports.
    """
    if not reports:
        raise ValueError("`reports` must contain at least one method.")
    if bin_mode not in ("quantile", "linear"):
        raise ValueError(f"bin_mode must be 'quantile' or 'linear'; got {bin_mode!r}")

    texture_image = np.asarray(texture_image)
    tiles = enumerate_tiles(texture_image.shape, tile_size, overlap)
    texture_by_coord = _per_tile_texture(
        texture_image, tiles, use_gradient_magnitude
    )

    # The set of valid (testable) coords and their texture is geometry-driven,
    # hence identical across methods; gather it from the union to be safe.
    valid_coords: set[tuple[int, ...]] = set()
    for ir in reports.values():
        for tr in ir.channels[channel].tiles:
            if not np.isnan(tr.p):
                valid_coords.add(tr.coord)
    coords = sorted(
        c for c in valid_coords if not np.isnan(texture_by_coord.get(c, np.nan))
    )
    if not coords:
        raise ValueError("No valid tiles with finite texture across reports.")

    textures = np.array([texture_by_coord[c] for c in coords], dtype=np.float64)

    if bin_mode == "quantile":
        edges = np.quantile(textures, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(textures.min(), textures.max(), n_bins + 1)
    # assign each coord to a bin (rightmost edge inclusive)
    bin_idx = np.clip(np.digitize(textures, edges[1:-1], right=False), 0, len(edges) - 2)
    n_actual_bins = len(edges) - 1

    # shared x position per bin = mean texture of its tiles
    bin_x = np.array(
        [
            textures[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
            for b in range(n_actual_bins)
        ]
    )
    text_color = "black" if facecolor == "white" else "white"
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    fig.patch.set_facecolor(facecolor)

    # faint bin-edge guides (equal-count quantile bins, so an explicit count
    # axis is redundant — the guides just show where the bins fall)
    for e in edges:
        ax.axvline(e, color="0.8", linewidth=0.8, zorder=0)

    cmap = plt.get_cmap("tab10")
    for mi, (name, ir) in enumerate(reports.items()):
        reject_by_coord = {
            tr.coord: (tr.p < alpha)
            for tr in ir.channels[channel].tiles
            if not np.isnan(tr.p)
        }
        rej = np.array(
            [reject_by_coord.get(c, np.nan) for c in coords], dtype=np.float64
        )
        frac = np.full(n_actual_bins, np.nan)
        lo = np.full(n_actual_bins, np.nan)
        hi = np.full(n_actual_bins, np.nan)
        for b in range(n_actual_bins):
            sel = (bin_idx == b) & np.isfinite(rej)
            n = int(np.sum(sel))
            if n == 0:
                continue
            k = int(np.sum(rej[sel]))
            frac[b] = k / n
            lo[b], hi[b] = _wilson_interval(k, n)

        good = np.isfinite(frac) & np.isfinite(bin_x)
        color = cmap(mi % 10)
        ax.plot(
            bin_x[good],
            frac[good],
            marker="o",
            color=color,
            label=name,
            zorder=3,
        )
        ax.fill_between(
            bin_x[good], lo[good], hi[good], color=color, alpha=0.2, zorder=2
        )

    ax.axhline(
        alpha,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=rf"$\alpha = {alpha:g}$",
        zorder=1,
    )
    if log_x:
        ax.set_xscale("log")
    else:
        # crop the empty upper tail: quantile binning makes the last bin span a
        # long sparse range, but the last plotted point sits well below its
        # upper edge. Stop the axis just past the last point.
        finite_bx = bin_x[np.isfinite(bin_x)]
        if finite_bx.size:
            span = finite_bx.max() - finite_bx.min()
            ax.set_xlim(right=finite_bx.max() + 0.1 * span)
    tex_label = (
        "gradient-magnitude std"
        if use_gradient_magnitude
        else "intensity std"
    )
    ax.set_xlabel(
        f"tile texture — per-tile {tex_label} of the reference image "
        f"(flat → textured)",
        fontsize=14,
    )
    ax.set_ylabel("fraction of tiles rejected", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(-0.02, 1.02)
    if title is None:
        title = (
            f"Rejection rate vs. tile texture  "
            f"({bin_mode} bins, n={len(coords)} tiles)"
        )
    if title:  # pass title="" to suppress
        ax.set_title(title, color=text_color)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    return fig


def plot_significance_overlay_grid(
    method_report: MethodReport,
    predictions: NDArray,
    *,
    tile_size: Sequence[int],
    overlap: Sequence[int],
    channel: int,
    alpha: float = 0.05,
    z_idx: Optional[int] = None,
    y_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    x_ROI: Optional[Union[slice, tuple[int, int]]] = None,
    ncols: int = 4,
    cmap: str = "gray",
    overlay_cmap: str = "inferno",
    overlay_alpha: float = 0.6,
    draw_seams: bool = False,
    suptitle: Optional[str] = None,
    facecolor: str = "black",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> "plt.Figure":
    """Grid of per-image significance overlays for a whole :class:`MethodReport`.

    One subplot per image in ``method_report.images``; each background image is
    taken as ``predictions[n, channel]`` (exactly as :func:`run_gradient_analysis`
    slices it), so the overlay aligns pixel-for-pixel. Contrast and the shared
    colour bar are common across the grid for comparability.

    Parameters
    ----------
    method_report : MethodReport
        The report returned by :func:`run_gradient_analysis`.
    predictions : NDArray
        The same ``(N, C, ...)`` prediction array passed to the analysis.
    tile_size, overlap : Sequence[int]
        TiledPatching tile size / overlap per spatial axis.
    channel : int
        Channel index that was analysed.
    alpha : float
        Significance threshold for masking. Default ``0.05``.
    z_idx : Optional[int]
        z-slice to display for 3-D images. Required when images are 3-D.
    y_ROI, x_ROI : Optional[Union[slice, tuple[int, int]]]
        Region of interest along y / x.
    ncols : int
        Number of columns in the grid. Default ``4``.
    cmap, overlay_cmap, overlay_alpha, draw_seams :
        As in :func:`plot_significance_overlay`.
    suptitle : Optional[str]
        Overall figure title.
    facecolor : str
        Figure background colour. Default ``"black"``.
    save_path : Optional[Union[str, Path]]
        If given, save the figure here.
    dpi : int
        Resolution for the saved figure. Default ``150``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure handle.

    Raises
    ------
    ValueError
        If ``method_report`` has no images.
    """
    predictions = np.asarray(predictions)
    n_images = len(method_report.images)
    if n_images == 0:
        raise ValueError("`method_report` has no images to plot.")

    y_slice = _as_slice(y_ROI)
    x_slice = _as_slice(x_ROI)
    origin = (_slice_start(y_slice), _slice_start(x_slice))

    # build aligned (image, pmap) pairs up front so contrast can be shared.
    # ``images`` is a dict keyed by image_id, filled in prediction-row order by
    # run_gradient_analysis, so enumerate() index n aligns with predictions[n].
    items: list[tuple[NDArray, NDArray, tuple[int, int], str, float]] = []
    for n, ir in enumerate(method_report.images.values()):
        channel_report = ir.channels[channel]
        image = np.asarray(predictions[n, channel])
        pmap = _build_tile_pvalue_map(
            channel_report, image.shape, tile_size, overlap
        )
        img2d, pmap2d = _select_2d(image, pmap, z_idx)
        full_hw = (img2d.shape[0], img2d.shape[1])
        img2d = _crop(img2d, y_slice, x_slice)
        pmap2d = _crop(pmap2d, y_slice, x_slice)
        items.append(
            (img2d, pmap2d, full_hw, ir.image_id, channel_report.frac_rejected)
        )

    vlims = _score_vlims([p for _, p, *_ in items], alpha)
    text_color = "white" if facecolor == "black" else "black"

    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor(facecolor)
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, color=text_color)

    mappable = None
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= n_images:
            ax.axis("off")
            continue
        img2d, pmap2d, full_hw, image_id, frac_rejected = items[idx]
        mappable = _draw_overlay_on_ax(
            ax,
            img2d,
            pmap2d,
            alpha=alpha,
            cmap=cmap,
            overlay_cmap=overlay_cmap,
            overlay_alpha=overlay_alpha,
            vlims=vlims,
            draw_seams=draw_seams,
            full_hw=full_hw,
            tile_size=tile_size,
            overlap=overlap,
            origin=origin,
        )
        ax.set_title(
            f"image {image_id} — rej {frac_rejected:.2f}",
            fontsize=11,
            color=text_color,
        )

    if mappable is not None:
        cbar = fig.colorbar(
            mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02
        )
        cbar.set_label(f"-log10(p)  (bottom = α = {alpha})", color=text_color)
        cbar.ax.tick_params(colors=text_color)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving plot to {save_path}")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    return fig
