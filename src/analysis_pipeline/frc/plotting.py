"""Headline FRC figure: mean curves with 95% CI bands per method.

For each dataset we report one figure with the per-method mean FRC curve,
a shaded 95% CI band, and (optionally) dashed verticals at the expected
seam harmonics ``k/S``. Non-overlapping CI bands at the dip locations are
themselves visual evidence of significance.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from analysis_pipeline.frc.aggregation import FRCMethodReport
from analysis_pipeline.frc.reduction import FRC_THRESHOLD_1_7

_GENERIC_FONT_FAMILIES = frozenset(
    {"serif", "sans-serif", "cursive", "fantasy", "monospace"}
)
"""Matplotlib's generic family aliases — always valid, never in ``ttflist``."""


def _font_is_available(font_family: str) -> bool:
    """True if ``font_family`` is a generic alias or a registered font name."""
    if font_family in _GENERIC_FONT_FAMILIES:
        return True
    return any(f.name == font_family for f in font_manager.fontManager.ttflist)


@contextmanager
def _set_font(font_family: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily set the matplotlib font family for the enclosed block.

    On exit the previous ``rcParams`` are restored, so this is safe to use
    inside any plotting function without leaking global state.

    If ``font_family`` is not installed, a :class:`RuntimeWarning` is raised once
    and the block runs with the matplotlib default — rather than letting
    matplotlib emit a ``findfont`` warning per text artist and silently
    substitute DejaVu.

    Parameters
    ----------
    font_family : str, optional
        Font family name to use (e.g. ``"Latin Modern Roman"``, ``"serif"``,
        ``"DejaVu Sans"``). Note the installed name for Latin Modern / LMRoman10
        is ``"Latin Modern Roman"``. If ``None`` (default), the current
        matplotlib default is kept and this is a no-op.

    Examples
    --------
    >>> with _set_font("Latin Modern Roman"):
    ...     fig, ax = plt.subplots()
    ...     ax.set_title("LaTeX-friendly title")
    """
    if font_family is None:
        yield
        return

    if not _font_is_available(font_family):
        warnings.warn(
            f"font family {font_family!r} is not installed; falling back to the "
            "matplotlib default. List installed names with "
            "`{f.name for f in matplotlib.font_manager.fontManager.ttflist}`.",
            RuntimeWarning,
            stacklevel=3,
        )
        yield
        return

    old_params = mpl.rcParams.copy()
    try:
        mpl.rcParams["font.family"] = font_family
        # Use Computer Modern for math text (LaTeX-style symbols).
        mpl.rcParams["mathtext.fontset"] = "cm"
        # Embed TrueType (Type 42) fonts in PDF/PS so text stays editable in
        # vector-graphics editors like Adobe Illustrator.
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        yield
    finally:
        mpl.rcParams.update(old_params)


def shared_ylim(
    reports: list[FRCMethodReport],
    channels: Optional[Sequence[int]] = None,
    *,
    pad: float = 0.05,
) -> Optional[tuple[float, float]]:
    """Common y-limits spanning every method and channel.

    Feed the result to :func:`plot_frc_curves`'s ``ylim`` so that a dataset's
    per-channel figures share one vertical scale and can be compared directly
    (otherwise each panel autoscales to its own curve).

    The extent covers the mean curves *and* their CI bands; ``NaN`` bins are
    ignored.

    Parameters
    ----------
    reports : list of FRCMethodReport
        Per-method reports to span.
    channels : sequence of int, optional
        Channels to include. ``None`` (default) spans every channel present in
        any report.
    pad : float, default=0.05
        Fractional margin added on each side of the observed extent.

    Returns
    -------
    tuple of float, or None
        ``(lo, hi)`` limits, or ``None`` when no finite values were found.
    """
    lo_vals: list[float] = []
    hi_vals: list[float] = []
    for m in reports:
        wanted = m.mean_frc.keys() if channels is None else channels
        for c in wanted:
            if c not in m.mean_frc:
                continue
            for arr in (m.mean_frc[c], m.ci95_lo.get(c), m.ci95_hi.get(c)):
                if arr is None:
                    continue
                finite = np.asarray(arr, dtype=np.float64)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    lo_vals.append(float(finite.min()))
                    hi_vals.append(float(finite.max()))

    if not lo_vals:
        return None
    lo, hi = min(lo_vals), max(hi_vals)
    span = hi - lo
    margin = pad * span if span > 0 else 0.05
    return (lo - margin, hi + margin)


def plot_frc_curves(
    reports: list[FRCMethodReport],
    steps: Optional[dict[str, Optional[int]]] = None,
    save_path: Optional[Path] = None,
    *,
    channel: int = 0,
    threshold: Optional[float] = FRC_THRESHOLD_1_7,
    ylim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    font_family: Optional[str] = "Latin Modern Roman",
) -> "plt.Figure":
    """Plot mean FRC curves with 95% CI bands for every method.

    Parameters
    ----------
    reports : list of FRCMethodReport
        Per-method aggregated FRC reports, one per curve. Plotted in the given
        order; each report's ``method_name`` labels its curve and keys into
        ``steps``.
    steps : dict of str to (int or None), optional
        Per-method seam interval in image pixels, keyed by ``method_name``: the
        spacing at which the method lays down seams (``tile_size - overlap`` for
        inner tiling; the sliding stride for SWiTi, whose seams are dimmer but
        still periodic). When provided for a method, dashed verticals are drawn
        in that method's colour at the expected seam harmonics ``k / step`` (in
        cycles/pixel) for ``k = 1, ..., step // 2`` — i.e. every harmonic up to
        Nyquist. Methods mapped to ``None`` (seam-free) skip the verticals.
    save_path : pathlib.Path, optional
        If provided, save the figure to this path (PNG suggested) at 300 dpi.
    channel : int, default=0
        Channel index whose per-channel mean curve and CI band to plot. Methods
        lacking this channel are skipped.
    threshold : float, optional, default=``1/7``
        Draw a horizontal dashed line at this FRC value — the resolution
        criterion each curve's crossing is read off (see
        :data:`~analysis_pipeline.frc.reduction.FRC_THRESHOLD_1_7`). ``None``
        omits the line.
    ylim : tuple of float, optional
        Explicit ``(lo, hi)`` y-limits. Pass the same value for every channel of
        a dataset — e.g. from :func:`shared_ylim` — so the panels share a
        vertical scale. ``None`` (default) autoscales each figure on its own.
    title : str, optional
        Axes title. ``None`` (default) derives ``"{dataset} - Ch. {channel}"``
        from the reports' common ``dataset``, falling back to just the channel
        when the reports carry no dataset name.
    font_family : str, optional
        Matplotlib font family applied for the duration of the plot via
        :func:`_set_font` (e.g. ``"serif"``, ``"LMRoman10"``). ``None`` keeps
        the current default. The change is scoped to this call and reverted on
        return.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle, for downstream customisation or display.
    """
    with _set_font(font_family):
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

        palette = plt.get_cmap("tab10").colors

        for i, m in enumerate(reports):
            name = m.method_name
            if m.n_images == 0 or channel not in m.mean_frc:
                continue
            colour = palette[i % len(palette)]
            ax.plot(
                m.freqs,
                m.mean_frc[channel],
                color=colour,
                linewidth=1.8,
                label=f"{name} (N={m.n_images})",
            )
            ax.fill_between(
                m.freqs,
                m.ci95_lo[channel],
                m.ci95_hi[channel],
                color=colour,
                alpha=0.20,
                linewidth=0,
            )

            if steps is None:
                continue
            step = steps.get(name)
            if step is None:
                continue
            # Seams every `step` px => harmonics at k/step cycles/pixel; only
            # those at or below Nyquist (0.5) are representable, i.e.
            # k <= step/2 (spec §4).
            for k in range(1, step // 2 + 1):
                ax.axvline(
                    k / step,
                    color=colour,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                )

        if threshold is not None:
            # Resolution criterion: each curve's crossing of this line is the
            # frequency reported by reduction.frc_resolution.
            ax.axhline(
                threshold,
                color="dimgray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label=f"threshold = 1/7 $\\approx$ {threshold:.3f}",
            )

        ax.set_xlim(0.0, 0.5)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("FRC")
        if title is None:
            names = {m.dataset for m in reports if m.dataset}
            dataset = names.pop() if len(names) == 1 else None
            title = (
                f"{dataset} - Ch. {channel}" if dataset else f"Ch. {channel}"
            )
        ax.set_title(title)
        ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.2)

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

    return fig
