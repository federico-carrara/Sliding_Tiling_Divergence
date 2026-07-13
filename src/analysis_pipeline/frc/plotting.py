"""Headline FRC figure: mean curves with 95% CI bands per method.

For each dataset we report one figure with the per-method mean FRC curve,
a shaded 95% CI band, and (optionally) dashed verticals at the expected
seam harmonics ``k/S``. Non-overlapping CI bands at the dip locations are
themselves visual evidence of significance.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

from analysis_pipeline.frc.aggregation import FRCMethodReport


@contextmanager
def _set_font(font_family: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily set the matplotlib font family for the enclosed block.

    On exit the previous ``rcParams`` are restored, so this is safe to use
    inside any plotting function without leaking global state.

    Parameters
    ----------
    font_family : str, optional
        Font family name to use (e.g. ``"LMRoman10"``, ``"serif"``,
        ``"DejaVu Sans"``). If ``None`` (default), the current matplotlib
        default is kept and this is a no-op.

    Examples
    --------
    >>> with _set_font("serif"):
    ...     fig, ax = plt.subplots()
    ...     ax.set_title("LaTeX-friendly title")
    """
    if font_family is None:
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


def plot_frc_curves(
    reports: list[FRCMethodReport],
    tile_inner_sizes: Optional[dict[str, Optional[int]]] = None,
    save_path: Optional[Path] = None,
    *,
    channel: int = 0,
    font_family: Optional[str] = "LMRoman10",
) -> "plt.Figure":
    """Plot mean FRC curves with 95% CI bands for every method.

    Parameters
    ----------
    reports : list of FRCMethodReport
        Per-method aggregated FRC reports, one per curve. Plotted in the given
        order; each report's ``method_name`` labels its curve and keys into
        ``tile_inner_sizes``.
    tile_inner_sizes : dict of str to (int or None), optional
        Per-method inner-tile size in image pixels, keyed by ``method_name``.
        When provided for a method, dashed verticals are drawn at expected
        harmonic frequencies ``k / S`` (in cycles/pixel) for
        ``k = 1, ..., N // (2*S)`` where ``N`` is the image side length used in
        the FRC computation. Methods with ``None`` (e.g. SWiTi) skip the
        verticals.
    save_path : pathlib.Path, optional
        If provided, save the figure to this path (PNG suggested) at 300 dpi.
    channel : int, default=0
        Channel index whose per-channel mean curve and CI band to plot. Methods
        lacking this channel are skipped.
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

            if tile_inner_sizes is None:
                continue
            s = tile_inner_sizes.get(name)
            if s is None or not m.images:
                continue
            first_image = next(iter(m.images.values()))
            h, w = first_image.channels[channel].image_shape
            n_pix = min(h, w)
            k_max = n_pix // (2 * s)
            for k in range(1, k_max + 1):
                ax.axvline(
                    k / s,
                    color=colour,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.5,
                )

        ax.set_xlim(0.0, 0.5)
        ax.set_xlabel("Spatial frequency (cycles/pixel)")
        ax.set_ylabel("FRC")
        ax.set_title("Fourier Ring Correlation vs. ground truth")
        ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.2)

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

    return fig
