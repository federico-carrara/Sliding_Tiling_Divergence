"""Headline FRC figure: mean curves with 95% CI bands per method.

For each dataset we report one figure with the per-method mean FRC curve,
a shaded 95% CI band, and (optionally) dashed verticals at the expected
seam harmonics ``k/S``. Non-overlapping CI bands at the dip locations are
themselves visual evidence of significance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis_pipeline.frc.aggregation import FRCMultiMethodReport


def plot_frc_curves(
    report: FRCMultiMethodReport,
    tile_inner_sizes: Optional[dict[str, Optional[int]]] = None,
    save_path: Optional[Path] = None,
    *,
    channel: int = 0,
) -> "plt.Figure":
    """Plot mean FRC curves with 95% CI bands for every method.

    Parameters
    ----------
    report : FRCMultiMethodReport
        Aggregated multi-method FRC report.
    tile_inner_sizes : dict of str to (int or None), optional
        Per-method inner-tile size in image pixels. When provided for a
        method, dashed verticals are drawn at expected harmonic frequencies
        ``k / S`` (in cycles/pixel) for ``k = 1, ..., N // (2*S)`` where
        ``N`` is the image side length used in the FRC computation. Methods
        with ``None`` (e.g. SWiTi) skip the verticals.
    save_path : pathlib.Path, optional
        If provided, save the figure to this path (PNG suggested) at 150 dpi.
    channel : int, default=0
        Channel index whose per-channel mean curve and CI band to plot. Methods
        lacking this channel are skipped.

    Returns
    -------
    matplotlib.figure.Figure
        Figure handle, for downstream customisation or display.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    methods = list(report.methods.keys())
    palette = plt.get_cmap("tab10").colors

    for i, name in enumerate(methods):
        m = report.methods[name]
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
        fig.savefig(save_path, dpi=150)

    return fig
