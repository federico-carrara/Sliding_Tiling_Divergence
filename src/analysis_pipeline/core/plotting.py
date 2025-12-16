"""Plotting utilities for gradient analysis and visualization."""

from typing import List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from .metrics import compute_kl_matrix, normalize_histogram


def plot_multiple_hist(
    ax: plt.Axes,
    histograms: List[np.ndarray],
    bin_edges: np.ndarray,
    labels: List[str],
    colors: List[str],
    title: str,
    legend: bool = False,
) -> None:
    """
    Plot multiple precomputed histograms with fitted normal distributions.

    Args:
        ax: Matplotlib axis to plot on
        histograms: List of histogram count arrays
        bin_edges: Shared bin edges for all histograms
        labels: Labels for each histogram
        colors: Colors for each histogram
        title: Plot title
        legend: Whether to display legend
    """
    if not (len(histograms) == len(labels) == len(colors)):
        raise ValueError("histograms, labels, and colors must have the same length")

    bin_edges = np.asarray(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)

    data_handles = []
    fit_handles = []

    for counts, label, color in zip(histograms, labels, colors):
        if np.sum(counts) == 0:
            continue

        # Normalize histogram to density
        area = np.trapz(counts, bin_centers)
        density = counts / area if area > 0 else counts

        # Weighted mean & std
        mu = np.average(bin_centers, weights=density)
        var = np.average((bin_centers - mu) ** 2, weights=density)
        std = np.sqrt(var)

        # Plot histogram bars
        bars = ax.bar(
            bin_centers,
            density,
            width=np.diff(bin_edges),
            alpha=0.5,
            color=color,
            label=label,
            edgecolor="none",
        )
        data_handles.append(bars[0])

        # Plot normal fit line
        (line,) = ax.plot(
            x,
            norm.pdf(x, mu, std),
            color=color,
            lw=1.8,
            label=f"{label} fit: μ={mu:.2f}, σ={std:.2f}",
        )
        fit_handles.append(line)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.set_tick_params(labelleft=True)

    if legend:
        # Left legend: fit info
        leg_fit = ax.legend(
            handles=fit_handles,
            loc="upper left",
            fontsize=8,
            frameon=True,
            title="Normal Fit",
        )
        ax.add_artist(leg_fit)

        # Right legend: histogram names
        ax.legend(
            handles=data_handles,
            labels=labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            title="Histograms",
        )


def plot_multiple_boxplots(
    axs: List[plt.Axes],
    arrays_list: List[List[np.ndarray]],
    labels_list: List[List[str]],
    colors_list: List[List[str]],
    titles_list: List[str],
    legend: bool = False,
) -> None:
    """
    Plot multiple boxplots dynamically on given axes.

    Args:
        axs: List of matplotlib axes
        arrays_list: List of lists of arrays to plot
        labels_list: Labels for each array in each subplot
        colors_list: Colors for each box in each subplot
        titles_list: Titles for each subplot
        legend: Whether to add legend
    """
    if not (
        len(axs)
        == len(arrays_list)
        == len(labels_list)
        == len(colors_list)
        == len(titles_list)
    ):
        raise ValueError(
            "Length of axs, arrays_list, labels_list, colors_list, "
            "titles_list must match"
        )

    for ax, arrays, labels, colors, title in zip(
        axs, arrays_list, labels_list, colors_list, titles_list
    ):
        bp = ax.boxplot(arrays, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.grid(True)
        if legend:
            ax.legend(labels)


def plot_kl_heatmaps_for_range(
    grad_utils_list: List,
    bin_edges: np.ndarray,
    start: int = 29,
    end: int = 34,
    channels: int = 1,
    labels: Optional[List[str]] = None,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Generate KL divergence heatmaps for a range of tile positions.

    Args:
        grad_utils_list: List of gradient utility objects
        bin_edges: Bin edges for histograms
        start: Start index for position range
        end: End index for position range
        channels: Channel to analyze
        labels: Method labels
        cmap: Colormap for heatmap

    Returns:
        Matplotlib figure
    """
    from .gradient_analysis import GradientUtils

    n_utils = len(grad_utils_list)
    if labels is None:
        labels = [f"Model{i}" for i in range(n_utils)]

    middle_hists = []
    for gu in grad_utils_list:
        grad_mid = gu.get_gradients_at("middle", channels=channels)
        middle_hists.append(GradientUtils.compute_histograms(grad_mid, bin_edges))

    n_plots = end - start + 1
    fig, axes = plt.subplots(
        1, n_plots, figsize=(10 * n_plots, 7.5), constrained_layout=False
    )
    if n_plots == 1:
        axes = [axes]

    kl_mats = []
    for index in range(start, end + 1):
        histograms = []
        for gu, mid_hist in zip(grad_utils_list, middle_hists):
            grad_at_idx = gu.get_gradients_at(index, channels=channels)
            hist_at_idx = GradientUtils.compute_histograms(grad_at_idx, bin_edges)
            histograms.extend([hist_at_idx, mid_hist])
        kl_mats.append(compute_kl_matrix(histograms))

    vmin = min(np.min(mat) for mat in kl_mats)
    vmax = max(np.max(mat) for mat in kl_mats)

    for ax, index, kl_mat in zip(axes, range(start, end + 1), kl_mats):
        hist_labels = []
        for label in labels:
            hist_labels.extend([f"{label}-Edge", f"{label}-Mid"])
        sns.heatmap(
            kl_mat,
            annot=True,
            fmt=".3f",
            xticklabels=hist_labels,
            yticklabels=hist_labels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Index {index}")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.8,
        label="KL Divergence",
    )
    fig.suptitle("KL Divergence Between Gradient Distributions", fontsize=16)
    return fig


def save_figure(fig: plt.Figure, save_path: Path, dpi: int = 300) -> None:
    """
    Save figure to file and close it.

    Args:
        fig: Matplotlib figure
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path.name}")
