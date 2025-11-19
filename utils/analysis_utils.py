# analysis_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from usplit.core.psnr import RangeInvariantPsnr as psnr
from utils.gradient_utils import GradientUtils
from utils.plot_utils import (
    plot_multiple_hist,
    plot_multiple_bar,
    plot_kl_heatmaps_for_range,
    normalize_histogram,
    compute_kl_matrix,
    plot_multiple_boxplots
)
from typing import Dict, Tuple, Any
# -------------------------------
# Basic metric utilities
# -------------------------------


def compute_psnr(pred1, pred2):
    """Compute PSNR between two numpy arrays."""
    return psnr(pred1, pred2, data_range=pred1.max() - pred1.min())

def compute_peakiness(hist, bin_edges):
    """
    Calculates the 'peakiness' of a histogram, defined as the sum of the top 10% bin masses after normalization.

    Parameters:
        hist (array-like): The histogram bin counts.
        bin_edges (array-like): The edges of the histogram bins (unused in calculation).

    Returns:
        float: The sum of the top 10% normalized bin masses, representing the histogram's peakiness.
    """
    """Measure 'peakiness' of a histogram = ratio of top 10% bin mass to total."""


    hist = normalize_histogram(hist)
    sorted_vals = np.sort(hist)[::-1]
    top_frac = int(0.1 * len(sorted_vals))
    return np.sum(sorted_vals[:top_frac])


def summarize_gradients(
    grad_utils_og: Any,
    grad_utils_sw: Any,
    num_bins: int,
    channel: int,
    save_dir: str | Path,
) -> None:
    """
    Generate and save key gradient visualizations and metrics
    comparing Original (OG) vs Sliding Window (SW) methods.

    Args:
        grad_utils_og: Gradient utility object for original method.
        grad_utils_sw: Gradient utility object for sliding window method.
        num_bins: Number of bins for histogram computation.
        channel: Target channel index for KL divergence heatmap.
        save_dir: Directory where plots and summaries will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # === 1. Extract gradient arrays ===
    grad_data = _extract_gradients(grad_utils_og, grad_utils_sw)

    # === 2. Compute histograms ===
    bin_edges, histograms = _compute_all_histograms(grad_data, num_bins)
    
    _plot_combined_histogram(histograms, bin_edges, grad_data, save_dir)

    # === 3. Plot visualizations ===
    _plot_histograms(histograms, bin_edges, save_dir)
    
    
    # _plot_boxplots(grad_data, save_dir)

    # _plot_bar_charts(histograms, bin_edges, save_dir)

    # === 4. KL Divergence Heatmaps ===
    _plot_kl_heatmaps(grad_utils_og, grad_utils_sw, bin_edges, channel, save_dir)

    # === 5. Compute and write metrics summary ===
    _write_summary(histograms, save_dir)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _extract_gradients(
    grad_utils_og: Any,
    grad_utils_sw: Any
) -> Dict[str, np.ndarray]:
    """
    Extract edge and middle gradients from both OG and SW gradient utilities.
    
    Returns:
        Dictionary of gradient arrays (both raw and normalized).
    """
    # Raw gradients (unnormalized)
    grad_edge_og_raw = grad_utils_og.grad_edges
    grad_mid_og_raw = grad_utils_og.grad_middle
    grad_edge_sw_raw = grad_utils_sw.grad_edges
    grad_mid_sw_raw = grad_utils_sw.grad_middle
    
    # Normalized gradients (self-normalized)
    grad_edge_og = grad_utils_og._normalize_gradients(grad_utils_og.grad_edges)
    grad_mid_og = grad_utils_og._normalize_gradients(grad_utils_og.grad_middle)
    grad_edge_sw = grad_utils_sw._normalize_gradients(grad_utils_sw.grad_edges)
    grad_mid_sw = grad_utils_sw._normalize_gradients(grad_utils_sw.grad_middle)
    
    grad_mid_combined = np.concatenate([grad_mid_og, grad_mid_sw])
    
    return {
        # Normalized
        "edge_og": grad_edge_og,
        "mid_og": grad_mid_og,
        "edge_sw": grad_edge_sw,
        "mid_sw": grad_mid_sw,
        "mid_combined": grad_mid_combined,
        # Raw (for re-normalization)
        "edge_og_raw": grad_edge_og_raw,
        "mid_og_raw": grad_mid_og_raw,
        "edge_sw_raw": grad_edge_sw_raw,
        "mid_sw_raw": grad_mid_sw_raw,
    }



def _compute_all_histograms(
    grad_data: Dict[str, np.ndarray],
    num_bins: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute bin edges and histograms for all gradient sets.

    Returns:
        bin_edges: Computed histogram bin edges.
        hists: Dictionary of computed histograms.
    """
    bin_edges = GradientUtils.get_bin_edges(
        [grad_data["edge_og"], grad_data["edge_sw"],
         grad_data["mid_og"], grad_data["mid_sw"]],
        num_bins=num_bins,
    )

    hists = {
        "edge_og": GradientUtils.compute_histograms(grad_data["edge_og"], bin_edges),
        "mid_og": GradientUtils.compute_histograms(grad_data["mid_og"], bin_edges),
        "edge_sw": GradientUtils.compute_histograms(grad_data["edge_sw"], bin_edges),
        "mid_sw": GradientUtils.compute_histograms(grad_data["mid_sw"], bin_edges),
        "mid_combined": GradientUtils.compute_histograms(grad_data["mid_combined"], bin_edges),
    }

    return bin_edges, hists


def _plot_histograms(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    save_dir: str | Path,
) -> None:
    """
    Plot histogram comparisons for OG vs SW and save to disk.
    """
    fig, axs = plt.subplots(1, 3, figsize=(25, 5), sharey=True)

    plot_multiple_hist(
        axs[0],
        histograms=[h["edge_og"], h["mid_og"]],
        bin_edges=bin_edges,
        labels=["Gradient at Edges", "Gradients at Middle of Tiles"],
        colors=["blue", "black"],
        title="Gradients of Original vs In the Middle of Tiles",
        legend=True,
    )

    plot_multiple_hist(
        axs[1],
        histograms=[h["edge_sw"], h["mid_sw"]],
        bin_edges=bin_edges,
        labels=["Gradient at Edges", "Gradients at Middle of Tiles"],
        colors=["red", "black"],
        title="Gradients of SW vs In the Middle of Tiles",
        legend=True,
    )

    plot_multiple_hist(
        axs[2],
        histograms=[h["edge_og"], h["edge_sw"], h["mid_combined"]],
        bin_edges=bin_edges,
        labels=[
            "Gradient at Edges OG",
            "Gradient at Edges SW",
            "Combined Gradients in the Middle of Tiles",
        ],
        colors=["blue", "red", "black"],
        title="Combined Gradients in the Middle of Tiles",
        legend=True,
    )

    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_histograms_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_combined_histogram(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    grad_data: Dict[str, np.ndarray],
    save_dir: str | Path,
) -> None:
    """
    Plot histogram comparisons using both self-normalized and combined-normalized gradients.
    """
    # Compute mean and std from combined middle gradients (already normalized, so we use raw instead)
    mid_combined_raw = np.concatenate([grad_data["mid_og_raw"], grad_data["mid_sw_raw"]])
    mu_combined = mid_combined_raw.mean()
    sigma_combined = mid_combined_raw.std()
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    
    # Row 1: Self-normalized
    plot_multiple_hist(
        axs[0, 0],
        histograms=[h["edge_og"], h["mid_og"]],
        bin_edges=bin_edges,
        labels=["Edge OG (self-norm)", "Middle OG (self-norm)"],
        colors=["blue", "black"],
        title="OG Method - Self Normalized",
        legend=True,
    )
    
    plot_multiple_hist(
        axs[0, 1],
        histograms=[h["edge_sw"], h["mid_sw"]],
        bin_edges=bin_edges,
        labels=["Edge SW (self-norm)", "Middle SW (self-norm)"],
        colors=["red", "black"],
        title="SW Method - Self Normalized",
        legend=True,
    )
    
    # Row 2: Combined-normalized (using raw gradients)
    hist_edge_og_combined = GradientUtils.compute_histograms(
        (grad_data["edge_og_raw"] - mu_combined) / (sigma_combined + 1e-8), bin_edges
    )
    hist_edge_sw_combined = GradientUtils.compute_histograms(
        (grad_data["edge_sw_raw"] - mu_combined) / (sigma_combined + 1e-8), bin_edges
    )
    hist_mid_combined_norm = GradientUtils.compute_histograms(
        (mid_combined_raw - mu_combined) / (sigma_combined + 1e-8), bin_edges
    )
    
    plot_multiple_hist(
        axs[1, 0],
        histograms=[hist_edge_og_combined, hist_mid_combined_norm],
        bin_edges=bin_edges,
        labels=["Edge OG (combined-norm)", "Middle Combined (combined-norm)"],
        colors=["blue", "black"],
        title="OG Edges - Combined Normalized",
        legend=True,
    )
    
    plot_multiple_hist(
        axs[1, 1],
        histograms=[hist_edge_sw_combined, hist_mid_combined_norm],
        bin_edges=bin_edges,
        labels=["Edge SW (combined-norm)", "Middle Combined (combined-norm)"],
        colors=["red", "black"],
        title="SW Edges - Combined Normalized",
        legend=True,
    )
    
    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_histograms_combined_normalized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_boxplots(
    grad_data: Dict[str, np.ndarray],
    save_dir: str | Path,
) -> None:
    """
    Generate boxplots comparing OG and SW gradient distributions.
    """
    fig, axs = plt.subplots(1, 2, figsize=(25, 5), sharey=True)

    plot_multiple_boxplots(
        axs=axs,
        arrays_list=[
            [grad_data["edge_og"], grad_data["mid_og"]],
            [grad_data["mid_sw"], grad_data["edge_sw"]],
        ],
        labels_list=[
            ["Gradient at Edges", "Gradients at Middle of Tiles"],
            ["Gradients at Middle of Tiles", "Gradient at Edges"],
        ],
        colors_list=[
            ["blue", "black"],
            ["black", "red"],
        ],
        titles_list=[
            "Gradients of Original vs In the Middle of Tiles",
            "Gradients of SW vs In the Middle of Tiles",
        ],
    )

    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_boxplots_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_charts(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    save_dir: str | Path,
) -> None:
    """
    Create bar chart comparisons between edge and middle gradients.
    """
    fig, ax = plt.subplots(4, 1, figsize=(17, 12))

    plot_multiple_bar(
        ax[0], [h["edge_sw"], h["mid_sw"]],
        ["SW: Edge of Tiles", "SW: Middle of Tiles"],
        ["red", "black"],
        "Sliding Window Gradients: Edge vs Middle",
        25, bin_edges[:-1],
    )

    plot_multiple_bar(
        ax[1], [h["edge_og"], h["mid_og"]],
        ["OG: Edge of Tiles", "OG: Middle of Tiles"],
        ["blue", "black"],
        "Original Image Gradients: Edge vs Middle",
        25, bin_edges[:-1],
    )

    plot_multiple_bar(
        ax[2], [h["mid_og"] - h["edge_og"], h["mid_sw"] - h["edge_sw"]],
        ["OG: Middle - Edge", "SW: Middle - Edge"],
        ["blue", "red"],
        "Histogram Differences (Middle minus Edge) per Image",
        25, bin_edges[:-1],
    )

    plot_multiple_bar(
        ax[3], [h["edge_sw"] - h["edge_og"], h["mid_sw"] - h["mid_og"]],
        ["Edge: SW - OG", "Middle: SW - OG"],
        ["orange", "black"],
        "Histogram Differences Between SW and OG at Tile Positions",
        25, bin_edges[:-1],
    )

    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_bar_charts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_kl_heatmaps(
    grad_utils_og: Any,
    grad_utils_sw: Any,
    bin_edges: np.ndarray,
    channel: int,
    save_dir: str | Path,
) -> None:
    """
    Generate and save KL divergence heatmaps for the given channel range.
    """
    fig_kl = plot_kl_heatmaps_for_range(
        [grad_utils_og, grad_utils_sw],
        bin_edges,
        start=29,
        end=33,
        channels=channel,
        labels=["OG", "SW"],
    )
    if fig_kl is not None:
        fig_kl.savefig(Path(save_dir) / "kl_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig_kl)


def _write_summary(
    h: Dict[str, np.ndarray],
    save_dir: str | Path,
) -> None:
    """
    Compute key metrics (peakiness, KL divergence) and write summary to file.
    """
    peakiness_og = GradientUtils.get_peakiness_scores(h["edge_og"], h["mid_og"])[-1]
    peakiness_sw = GradientUtils.get_peakiness_scores(h["edge_sw"], h["mid_sw"])[-1]
    peakiness_delta = peakiness_og - peakiness_sw

    kl_edge_mid_og = compute_kl_matrix([h["edge_og"], h["mid_og"]])
    kl_edge_mid_sw = compute_kl_matrix([h["edge_sw"], h["mid_sw"]])

    summary_path = Path(save_dir) / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== Gradient Analysis Summary ===\n\n")
        f.write("Peakiness Scores (Lower is better):\n")
        f.write(f"  Original Method: {peakiness_og:.6f}\n")
        f.write(f"  Sliding Window Method: {peakiness_sw:.6f}\n")
        f.write(f"  Δ (OG - SW): {peakiness_delta:.6f}\n")

        if peakiness_delta > 0:
            f.write("  ✅ Sliding Window Method performs better (lower peakiness)\n")
        else:
            f.write("  ❌ Original Method performs better (lower peakiness)\n")

        f.write(f"\nKL Divergence OG (edge vs mid): {kl_edge_mid_og[0,1]:.6f}\n")
        f.write(f"KL Divergence SW (edge vs mid): {kl_edge_mid_sw[0,1]:.6f}\n")

    print(f"✅ Gradient analysis summary saved to {summary_path}")


# -------------------------------
# PSNR + reconstruction helpers
# -------------------------------

def compute_psnr_and_plot(pred_sw, pred_og, target, save_dir):
    """
    Compare SW vs OG predictions against target and save PSNR plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    psnr_og = compute_psnr(pred_og, target)
    psnr_sw = compute_psnr(pred_sw, target)
    delta = psnr_sw - psnr_og

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["OG", "SW"], [psnr_og, psnr_sw], color=["gray", "orange"])
    ax.set_title(f"PSNR Comparison (Î”={delta:.2f})")
    ax.set_ylabel("PSNR (dB)")
    fig.savefig(Path(save_dir) / "psnr_comparison.png", dpi=300)
    plt.close(fig)

    with open(Path(save_dir) / "psnr_summary.txt", "w") as f:
        f.write(f"OG PSNR: {psnr_og:.4f}\nSW PSNR: {psnr_sw:.4f}\nÎ”: {delta:.4f}\n")

    print(f"✅ PSNR analysis saved to {save_dir}")