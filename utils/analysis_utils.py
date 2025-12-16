# analysis_utils.py
"""
Generalized analysis utilities for comparing multiple (N>2) prediction methods.
Supports comparing 2-5 input images/methods side-by-side.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from usplit.core.psnr import RangeInvariantPsnr as psnr

from utils.gradient_utils import GradientUtils2D as GradientUtils

from utils.plot_utils import (
    plot_multiple_hist,
    plot_multiple_bar,
    plot_kl_heatmaps_for_range,
    normalize_histogram,
    compute_kl_matrix,
    plot_multiple_boxplots
)
from typing import Dict, Tuple, Any, List

# ===============================================
# Basic Metrics
# ===============================================



def compute_peakiness(hist):
    """
    Calculate histogram 'peakiness': sum of top 10% bin masses after normalization.
    Lower peakiness = better (smoother gradients).
    """
    hist = normalize_histogram(hist)
    sorted_vals = np.sort(hist)[::-1]
    top_frac = int(0.1 * len(sorted_vals))
    return np.sum(sorted_vals[:top_frac])

# ===============================================
# Multi-Method Gradient Analysis
# ===============================================

def summarize_gradients_multi(
    grad_utils_list: List[Any],
    method_names: List[str],
    num_bins: int,
    channel: int,
    save_dir: str | Path,
) -> None:
    """
    Generate gradient visualizations and metrics comparing N methods.
    
    Args:
        grad_utils_list: List of gradient utility objects (one per method)
        method_names: List of method names (e.g., ["OG", "SW", "Method3"])
        num_bins: Number of bins for histograms
        channel: Target channel index for analysis
        save_dir: Directory to save outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_methods = len(grad_utils_list)
    assert len(method_names) == n_methods, "Number of names must match number of methods"
    
    # === 1. Extract gradients for all methods ===
    grad_data = _extract_gradients_multi(grad_utils_list)
    
    # === 2. Compute histograms ===
    bin_edges, histograms = _compute_histograms_multi(grad_data, num_bins, n_methods)
    
    # === 3. Plot visualizations ===
    _plot_histograms_multi(histograms, bin_edges, method_names, save_dir)
    _plot_combined_histogram_multi(histograms, bin_edges, grad_data, method_names, save_dir)
    
    # === 4. KL Divergence Heatmaps ===
    _plot_kl_heatmaps_multi(grad_utils_list, bin_edges, method_names, channel, save_dir)
    
    # === 5. Compute and write metrics summary ===
    _write_summary_multi(histograms, method_names, save_dir)

# ===============================================
# Helper: Extract Gradients for N Methods
# ===============================================

def _extract_gradients_multi(grad_utils_list: List[Any]) -> Dict[str, np.ndarray]:
    """
    Extract edge and middle gradients from all methods.
    Returns dict with keys: "edge_0", "mid_0", "edge_1", "mid_1", etc.
    """
    grad_data = {}
    mid_all_raw = []
    
    for i, grad_utils in enumerate(grad_utils_list):
        # Raw gradients
        edge_raw = grad_utils.grad_edges
        mid_raw = grad_utils.grad_middle
        
        # Normalized gradients
        edge_norm = grad_utils._normalize_gradients(edge_raw)
        mid_norm = grad_utils._normalize_gradients(mid_raw)
        
        grad_data[f"edge_{i}_raw"] = edge_raw
        grad_data[f"mid_{i}_raw"] = mid_raw
        grad_data[f"edge_{i}"] = edge_norm
        grad_data[f"mid_{i}"] = mid_norm
        
        mid_all_raw.append(mid_raw)
    
    # Combined middle gradients (all methods concatenated)
    grad_data["mid_combined"] = np.concatenate(mid_all_raw)
    
    return grad_data

# ===============================================
# Helper: Compute Histograms for N Methods
# ===============================================

def _compute_histograms_multi(
    grad_data: Dict[str, np.ndarray],
    num_bins: int,
    n_methods: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute bin edges and histograms for all methods.
    """
    # Collect all gradients to determine bin edges
    all_grads = [grad_data[f"mid_{i}"] for i in range(n_methods)]
    all_grads.extend([grad_data[f"edge_{i}"] for i in range(n_methods)])
    
    bin_edges = GradientUtils.get_bin_edges(all_grads, num_bins=num_bins)
    
    hists = {}
    for i in range(n_methods):
        hists[f"edge_{i}"] = GradientUtils.compute_histograms(grad_data[f"edge_{i}"], bin_edges)
        hists[f"mid_{i}"] = GradientUtils.compute_histograms(grad_data[f"mid_{i}"], bin_edges)
    
    hists["mid_combined"] = GradientUtils.compute_histograms(grad_data["mid_combined"], bin_edges)
    
    return bin_edges, hists

# ===============================================
# Helper: Plot Histograms for N Methods
# ===============================================

def _plot_histograms_multi(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    method_names: List[str],
    save_dir: str | Path,
) -> None:
    """
    Create histogram comparison plots for each method.
    Layout: one subplot per method, showing edge vs middle gradients.
    """
    n_methods = len(method_names)
    colors = ["blue", "red", "green", "orange", "purple"][:n_methods]
    
    fig, axs = plt.subplots(1, n_methods, figsize=(8 * n_methods, 4), sharey=True)
    if n_methods == 1:
        axs = [axs]
    
    for i, (ax, method_name, color) in enumerate(zip(axs, method_names, colors)):
        plot_multiple_hist(
            ax,
            histograms=[h[f"edge_{i}"], h[f"mid_{i}"]],
            bin_edges=bin_edges,
            labels=[f"{method_name}: Edge", f"{method_name}: Middle"],
            colors=[color, "black"],
            title=f"{method_name} - Edge vs Middle Gradients",
            legend=True,
        )
    
    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_histograms_all_methods.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: gradient_histograms_all_methods.png")

# ===============================================
# Helper: Plot Combined Histograms for N Methods
# ===============================================

def _plot_combined_histogram_multi(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    grad_data: Dict[str, np.ndarray],
    method_names: List[str],
    save_dir: str | Path,
) -> None:
    """
    Plot histograms using both self-normalized and combined-normalized gradients.
    """
    n_methods = len(method_names)
    colors = ["blue", "red", "green", "orange", "purple"][:n_methods]
    
    # Combined normalization stats
    mid_combined_raw = grad_data["mid_combined"]
    mu_combined = mid_combined_raw.mean()
    sigma_combined = mid_combined_raw.std()
    
    # 2 rows: self-normalized and combined-normalized
    fig, axs = plt.subplots(2, n_methods, figsize=(8 * n_methods, 8))
    
    for i, (method_name, color) in enumerate(zip(method_names, colors)):
        # Row 0: Self-normalized
        plot_multiple_hist(
            axs[0, i],
            histograms=[h[f"edge_{i}"], h[f"mid_{i}"]],
            bin_edges=bin_edges,
            labels=[f"Edge (self-norm)", f"Middle (self-norm)"],
            colors=[color, "black"],
            title=f"{method_name} - Self Normalized",
            legend=True,
        )
        
        # Row 1: Combined-normalized
        hist_edge_combined = GradientUtils.compute_histograms(
            (grad_data[f"edge_{i}_raw"] - mu_combined) / (sigma_combined + 1e-8), bin_edges
        )
        hist_mid_combined = GradientUtils.compute_histograms(
            (grad_data[f"mid_{i}_raw"] - mu_combined) / (sigma_combined + 1e-8), bin_edges
        )
        
        plot_multiple_hist(
            axs[1, i],
            histograms=[hist_edge_combined, hist_mid_combined],
            bin_edges=bin_edges,
            labels=[f"Edge (combined-norm)", f"Middle (combined-norm)"],
            colors=[color, "black"],
            title=f"{method_name} - Combined Normalized",
            legend=True,
        )
    
    plt.tight_layout()
    fig.savefig(Path(save_dir) / "gradient_histograms_normalized_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: gradient_histograms_normalized_comparison.png")

# ===============================================
# Helper: KL Divergence Heatmaps for N Methods
# ===============================================

def _plot_kl_heatmaps_multi(
    grad_utils_list: List[Any],
    bin_edges: np.ndarray,
    method_names: List[str],
    channel: int,
    save_dir: str | Path,
) -> None:
    """
    Generate KL divergence heatmaps for each method.
    """
    n_methods = len(grad_utils_list)
    
    # For each method, plot KL heatmap comparing edge vs middle
    for i, (grad_utils, method_name) in enumerate(zip(grad_utils_list, method_names)):
        try:
            fig_kl = plot_kl_heatmaps_for_range(
                [grad_utils],
                bin_edges,
                start=29,
                end=33,
                channels=channel,
                labels=[method_name],
            )
            if fig_kl is not None:
                fig_kl.savefig(
                    Path(save_dir) / f"kl_heatmap_{method_name.replace(' ', '_')}.png",
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close(fig_kl)
                print(f"✅ Saved: kl_heatmap_{method_name}.png")
        except Exception as e:
            print(f"⚠️  KL heatmap failed for {method_name}: {e}")

# ===============================================
# Helper: Summary Statistics for N Methods
# ===============================================

def _write_summary_multi(
    h: Dict[str, np.ndarray],
    method_names: List[str],
    save_dir: str | Path,
) -> None:
    """
    Compare peakiness and KL(mid || edge) for each method.
    Lower KL means the method's middle-tile gradient distribution
    is closer to its edge-tile distribution → more stable.
    """
    import numpy as np
    save_dir = Path(save_dir)

    peakiness_scores = {}
    kl_self_divergence = {}  # NEW

    # --- Compute metrics for each method ---
    for i, method_name in enumerate(method_names):

        # Peakiness (unchanged)
        peakiness = GradientUtils.get_peakiness_scores(
            h[f"edge_{i}"],
            h[f"mid_{i}"]
        )[-1]
        peakiness_scores[method_name] = peakiness

        # NEW: KL(mid_i || edge_i)
        kl_mat = compute_kl_matrix([h[f"mid_{i}"], h[f"edge_{i}"]])
        kl_value = kl_mat[0, 1]
        kl_self_divergence[method_name] = kl_value

    # Rank methods by KL divergence
    best_method_kl = min(kl_self_divergence, key=kl_self_divergence.get)
    worst_method_kl = max(kl_self_divergence, key=kl_self_divergence.get)

    # Rank by peakiness (existing logic)
    best_peak = min(peakiness_scores, key=peakiness_scores.get)
    worst_peak = max(peakiness_scores, key=peakiness_scores.get)

    # --- Write summary ---
    summary_path = save_dir / "summary_multi_method.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MULTI-METHOD GRADIENT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Peakiness
        f.write("Peakiness Scores (Lower is Better):\n")
        f.write("-" * 40 + "\n")
        for m in method_names:
            score = peakiness_scores[m]
            f.write(f"  {m:15s}: {score:.6f}")
            if m == best_peak: f.write("  ✅ BEST")
            if m == worst_peak: f.write("  ❌ WORST")
            f.write("\n")

        f.write("\nKL Divergence KL(mid || edge) for Each Method:\n")
        f.write("Lower is Better\n")
        f.write("-" * 40 + "\n")
        for m in method_names:
            kl_val = kl_self_divergence[m]
            f.write(f"  {m:15s}: {kl_val:.6f}")
            if m == best_method_kl: f.write("  ✅ BEST")
            if m == worst_method_kl: f.write("  ❌ WORST")
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Best Method by KL(mid||edge): {best_method_kl}\n")
        f.write(f"Best Method by Peakiness:    {best_peak}\n")
        f.write("=" * 60 + "\n")

    print(f"✅ Summary saved: {summary_path}")