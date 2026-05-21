"""Main analysis functions for multi-method gradient comparison."""

from pathlib import Path
from typing import List, Dict, Any
import os

import numpy as np
import matplotlib.pyplot as plt

from .gradient_analysis import GradientUtils2D, GradientUtils3D
from .metrics import compute_kl_matrix
from .plotting import plot_multiple_hist, plot_kl_heatmaps_for_range, save_figure


def _broadcast_per_method_spec(
    spec: list, n_methods: int, name: str
) -> list[list[int]]:
    """Normalise a per-axis or per-method tiling spec to a per-method list.

    Accepts:
      - a single per-axis spec (e.g. ``[64, 64]`` or ``[4, 64, 64]``) — applied
        to every method;
      - a per-method list of per-axis specs (e.g. ``[[64, 64], [32, 32]]``).

    Returns a list of length ``n_methods``, each entry a list of ints.
    """
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"{name} must be a non-empty list/tuple, got {spec!r}")

    if all(isinstance(v, int) for v in spec):
        return [list(spec) for _ in range(n_methods)]

    if len(spec) != n_methods:
        raise ValueError(
            f"{name} has {len(spec)} per-method entries; expected {n_methods} "
            "to match the number of predictions"
        )
    return [list(s) for s in spec]


def run_gradient_analysis_multi(
    predictions_list: List[np.ndarray],
    method_names: List[str],
    save_dir: Path,
    tile_size: list,
    overlap: list,
    bins: int = 200,
    channel: int = None,
) -> None:
    """
    Run gradient-based analysis for multiple prediction methods.

    Args:
        predictions_list: List of prediction arrays.
        method_names: Names of methods.
        save_dir: Save directory.
        tile_size: Tile size used during prediction. Either a single per-axis
            spec applied to every method (e.g. ``[64, 64]`` for 2D, ``[4, 64,
            64]`` for 3D) or a per-method list of such specs.
        overlap: Tile overlap used during prediction. Same per-method/per-axis
            shape conventions as ``tile_size``.
        bins: Number of histogram bins.
        channel: Channel to analyze (None for all).
    """
    print("Running gradient-based analysis...")

    tile_size = _broadcast_per_method_spec(tile_size, len(predictions_list), "tile_size")
    overlap = _broadcast_per_method_spec(overlap, len(predictions_list), "overlap")

    # Determine 2D vs 3D from prediction layout: (N, C, H, W) vs (N, C, D, H, W).
    is_4d = len(predictions_list[0].shape) == 4
    GradientUtils = GradientUtils2D if is_4d else GradientUtils3D

    # Compute gradient utilities for each method
    grad_utils_list = []
    for pred, method_name, ts, ov in zip(
        predictions_list, method_names, tile_size, overlap
    ):
        print(f"  Computing gradients for {method_name}...")
        grad_utils = GradientUtils(
            pred, tile_size=ts, overlap=ov, channel=channel
        )
        grad_utils_list.append(grad_utils)

    # Run multi-method analysis
    analysis_dir = save_dir / "Gradient_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summarize_gradients_multi(
        grad_utils_list=grad_utils_list,
        method_names=method_names,
        num_bins=bins,
        channel=channel if channel is not None else 0,
        save_dir=analysis_dir,
    )

    print(f"✅ Gradient analysis complete: {analysis_dir}")


def summarize_gradients_multi(
    grad_utils_list: List[Any],
    method_names: List[str],
    num_bins: int,
    channel: int,
    save_dir: Path,
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
    GradientUtilsClass = type(grad_utils_list[0])
    bin_edges, histograms = _compute_histograms_multi(
        grad_data, num_bins, n_methods, GradientUtilsClass
    )

    # === 3. Plot visualizations ===
    _plot_histograms_multi(histograms, bin_edges, method_names, save_dir)
    # Removed: combined histogram and KL heatmaps

    # === 4. Compute and write metrics summary (KL only) ===
    _write_summary_multi(histograms, method_names, save_dir, GradientUtilsClass)


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


def _compute_histograms_multi(
    grad_data: Dict[str, np.ndarray],
    num_bins: int,
    n_methods: int,
    GradientUtils: Any,
) -> tuple:
    """Compute bin edges and histograms for all methods."""
    # Collect all gradients to determine bin edges
    all_grads = [grad_data[f"mid_{i}"] for i in range(n_methods)]
    all_grads.extend([grad_data[f"edge_{i}"] for i in range(n_methods)])

    bin_edges = GradientUtils.get_bin_edges(all_grads, num_bins=num_bins)

    hists = {}
    for i in range(n_methods):
        hists[f"edge_{i}"] = GradientUtils.compute_histograms(
            grad_data[f"edge_{i}"], bin_edges
        )
        hists[f"mid_{i}"] = GradientUtils.compute_histograms(
            grad_data[f"mid_{i}"], bin_edges
        )

    hists["mid_combined"] = GradientUtils.compute_histograms(
        grad_data["mid_combined"], bin_edges
    )

    return bin_edges, hists


def _plot_histograms_multi(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    method_names: List[str],
    save_dir: Path,
) -> None:
    """Create histogram comparison plots for each method."""
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
    save_figure(fig, save_dir / "gradient_histograms_all_methods.png")


def _plot_combined_histogram_multi(
    h: Dict[str, np.ndarray],
    bin_edges: np.ndarray,
    grad_data: Dict[str, np.ndarray],
    method_names: List[str],
    save_dir: Path,
) -> None:
    """Plot histograms using both self-normalized and combined-normalized gradients."""
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
        GradientUtilsClass = type(list(grad_data.values())[0])
        hist_edge_combined = GradientUtilsClass.compute_histograms(
            (grad_data[f"edge_{i}_raw"] - mu_combined) / (sigma_combined + 1e-8),
            bin_edges,
        )
        hist_mid_combined = GradientUtilsClass.compute_histograms(
            (grad_data[f"mid_{i}_raw"] - mu_combined) / (sigma_combined + 1e-8),
            bin_edges,
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
    save_figure(fig, save_dir / "gradient_histograms_normalized_comparison.png")


def _plot_kl_heatmaps_multi(
    grad_utils_list: List[Any],
    bin_edges: np.ndarray,
    method_names: List[str],
    channel: int,
    save_dir: Path,
) -> None:
    """Generate KL divergence heatmaps for each method."""
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
                save_figure(
                    fig_kl,
                    save_dir / f"kl_heatmap_{method_name.replace(' ', '_')}.png",
                )
        except Exception as e:
            print(f"⚠️  KL heatmap failed for {method_name}: {e}")


def _write_summary_multi(
    h: Dict[str, np.ndarray],
    method_names: List[str],
    save_dir: Path,
    GradientUtils: Any,
) -> None:
    """
    Compute KL(mid || edge) for each method.

    Lower KL means the method's middle-tile gradient distribution
    is closer to its edge-tile distribution → more stable.
    """
    save_dir = Path(save_dir)

    kl_self_divergence = {}

    # --- Compute KL divergence for each method ---
    for i, method_name in enumerate(method_names):
        # KL(mid_i || edge_i)
        # NOTE: These histograms are computed from gradients normalized
        # by THIS method's middle gradients' mean/std (not combined dataset).
        # This is the correct normalization as per user requirements.
        kl_mat = compute_kl_matrix([h[f"mid_{i}"], h[f"edge_{i}"]])
        kl_value = kl_mat[0, 1]
        kl_self_divergence[method_name] = kl_value

    # Rank methods by KL divergence
    best_method_kl = min(kl_self_divergence, key=kl_self_divergence.get)
    worst_method_kl = max(kl_self_divergence, key=kl_self_divergence.get)

    # --- Write summary ---
    summary_path = save_dir / "summary_kl_divergence.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("GRADIENT ANALYSIS: KL DIVERGENCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("KL Divergence KL(mid || edge) for Each Method:\n")
        f.write("(Lower is Better - indicates more consistent gradients)\n")
        f.write("-" * 60 + "\n")
        for m in method_names:
            kl_val = kl_self_divergence[m]
            f.write(f"  {m:25s}: {kl_val:.6f}")
            if m == best_method_kl:
                f.write("  ✅ BEST")
            if m == worst_method_kl:
                f.write("  ❌ WORST")
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Best Method (lowest KL): {best_method_kl}\n")
        f.write(f"KL Value: {kl_self_divergence[best_method_kl]:.6f}\n")
        f.write("=" * 60 + "\n")

    print(f"✅ Summary saved: {summary_path}")
