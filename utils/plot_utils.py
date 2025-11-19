# plotting_utils.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
from matplotlib.gridspec import GridSpec
from microsplit_reproducibility.utils.paper_metrics import avg_range_inv_psnr
from microsplit_reproducibility.utils.paper_metrics import RangeInvariantPsnr
from utils.gradient_utils import GradientUtils2D as GradientUtils
import pandas as pd
# --------------------------
# Gradient Histogram & KL Functions
# --------------------------
def normalize_histogram(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=float)
    return arr / (arr.sum() + eps)

def kl_divergence(p, q, eps=1e-12):
    p = normalize_histogram(p, eps)
    q = normalize_histogram(q, eps)
    return np.sum(p * np.log((p + eps) / (q + eps)))

def compute_kl_matrix(histograms):
    n = len(histograms)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_mat[i, j] = kl_divergence(histograms[i], histograms[j])
    return kl_mat

def plot_multiple_boxplots(
    axs, arrays_list, labels_list, colors_list, titles_list, legend=False
):
    """
    Plot multiple boxplots dynamically on given axes.

    Parameters
    ----------
    axs : list or array of matplotlib axes
        Axes to plot on.
    arrays_list : list of list of arrays
        Each sublist contains arrays to plot on corresponding axis.
    labels_list : list of list of str
        Labels for each array in each subplot.
    colors_list : list of list of str
        Colors for each box in each subplot.
    titles_list : list of str
        Titles for each subplot.
    legend : bool, optional
        If True, add legend to each subplot.
    """
    if not (len(axs) == len(arrays_list) == len(labels_list) == len(colors_list) == len(titles_list)):
        raise ValueError("Length of axs, arrays_list, labels_list, colors_list, titles_list must match")

    for ax, arrays, labels, colors, title in zip(axs, arrays_list, labels_list, colors_list, titles_list):
        bp = ax.boxplot(arrays, patch_artist=True, labels=labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.grid(True)
        if legend:
            ax.legend(labels)


def plot_multiple_hist(ax, histograms, bin_edges, labels, colors, title, legend=False):
    """
    Plot multiple precomputed histograms with fitted normal distributions on a single axis.
    Shows two legend boxes: left = fit info, right = histogram labels.
    Keeps y-axis labels visible on all subplots (even if sharey=True).
    """
    if not (len(histograms) == len(labels) == len(colors)):
        raise ValueError("histograms, labels, and colors must have the same length")

    # Ensure numpy array
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
            bin_centers, density, width=np.diff(bin_edges),
            alpha=0.5, color=color, label=label, edgecolor='none'
        )
        data_handles.append(bars[0])

        # Plot normal fit line
        line, = ax.plot(
            x, norm.pdf(x, mu, std),
            color=color, lw=1.8, label=f"{label} fit: μ={mu:.2f}, σ={std:.2f}"
        )
        fit_handles.append(line)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Make sure y-axis labels show even when sharey=True
    ax.yaxis.set_tick_params(labelleft=True)

    if legend:
        # Left legend: fit info
        leg_fit = ax.legend(
            handles=fit_handles,
            loc="upper left",
            fontsize=8,
            frameon=True,
            title="Normal Fit"
        )
        ax.add_artist(leg_fit)  # Add first legend manually

        # Right legend: histogram names
        ax.legend(
            handles=data_handles,
            labels=labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            title="Histograms"
        )

def plot_multiple_bar(ax, arrays, bin_edges, labels, colors, title, smooth_window=3, legend=True):
    n_bins = len(arrays[0])
    if len(bin_edges) != n_bins:
        bin_edges = np.broadcast_arrays(bin_edges, np.zeros(n_bins))
        raise ValueError("Length of bin_edges must match length of arrays")

    bar_width = np.min(np.diff(bin_edges)) * 0.4

    for i, (arr, label, color) in enumerate(zip(arrays, labels, colors)):
        ax.bar(bin_edges + i * bar_width, arr, width=bar_width, color=color, alpha=0.7, label=label)

        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            arr_smooth = np.convolve(arr, kernel, mode='same')
        else:
            arr_smooth = arr

        ax.plot(bin_edges, arr_smooth, color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Bin edges")
    ax.set_ylabel("Value")
    ax.grid(True, axis='y')
    if legend:
        ax.legend()

def plot_kl_heatmaps_for_range(grad_utils_list, bin_edges, start=29, end=34, channels=1, labels=None, cmap="coolwarm"):
    n_utils = len(grad_utils_list)
    if labels is None:
        labels = [f"Model{i}" for i in range(n_utils)]

    middle_hists = []
    for gu in grad_utils_list:
        grad_mid = gu.get_gradients_at("middle", channels=channels)
        middle_hists.append(GradientUtils.compute_histograms(grad_mid, bin_edges))

    n_plots = end - start + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 7.5), constrained_layout=False)
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
        sns.heatmap(kl_mat, annot=True, fmt=".3f", xticklabels=hist_labels,
                    yticklabels=hist_labels, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, ax=ax)
        ax.set_title(f"Index {index}")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.8,
        label="KL Divergence"
    )
    fig.suptitle("KL Divergence Between Gradient Distributions", fontsize=16)
    # plt.show()
    return fig

def plot_multiple_hist(ax, histograms, bin_edges, labels, colors, title, legend=False):
    """
    Plot multiple precomputed histograms with fitted normal distributions on a single axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    histograms : list of array_like
        Each element is a 1D array of counts (same bins for all histograms).
    bin_edges : array_like
        Shared bin edges used for all histograms.
    labels : list of str
        Labels for each histogram.
    colors : list of str
        Colors for each histogram.
    title : str
        Title for the plot.
    legend : bool, optional
        If True, display a legend (default: False).
    """
    if not (len(histograms) == len(labels) == len(colors)):
        raise ValueError("histograms, labels, and colors must have the same length")

    # Convert bin_edges to array (in case it's a list)
    bin_edges = np.asarray(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Filter out empty histograms
    non_empty_histograms = [h for h in histograms if np.sum(h) > 0]
    if not non_empty_histograms:
        raise ValueError("All input histograms are empty, cannot plot")

    all_min, all_max = bin_edges[0], bin_edges[-1]
    if all_min == all_max:
        all_min -= 1e-3
        all_max += 1e-3

    x = np.linspace(all_min, all_max, 1000)

    # Plot each histogram and its fitted normal curve
    for counts, label, color in zip(histograms, labels, colors):
        if np.sum(counts) == 0:
            continue

        # Normalize to density
        area = np.trapz(counts, bin_centers)
        density = counts / area if area > 0 else counts

        # Fit normal distribution (weighted mean/std)
        mu = np.average(bin_centers, weights=density)
        variance = np.average((bin_centers - mu)**2, weights=density)
        std = np.sqrt(variance)

        # Plot histogram and fit
        ax.bar(bin_centers, density, width=np.diff(bin_edges), alpha=0.5, color=color, label=label)
        ax.plot(x, norm.pdf(x, mu, std), linestyle='-', color=color,
                label=f'{label}\nFit μ={mu:.2f}, σ={std:.2f}')

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True)
    if legend:
        ax.legend()

# --------------------------
# PSNR Functions
# --------------------------
def plot_per_image_psnr(df):
    for ch in df['Channel'].unique():
        sub_df = df[df['Channel'] == ch]
        plt.figure(figsize=(8, 4))
        plt.plot(sub_df['Image'], sub_df['PSNR_OG'], label='PSNR_OG', marker='o')
        plt.plot(sub_df['Image'], sub_df['PSNR_WIN'], label='PSNR_WIN', marker='x')
        plt.title(f'Per-Image PSNR for {ch}')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

def plot_avg_psnr_with_ci(avg_df, n_samples=6):
    t_crit = t.ppf(0.975, df=n_samples-1)
    ci_og = avg_df['SE_OG'] * t_crit
    ci_win = avg_df['SE_WIN'] * t_crit

    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, yerr=ci_og, capsize=5, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, yerr=ci_win, capsize=5, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Avg PSNR with 95% CI')
    ax.set_title('Mean PSNR with Confidence Intervals')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    # plt.show()

def plot_psnr_difference(avg_df,save_dir):
    delta = avg_df['Avg_PSNR_WIN'] - avg_df['Avg_PSNR_OG']
    x = np.arange(len(avg_df))
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, delta, color='orange')
    plt.xticks(x, avg_df['Channel'])
    plt.ylabel('Δ PSNR (WIN - OG)')
    plt.title('PSNR Improvement of WIN over OG')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    for bar, d in zip(bars, delta):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{d:.2f}', ha='center', va='bottom')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'PSNR_Improvement.png', dpi=300)
    # plt.show()

def plot_avg_psnr_zoomed(avg_df,save_dir):
    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Average PSNR')
    ax.set_title('Zoomed-In Average PSNR Comparison')
    ax.legend()
    y_min = min(avg_df[['Avg_PSNR_OG', 'Avg_PSNR_WIN']].min()) - 0.2
    y_max = max(avg_df[['Avg_PSNR_OG', 'Avg_PSNR_WIN']].max()) + 0.2
    ax.set_ylim([y_min, y_max])
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'Zoomed_Avg_PSNR_Comparison.png', dpi=300)
    # plt.show()

def plot_all_psnr(avg_df, n_samples=6, save_dir=None):
    t_crit = t.ppf(0.975, df=n_samples - 1)

    # Safely get SE columns or fill with zeros if missing
    ci_og = avg_df.get('SE_OG', np.zeros(len(avg_df))) * t_crit
    ci_win = avg_df.get('SE_WIN', np.zeros(len(avg_df))) * t_crit

    x = np.arange(len(avg_df))
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    # Paired line plot
    for i in x:
        axs[0].plot([0, 1],
                    [avg_df['Avg_PSNR_OG'][i], avg_df['Avg_PSNR_WIN'][i]],
                    marker='o', linewidth=2)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(['OG', 'WIN'])
    axs[0].set_ylabel('Avg PSNR')
    axs[0].set_title('Paired Line Plot (OG vs WIN)')
    axs[0].grid(True, axis='y')

    # Error bar plot
    axs[1].errorbar(x - 0.05, avg_df['Avg_PSNR_OG'], yerr=ci_og, fmt='o', capsize=5, label='OG')
    axs[1].errorbar(x + 0.05, avg_df['Avg_PSNR_WIN'], yerr=ci_win, fmt='o', capsize=5, label='WIN')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(avg_df.get('Channel', [str(i) for i in x]))
    axs[1].set_ylabel('Avg PSNR')
    axs[1].set_title('Dot Plot with 95% Confidence Interval')
    axs[1].legend()
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'All_PSNR_Plots.png', dpi=300)

def to_scalar_tuple(x):
    if isinstance(x, (tuple, list)):
        return float(x[0]), float(x[1])
    elif hasattr(x, 'item'):
        return float(x.item()), 0.0
    return float(x), 0.0

def compute_psnr_and_plot(test_data, img_og, img_win, save_dir):


    # --- Per-image PSNR ---
    records = []
    for i in range(min(6, len(test_data))):
        tar = test_data[i:i+1]
        inp_win = img_win[i:i+1]
        inp_og = img_og[i:i+1]

        for ch in range(2):
            psnr_og, _ = to_scalar_tuple(RangeInvariantPsnr(tar[..., ch], inp_og[..., ch]))
            psnr_win, _ = to_scalar_tuple(RangeInvariantPsnr(tar[..., ch], inp_win[..., ch]))

            records.append({
                'Image': i,
                'Channel': f'Ch{ch+1}',
                'PSNR_OG': psnr_og,
                'PSNR_WIN': psnr_win
            })

    psnr_df = pd.DataFrame(records)

    # --- Average PSNR + STD ---
    avg_records = []
    for ch in [0, 1]:
        avg_og, std_og = to_scalar_tuple(avg_range_inv_psnr(test_data[..., ch], img_og[..., ch]))
        avg_win, std_win = to_scalar_tuple(avg_range_inv_psnr(test_data[..., ch], img_win[..., ch]))

        avg_records.append({
            'Channel': f'Ch{ch+1}',
            'Avg_PSNR_OG': avg_og,
            'Std_OG': std_og,
            'Avg_PSNR_WIN': avg_win,
            'Std_WIN': std_win
        })

    avg_df = pd.DataFrame(avg_records)

    # --- Plot: Per-image PSNR ---
    for ch in ['Ch1', 'Ch2']:
        sub_df = psnr_df[psnr_df['Channel'] == ch]
        plt.figure(figsize=(8, 4))
        plt.plot(sub_df['Image'], sub_df['PSNR_OG'], label='PSNR_OG', marker='o')
        plt.plot(sub_df['Image'], sub_df['PSNR_WIN'], label='PSNR_WIN', marker='x')
        plt.title(f'Per-Image PSNR for {ch}')
        plt.xlabel('Image Index')
        plt.ylabel('PSNR')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'Per_Image_PSNR_{ch}.png', dpi=300)
        # plt.show()

    # --- Plot: Average PSNR with Std ---
    x = np.arange(len(avg_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, avg_df['Avg_PSNR_OG'], width, yerr=avg_df['Std_OG'], capsize=5, label='OG')
    ax.bar(x + width/2, avg_df['Avg_PSNR_WIN'], width, yerr=avg_df['Std_WIN'], capsize=5, label='WIN')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_df['Channel'])
    ax.set_ylabel('Average PSNR')
    ax.set_title('Average PSNR with StdDev by Channel')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'Avg_PSNR_with_Std.png', dpi=300)
    # plt.show()

    return psnr_df, avg_df



def full_frame_evaluation(
    predictions_list,
    tar_list,
    inp_list,
    metrics_list,
    frame_idx,
    titles,
    save_path=None
):
    """
    Plots the input, target, and prediction images along with associated metrics
    for a given frame, comparing different evaluation methods.
    Custom layout:
    Row 1: [Empty] [Input Image] [Empty] | [Empty] [Input Image] [Empty]
    Row 2: Target Ch0 | Target Ch1 | Target Ch0 | Target Ch1
    Row 3: Pred Ch0 | Pred Ch1 | Pred Ch0 | Pred Ch1
    Row 4: Metrics Ch0 | Metrics Ch1 | Metrics Ch0 | Metrics Ch1

    Args:
        predictions_list (list): A list of prediction arrays. Each array
                                 should have a shape compatible with (..., H, W, 2).
        tar_list (list): A list of target arrays. Each array should have a
                         shape compatible with (..., H, W, 2).
        inp_list (list): A list of input arrays. Each array should have a
                         shape compatible with (..., H, W) (grayscale).
        metrics_list (list): A list of dictionaries, where each dictionary
                             contains metrics for a specific evaluation method.
                             Metrics are expected to be per-channel, e.g.,
                             {'metric_name': [(value_ch0, 0.0), (value_ch1, 0.0)]}.
        frame_idx (int): The index of the current frame being evaluated.
        titles (list): A list of strings, where each string is the title
                       for a set of results (e.g., ["windowed", "MMSE 64"]).
        save_path (str, optional): If provided, the plot will be saved to this path.
                                   Defaults to None (plot is displayed).
    """
    
    fig = plt.figure(figsize=(24, 20)) 

    gs_main = GridSpec(4, 6, figure=fig, wspace=0.05, hspace=0.15, 
                    height_ratios=[1.2, 1, 1, 0.4]) 

    metrics_data = [] 

    for metrics_dict in metrics_list:
        method_metrics = {'ch0': {}, 'ch1': {}}
        for metric_name, values in metrics_dict.items():
            method_metrics['ch0'][metric_name] = values[0][0]
            method_metrics['ch1'][metric_name] = values[1][0]
        metrics_data.append(method_metrics)

    print(" --- Row 1 (gs_main[0, :]): Input Images (centered) ---")
    for i in range(2): 
        col_offset = i * 3 
        print(f"Plotting input for method {titles[i]} at column offset {col_offset}")
        current_inp = (inp_list[i][...,0] + inp_list[i][...,1])/2.  # Accessing the correct frame_idx for input

        gs_input_section = gs_main[0, col_offset:col_offset+3].subgridspec(1, 1, wspace=0.05, hspace=0.05)
        
        ax_input = fig.add_subplot(gs_input_section[0, 0])
        ax_input.imshow(current_inp)
        ax_input.set_title(f"{titles[i]}\nInput (Frame {frame_idx})")
        ax_input.axis('off')

    current_target = tar_list[0] # Accessing the correct frame_idx for target

    current_prediction_win = predictions_list[0]
    current_prediction_og = predictions_list[1]

    print("--- Row 2 (gs_main[1, :]): Target Channels ---")
    gs_targets = gs_main[1, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    ax_tar_win_ch0 = fig.add_subplot(gs_targets[0, 0])
    ax_tar_win_ch0.imshow(current_target[..., 0])
    ax_tar_win_ch0.set_title(f"{titles[0]}\nTarget Ch0")
    ax_tar_win_ch0.axis('off')

    ax_tar_og_ch0 = fig.add_subplot(gs_targets[0, 1])
    ax_tar_og_ch0.imshow(current_target[..., 0])
    ax_tar_og_ch0.set_title(f"{titles[1]}\nTarget Ch0")
    ax_tar_og_ch0.axis('off')

    ax_tar_win_ch1 = fig.add_subplot(gs_targets[0, 2])
    ax_tar_win_ch1.imshow(current_target[..., 1])
    ax_tar_win_ch1.set_title(f"{titles[0]}\nTarget Ch1")
    ax_tar_win_ch1.axis('off')

    ax_tar_og_ch1 = fig.add_subplot(gs_targets[0, 3])
    ax_tar_og_ch1.imshow(current_target[..., 1])
    ax_tar_og_ch1.set_title(f"{titles[1]}\nTarget Ch1")
    ax_tar_og_ch1.axis('off')

    print("--- Row 3 (gs_main[2, :]): Prediction Channels ---")
    gs_preds = gs_main[2, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    ax_pred_win_ch0 = fig.add_subplot(gs_preds[0, 0])
    ax_pred_win_ch0.imshow(current_prediction_win[..., 0])
    ax_pred_win_ch0.set_title(f"{titles[0]}\nPrediction Ch0")
    ax_pred_win_ch0.axis('off')

    ax_pred_og_ch0 = fig.add_subplot(gs_preds[0, 1])
    ax_pred_og_ch0.imshow(current_prediction_og[..., 0])
    ax_pred_og_ch0.set_title(f"{titles[1]}\nPrediction Ch0")
    ax_pred_og_ch0.axis('off')

    ax_pred_win_ch1 = fig.add_subplot(gs_preds[0, 2])
    ax_pred_win_ch1.imshow(current_prediction_win[..., 1])
    ax_pred_win_ch1.set_title(f"{titles[0]}\nPrediction Ch1")
    ax_pred_win_ch1.axis('off')

    ax_pred_og_ch1 = fig.add_subplot(gs_preds[0, 3])
    ax_pred_og_ch1.imshow(current_prediction_og[..., 1])
    ax_pred_og_ch1.set_title(f"{titles[1]}\nPrediction Ch1")
    ax_pred_og_ch1.axis('off')
    
    # --- Row 4 (gs_main[3, :]): Metrics ---
    gs_metrics = gs_main[3, :].subgridspec(1, 4, wspace=0.05, hspace=0.05)
    
    metric_axes = [
        fig.add_subplot(gs_metrics[0, 0]), 
        fig.add_subplot(gs_metrics[0, 1]), 
        fig.add_subplot(gs_metrics[0, 2]), 
        fig.add_subplot(gs_metrics[0, 3])  
    ]

    for ax in metric_axes:
        ax.axis('off') 

    def plot_metrics_with_bolding(ax, method_idx, channel_key, title):
        ax.set_title(title)
        
        metric_names = sorted(metrics_data[method_idx][channel_key].keys())
        
        y_position = 0.5 
        line_spacing = 1.0 / (len(metric_names) + 1) 
        
        bbox_height_per_line = 0.22 
        total_bbox_height = len(metric_names) * bbox_height_per_line

        rect = plt.Rectangle((0, 0.5 - total_bbox_height/2), 1, total_bbox_height, 
                            transform=ax.transAxes,
                            facecolor='lightgray', edgecolor='k', linewidth=1, alpha=0.7)
        ax.add_patch(rect)

        for i, metric_name in enumerate(metric_names):
            current_value = metrics_data[method_idx][channel_key][metric_name]
            other_method_idx = 1 if method_idx == 0 else 0
            other_value = metrics_data[other_method_idx][channel_key][metric_name]

            text_weight = 'normal'
            # Only bold if the current value is strictly greater than the other value
            if round(current_value, 2) > round(other_value, 2):
                text_weight = 'bold'
            
            ax.text(0.5, y_position + ( (len(metric_names) - 1) / 2 - i) * line_spacing,
                    f"{metric_name}: {current_value:.2f}",
                    verticalalignment='center', horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=12,
                    weight=text_weight) 

    # Plotting for Windowed (Method 0)
    plot_metrics_with_bolding(metric_axes[0], 0, 'ch0', f"")
    plot_metrics_with_bolding(metric_axes[1], 1, 'ch0', f"") 
    plot_metrics_with_bolding(metric_axes[2], 0, 'ch1', f"") 
    plot_metrics_with_bolding(metric_axes[3], 1, 'ch1', f"") 


    plt.tight_layout() 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def full_frame_evaluation_zoomed(
    predictions_list,
    tar_list,
    inp_list,
    frame_idx,
    titles,
    save_path=None
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    import numpy as np

    # Validate input lengths
    n_methods = len(predictions_list)
    assert len(tar_list) == n_methods and len(inp_list) == n_methods, \
        "predictions_list, tar_list, and inp_list must have the same length"
    assert len(titles) >= n_methods, \
        "titles must have at least as many entries as there are methods"

    # === Normalization range based on GT ===
    all_targets = np.concatenate([t for t in tar_list], axis=-1)
    vmin = np.nanpercentile(all_targets, 1)
    vmax = np.nanpercentile(all_targets, 99)
    if vmin == vmax:
        vmin, vmax = 0, 1

    # === Figure setup ===
    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    plt.rcParams.update({'font.size': 12})

    # === Compute full input and crop region ===
    full_input_img = np.mean(inp_list[0][..., :2], axis=-1)
    H, W = full_input_img.shape
    crop_h = H // 4
    crop_w = W // 4
    crop_y = (H - crop_h) // 2
    crop_x = (W - crop_w) // 2
    crop_slice_y = slice(crop_y, crop_y + crop_h)
    crop_slice_x = slice(crop_x, crop_x + crop_w)
    rect_kwargs = dict(linewidth=2, edgecolor='yellow', facecolor='none')

    # === Prepare crops ===
    preds_crops = []
    targets_crops = []
    for i in range(n_methods):
        targets_crops.append(tar_list[i][crop_slice_y, crop_slice_x, :])
        preds_crops.append(predictions_list[i][crop_slice_y, crop_slice_x, :])

    # === Row 1: GTs, full input, zoomed input ===
    full_target = tar_list[0]

    # GT channel 0
    ax_gt0 = fig.add_subplot(gs[0, 0])
    ax_gt0.imshow(full_target[..., 0], cmap='gray', vmin=vmin, vmax=vmax)
    ax_gt0.set_title("GT Channel 0 (Full)")
    ax_gt0.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_gt0.axis('off')

    # GT channel 1
    ax_gt1 = fig.add_subplot(gs[0, 1])
    ax_gt1.imshow(full_target[..., 1], cmap='gray', vmin=vmin, vmax=vmax)
    ax_gt1.set_title("GT Channel 1 (Full)")
    ax_gt1.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_gt1.axis('off')

    # Full input
    ax_inp_full = fig.add_subplot(gs[0, 2])
    ax_inp_full.imshow(full_input_img, cmap='gray', vmin=vmin, vmax=vmax)
    ax_inp_full.set_title(f"{titles[0]} Input (Frame {frame_idx})")
    ax_inp_full.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_inp_full.axis('off')

    # Zoomed input
    ax_inp_zoom = fig.add_subplot(gs[0, 3])
    ax_inp_zoom.imshow(full_input_img[crop_slice_y, crop_slice_x], cmap='gray', vmin=vmin, vmax=vmax)
    ax_inp_zoom.set_title("Zoomed Input")
    ax_inp_zoom.axis('off')

    # === Row 2: Predictions (Channel 0) ===
    ax_pred_orig_ch0 = fig.add_subplot(gs[1, 0])
    ax_pred_orig_ch0.imshow(preds_crops[0][..., 0], cmap='gray', vmin=vmin, vmax=vmax)
    ax_pred_orig_ch0.set_title(f"{titles[0]} Prediction Ch0")
    ax_pred_orig_ch0.axis('off')

    ax_pred_slide_ch0 = fig.add_subplot(gs[1, 1])
    ax_pred_slide_ch0.imshow(preds_crops[1][..., 0], cmap='gray', vmin=vmin, vmax=vmax)
    ax_pred_slide_ch0.set_title(f"{titles[1]} Prediction Ch0")
    ax_pred_slide_ch0.axis('off')

    # Add GT ch0 again for visual alignment reference
    ax_gt_crop_ch0 = fig.add_subplot(gs[1, 2])
    ax_gt_crop_ch0.imshow(targets_crops[0][..., 0], cmap='gray', vmin=vmin, vmax=vmax)
    ax_gt_crop_ch0.set_title("GT Ch0 (Crop)")
    ax_gt_crop_ch0.axis('off')

    # Optional: difference map between two methods (abs diff)
    diff_ch0 = np.abs(preds_crops[0][..., 0] - preds_crops[1][..., 0])
    ax_diff_ch0 = fig.add_subplot(gs[1, 3])
    ax_diff_ch0.imshow(diff_ch0, cmap='magma')
    ax_diff_ch0.set_title("|Diff| Ch0")
    ax_diff_ch0.axis('off')

    # === Row 3: Predictions (Channel 1) ===
    ax_pred_orig_ch1 = fig.add_subplot(gs[2, 0])
    ax_pred_orig_ch1.imshow(preds_crops[0][..., 1], cmap='gray', vmin=vmin, vmax=vmax)
    ax_pred_orig_ch1.set_title(f"{titles[0]} Prediction Ch1")
    ax_pred_orig_ch1.axis('off')

    ax_pred_slide_ch1 = fig.add_subplot(gs[2, 1])
    ax_pred_slide_ch1.imshow(preds_crops[1][..., 1], cmap='gray', vmin=vmin, vmax=vmax)
    ax_pred_slide_ch1.set_title(f"{titles[1]} Prediction Ch1")
    ax_pred_slide_ch1.axis('off')

    # Add GT ch1 again for comparison
    ax_gt_crop_ch1 = fig.add_subplot(gs[2, 2])
    ax_gt_crop_ch1.imshow(targets_crops[0][..., 1], cmap='gray', vmin=vmin, vmax=vmax)
    ax_gt_crop_ch1.set_title("GT Ch1 (Crop)")
    ax_gt_crop_ch1.axis('off')

    # Optional: difference map for ch1
    diff_ch1 = np.abs(preds_crops[0][..., 1] - preds_crops[1][..., 1])
    ax_diff_ch1 = fig.add_subplot(gs[2, 3])
    ax_diff_ch1.imshow(diff_ch1, cmap='magma')
    ax_diff_ch1.set_title("|Diff| Ch1")
    ax_diff_ch1.axis('off')

    # === Margins and spacing ===
    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        top=0.90,
        bottom=0.10,
        wspace=0.10,
        hspace=0.50
    )

    # === Save or show ===
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()
def full_frame_evaluation_zoomed(
    predictions_list,
    tar_list,
    inp_list,
    frame_idx,
    titles,          # e.g. ["Original", "Sliding Window"]
    save_path=None
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as patches
    import numpy as np

    # Validate input lengths
    n_methods = len(predictions_list)
    assert n_methods == 2, "This layout assumes exactly two methods (Original + Sliding Window)"
    assert len(tar_list) == n_methods and len(inp_list) == n_methods, \
        "predictions_list, tar_list, and inp_list must have the same length"
    assert len(titles) >= n_methods, "titles must have at least as many entries as methods"

    # === Normalization range based on GT (per channel) ===
    all_targets = np.concatenate([t for t in tar_list], axis=0)  # stack along batch dimension
    num_channels = all_targets.shape[-1]
    vmins, vmaxs = [], []
    for ch in range(num_channels):
        vmin = np.nanpercentile(all_targets[..., ch], 1)
        vmax = np.nanpercentile(all_targets[..., ch], 99)
        if vmin == vmax:  # fallback
            vmin, vmax = 0, 1
        vmins.append(vmin)
        vmaxs.append(vmax)

    # === Figure setup ===
    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    gs = GridSpec(4, 4, figure=fig)
    plt.rcParams.update({'font.size': 12})

    # === Compute crop region ===
    full_input_img = np.mean(inp_list[0][..., :2], axis=-1)
    H, W = full_input_img.shape
    crop_h = H // 4
    crop_w = W // 4
    crop_y = (H - crop_h) // 2
    crop_x = (W - crop_w) // 2
    crop_slice_y = slice(crop_y, crop_y + crop_h)
    crop_slice_x = slice(crop_x, crop_x + crop_w)

    rect_kwargs = dict(linewidth=2, edgecolor='yellow', facecolor='none')

    # === Row 1: Full GTs, Input, Zoomed Input ===
    # GT Ch0
    ax_gt0 = fig.add_subplot(gs[0, 0])
    ax_gt0.imshow(tar_list[0][..., 0], cmap='gray', vmin=vmins[0], vmax=vmaxs[0])
    ax_gt0.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_gt0.set_title("GT Channel 0 (Full)")
    ax_gt0.axis('off')

    # GT Ch1
    ax_gt1 = fig.add_subplot(gs[0, 1])
    ax_gt1.imshow(tar_list[0][..., 1], cmap='gray', vmin=vmins[1], vmax=vmaxs[1])
    ax_gt1.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_gt1.set_title("GT Channel 1 (Full)")
    ax_gt1.axis('off')

    # Full Input
    ax_input = fig.add_subplot(gs[0, 2])
    ax_input.imshow(full_input_img, cmap='gray')
    ax_input.add_patch(patches.Rectangle((crop_x, crop_y), crop_w, crop_h, **rect_kwargs))
    ax_input.set_title(f"Input (Frame {frame_idx})")
    ax_input.axis('off')

    # Zoomed Input
    ax_input_zoom = fig.add_subplot(gs[0, 3])
    ax_input_zoom.imshow(full_input_img[crop_slice_y, crop_slice_x], cmap='gray')
    ax_input_zoom.set_title("Input Zoomed")
    ax_input_zoom.axis('off')

    # === Row 2: Zoomed GT Channels (ch0, ch1) + 2 blanks ===
    ax_gtz0 = fig.add_subplot(gs[1, 0])
    ax_gtz0.imshow(tar_list[0][crop_slice_y, crop_slice_x, 0], cmap='gray',
                   vmin=vmins[0], vmax=vmaxs[0])
    ax_gtz0.set_title("GT Channel 0 (Zoomed)")
    ax_gtz0.axis('off')

    ax_gtz1 = fig.add_subplot(gs[1, 1])
    ax_gtz1.imshow(tar_list[0][crop_slice_y, crop_slice_x, 1], cmap='gray',
                   vmin=vmins[1], vmax=vmaxs[1])
    ax_gtz1.set_title("GT Channel 1 (Zoomed)")
    ax_gtz1.axis('off')

    for col in [2, 3]:
        fig.add_subplot(gs[1, col]).axis('off')

    # === Row 3: Predictions Ch0 (Full) ===
    ax_pred0_orig = fig.add_subplot(gs[2, 0])
    ax_pred0_orig.imshow(predictions_list[0][..., 0], cmap='gray',
                         vmin=vmins[0], vmax=vmaxs[0])
    ax_pred0_orig.set_title(f"{titles[0]} Prediction Ch0 (Full)")
    ax_pred0_orig.axis('off')

    ax_pred0_slide = fig.add_subplot(gs[2, 1])
    ax_pred0_slide.imshow(predictions_list[1][..., 0], cmap='gray',
                          vmin=vmins[0], vmax=vmaxs[0])
    ax_pred0_slide.set_title(f"{titles[1]} Prediction Ch0 (Full)")
    ax_pred0_slide.axis('off')

    for col in [2, 3]:
        fig.add_subplot(gs[2, col]).axis('off')

    # === Row 4: Predictions Ch1 (Full) ===
    ax_pred1_orig = fig.add_subplot(gs[3, 0])
    ax_pred1_orig.imshow(predictions_list[0][..., 1], cmap='gray',
                         vmin=vmins[1], vmax=vmaxs[1])
    ax_pred1_orig.set_title(f"{titles[0]} Prediction Ch1 (Full)")
    ax_pred1_orig.axis('off')

    ax_pred1_slide = fig.add_subplot(gs[3, 1])
    ax_pred1_slide.imshow(predictions_list[1][..., 1], cmap='gray',
                          vmin=vmins[1], vmax=vmaxs[1])
    ax_pred1_slide.set_title(f"{titles[1]} Prediction Ch1 (Full)")
    ax_pred1_slide.axis('off')

    for col in [2, 3]:
        fig.add_subplot(gs[3, col]).axis('off')

    # === Margins ===
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.93, bottom=0.05,
        wspace=0.12, hspace=0.25
    )

    # === Save or show ===
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()
