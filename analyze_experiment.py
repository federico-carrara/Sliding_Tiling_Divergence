#!/usr/bin/env python3
"""
analyze_experiment.py

Generalized experiment analysis pipeline supporting 2-5 input predictions.
Refactored to accept multiple prediction files and compare them systematically.
"""

import argparse
from os import remove
from pathlib import Path
import pickle
import dill
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from utils.analysis_utils import (
    summarize_gradients_multi,
)
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import get_input, get_target
from microsplit_reproducibility.utils.paper_metrics import compute_high_snr_stats

import sys
sys.path.append("/home/aman.kukde/sliding_windowed_tiling/microsplit/eval_microsplit/")

try:
    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

# ===============================================
# Utilities
# ===============================================

def load_prediction(path):
    """
    Load prediction from file (.pkl, .dill, or .tiff).
    Expected TIFF shape: (N, H, W, C) after transpose
    """
    path = Path(path)
    if path.suffix in [".pkl", ".dill"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        # TIFF expected shape: (N, C, H, W) → transpose to (N, H, W, C)
        return tiff.imread(path)#.transpose(0, 2, 3, 1)
        # return tiff.imread(path).transpose(0, 2, 3, 1)

def ensure_4d(arr):
    """Ensure array has 4 dimensions (N, H, W, C)."""
    if len(arr.shape) == 3:
        return arr[..., np.newaxis]
    return arr

def remove_padding(arr, pad):
    """Remove padding from array (supports 4D and 5D)."""
    if len(arr.shape) == 4:  # (N, H, W, C)
        return arr[:, pad:-pad, pad:-pad, :]
    elif len(arr.shape) == 5:  # (N, D, H, W, C)
        return arr[:, :, pad:-pad, pad:-pad, :]
    return arr

# ===============================================
# Analysis Functions
# ===============================================

def run_gradient_based_analysis_multi(
    predictions_list: List[np.ndarray],
    method_names: List[str],
    save_dir: Path,
    inner_tile_size: List[int]| int = 32,
    bins: int = 200,
    channel: int = None,
    border = 16
) -> None:
    """
    Run gradient-based analysis for multiple prediction methods.
    
    Args:
        predictions_list: List of prediction arrays
        method_names: Names of methods
        save_dir: Save directory
        inner_tile_size: Tile size used during prediction
        bins: Number of histogram bins
        channel: Channel to analyze (None for all)
    """
    print("Running gradient-based analysis...")
    if isinstance(inner_tile_size, int):
        inner_tile_size = [inner_tile_size] * len(predictions_list)
    else:
        assert len(inner_tile_size) == len(predictions_list), "inner_tile_size length must match number of predictions"
    # Determine if 4D or 5D
    is_4d = len(predictions_list[0].shape) == 4
    
    if is_4d:
        from utils.gradient_utils import GradientUtils2D as GradientUtils
        border_size = border
    else:
        from utils.gradient_utils import GradientUtils3D as GradientUtils
        border_size = [0, border, border]
    # inner_tile_size = [[4,32,32],[4,32,32],[4,64,64]]
    # inner_tile_size = [[5,32,32],[5,32,32],[5,32,32]]
    # Compute gradient utilities for each method
    grad_utils_list = []
    for pred, method_name, tile_size in zip(predictions_list, method_names, inner_tile_size):
        print(f"  Computing gradients for {method_name}...")
        grad_utils = GradientUtils(pred, tile_size=tile_size, border_size=border_size, channel=channel)
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

def remove_padding(arr,pad):
    if len(arr.shape) == 4:
        return arr[:,pad:-pad,pad:-pad,:]
    if len(arr.shape) == 5:
        return arr[:,:,pad:-pad,pad:-pad,:]

def parse_list_input(value):
    try:
        return [int(value)]
    except ValueError:
        return [int(v) for v in value.split(",")]
    
def parse_comma_separated(value):
    """Parse comma-separated integers."""
    try:
        return [int(v) for v in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated values: {value}")
# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-method experiment analysis (compare 2-5 predictions)"
    )
    
    # Required arguments
    parser.add_argument("--model_name", required=True, choices=["usplit", "microsplit", "HDN"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--predictions",
        required=True,
        type=str,
        help="Comma-separated list of prediction files (.tiff/.pkl), e.g., 'pred1.tiff,pred2.tiff,pred3.tiff'"
    )
    parser.add_argument("--method_names", required=True, type=str,
                       help="Comma-separated method names, e.g., 'OG,SW,Method3'")
    
    # Optional arguments
    parser.add_argument("--save_dir", required=True, help="Directory to save results")
    parser.add_argument("--inner_tile_size", type=parse_comma_separated, default=[32])
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--padding", type=parse_comma_separated, default=[48], help="Padding to remove from predictions")
    
    # Analysis flags
    parser.add_argument("--gradient_analysis", action="store_true", default=True)
    parser.add_argument("--qualitative_analysis", action="store_true", default=True)
    parser.add_argument("--skip_gradient_analysis", action="store_true", help="Skip gradient analysis")
    
    args = parser.parse_args()
    
    # Parse inputs
    pred_files = [p.strip() for p in args.predictions.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]
    
    # Validation
    if len(pred_files) != len(method_names):
        raise ValueError(f"Number of predictions ({len(pred_files)}) must match method names ({len(method_names)})")
    
    if len(pred_files) < 2:
        raise ValueError("At least 2 predictions required")
    
    if len(pred_files) > 5:
        raise ValueError("Maximum 5 predictions supported")
    
    print(f"\n{'='*60}")
    print(f"MULTI-METHOD ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Methods: {', '.join(method_names)}")
    print(f"Predictions: {pred_files}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all predictions
    print("Loading predictions...")
    predictions_list = []
    for pred_file, method_name, pad in zip(pred_files, method_names, args.padding):
        print(f"  Loading {method_name}: {pred_file}")
        pred = load_prediction(pred_file)
        if len(pred.shape) == 6: 
            pred = np.squeeze(pred,axis = 0)
        pred = ensure_4d(pred)
        print(f"padding recieved {pad}")
        pred = remove_padding(pred, pad)
        
        predictions_list.append(pred)
        print(f"    Shape: {pred.shape}")
    
    # Determine which analyses to run
    run_gradient = args.gradient_analysis and not args.skip_gradient_analysis
    

    if args.channel == 2:
        for channel in range(args.channel):
            print("\n" + "-" * 60)
            run_gradient_based_analysis_multi(
                predictions_list=predictions_list,
                method_names=method_names,
                save_dir=save_dir/f"Channel_{channel}",
                inner_tile_size=args.inner_tile_size,
                bins=args.bins,
                channel=channel,
                border = 0
            )
    else:
        print("\n" + "-" * 60)
        run_gradient_based_analysis_multi(
            predictions_list=predictions_list,
            method_names=method_names,
            save_dir=save_dir,
            inner_tile_size=args.inner_tile_size,
            bins=args.bins,
            channel=args.channel,
            border = 0
        )


if __name__ == "__main__":
    main()
