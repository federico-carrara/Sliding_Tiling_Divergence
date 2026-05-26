# Gradient Analysis Pipeline

A gradient-based analysis pipeline for comparing tiling methods in image prediction tasks.

## Features

- **Multi-Method Comparison**: Compare 2-5 different prediction methods
- **KL Divergence Analysis**: Quantify gradient consistency across tile boundaries
- **Gradient Visualization**: Compare edge vs. middle tile gradients
- **Flexible Configuration**: Support for 2D and 3D data with per-method tile sizes

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
analyze-experiment \
    --dataset MyDataset \
    --model_name microsplit \
    --predictions "pred1.tiff,pred2.tiff,pred3.tiff" \
    --method_names "Method1,Method2,Method3" \
    --save_dir ./results \
    --inner_tile_size 32 \
    --bins 200 \
    --padding 48
```

## Output

The analysis produces two files:

1. **`gradient_histograms_all_methods.png`**: Visual comparison of edge vs. middle gradients for each method
2. **`summary_kl_divergence.txt`**: KL divergence scores (lower = better gradient consistency)

### Example Summary Output

```
============================================================
GRADIENT ANALYSIS: KL DIVERGENCE SUMMARY
============================================================

KL Divergence KL(mid || edge) for Each Method:
(Lower is Better - indicates more consistent gradients)
------------------------------------------------------------
  Method1                  : 0.123456  ❌ WORST
  Method2                  : 0.045678
  Method3                  : 0.012345  ✅ BEST

============================================================
Best Method (lowest KL): Method3
KL Value: 0.012345
============================================================
```

**Interpretation**: Lower KL divergence indicates more uniform gradients between tile edges and middles, suggesting better tiling artifact reduction.

## Usage Examples

See [EXAMPLES.md](EXAMPLES.md) for detailed usage scenarios.

## Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset name | `MyDataset` |
| `--model_name` | Model type | `microsplit`, `usplit`, `HDN` |
| `--predictions` | Comma-separated prediction files | `"pred1.tiff,pred2.tiff"` |
| `--method_names` | Comma-separated method names | `"OG,SW"` |
| `--save_dir` | Output directory | `./results` |
| `--inner_tile_size` | Tile size specification | `32` or `"4,32,32;5,32,32"` |
| `--bins` | Number of histogram bins | `200` |
| `--padding` | Padding to remove | `48` or `"32,16,16"` |
| `--channel` | Channel to analyze | `0` (or `2` for both channels) |

## Tile Size Specifications

| Format | Example | Description |
|--------|---------|-------------|
| Single int | `32` | Same 2D tile for all methods |
| 3D same | `"5,32,32"` | Same 3D tile (Z,Y,X) for all |
| Per-method 3D | `"4,32,32;5,32,32"` | Different tiles per method |

**Tip**: Extract tile sizes from prediction filenames (e.g., `G5-32-32` → use `"5,32,32"`)

## Python API

```python
from pathlib import Path
from analysis_pipeline.gradient_test.analysis import run_gradient_analysis_multi
from analysis_pipeline.utils import load_prediction, ensure_4d, remove_padding

# Load predictions
predictions = []
for file in ["pred1.tiff", "pred2.tiff"]:
    pred = load_prediction(file)
    pred = ensure_4d(remove_padding(pred, 48))
    predictions.append(pred)

# Run analysis
run_gradient_analysis_multi(
    predictions_list=predictions,
    method_names=["Original", "Modified"],
    save_dir=Path("./results"),
    inner_tile_size=[32, 32],
    bins=200,
    channel=0,
)
```

## Documentation

- **[EXAMPLES.md](EXAMPLES.md)**: Comprehensive usage examples
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Command syntax cheat sheet
