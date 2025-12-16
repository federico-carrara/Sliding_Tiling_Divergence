# Usage Examples

## Basic 2D Analysis

Compare two 2D prediction methods:

```bash
analyze-experiment \
    --model_name microsplit \
    --dataset Flywing \
    --predictions "results/pred_og.tiff,results/pred_sw.tiff" \
    --method_names "Original,SlidingWindow" \
    --save_dir ./analysis_results/flywing \
    --inner_tile_size 32 \
    --bins 200 \
    --channel 0 \
    --padding 48
```

## 3D Analysis (Same Tile Size)

```bash
analyze-experiment \
    --model_name HDN \
    --dataset Denoising3D \
    --predictions "pred1.tiff,pred2.tiff,pred3.tiff" \
    --method_names "Method1,Method2,Method3" \
    --save_dir ./results/3d_analysis \
    --inner_tile_size "5,32,32" \
    --bins 200 \
    --channel 0 \
    --padding 16
```

## 3D Analysis (Per-Method Tile Sizes)

When different methods use different tile sizes:

```bash
analyze-experiment \
    --model_name HDN \
    --dataset Denoising3D \
    --predictions "pred_z4.tiff,pred_z5.tiff,pred_z6.tiff" \
    --method_names "Z4,Z5,Z6" \
    --save_dir ./results/3d_multisize \
    --inner_tile_size "4,32,32;5,32,32;6,32,32" \
    --bins 200 \
    --padding "16,16,16" \
    --channel 0
```

## Real-World Example: PAVIA_ATN

```bash
analyze-experiment \
    --dataset PAVIA_ATN \
    --model_name microsplit \
    --predictions "/path/to/pred_og_128_OT.tiff,/path/to/pred_og.pkl,/path/to/pred_swt.pkl" \
    --method_names "OuterTiling,InnerTiling,SWT" \
    --save_dir ./results/PAVIA_ATN \
    --inner_tile_size 32 \
    --bins 200 \
    --padding "32,16,16"
```

## Real-World Example: Care3D (Different Z-dimensions)

```bash
analyze-experiment \
    --dataset Care3D \
    --model_name microsplit \
    --predictions "/path/to/swt_G5-32-32.tiff,/path/to/og_G9-32-32.tif,/path/to/og_lc_G5-32-32.tif" \
    --method_names "SWT,OG,OG_LC" \
    --save_dir ./results/CARE3D \
    --inner_tile_size "5,32,32;9,32,32;5,32,32" \
    --bins 200 \
    --padding 0
```

**Note**: Tile sizes extracted from filenames (G5-32-32 and G9-32-32).

## Multi-Channel Analysis

Analyze both channels separately:

```bash
analyze-experiment \
    --dataset TwoChannel \
    --model_name microsplit \
    --predictions "pred1.tiff,pred2.tiff" \
    --method_names "OG,SW" \
    --save_dir ./results/multichannel \
    --inner_tile_size 32 \
    --bins 200 \
    --channel 2 \
    --padding 48
```

Creates subdirectories: `Channel_0/` and `Channel_1/`

## Different Padding Per Method

```bash
analyze-experiment \
    --dataset Mixed \
    --model_name microsplit \
    --predictions "pred_pad48.tiff,pred_pad64.tiff,pred_pad32.tiff" \
    --method_names "Pad48,Pad64,Pad32" \
    --save_dir ./results/mixed_padding \
    --inner_tile_size 32 \
    --bins 200 \
    --channel 0 \
    --padding "48,64,32"
```

## Decoding Filename Patterns

Extract tile sizes from filenames:

```
prediction_swt_P64_G5-32-32_M01.tiff → --inner_tile_size "5,32,32"
Test_P64_G9-32-32_M10_Sk0/pred.tif → --inner_tile_size "9,32,32"
pred_og_128.pkl → --inner_tile_size 128
```

**Pattern**: `G{z}-{y}-{x}` or `_{size}` in filename

## Output Files

```
save_dir/
└── Gradient_Analysis/
    ├── gradient_histograms_all_methods.png  # Visual comparison
    └── summary_kl_divergence.txt            # KL scores
```

## Interpreting Results

Lower KL divergence = better gradient consistency = fewer tiling artifacts

```
KL Divergence Results:
  OuterTiling  : 0.234567  ❌ WORST (high artifacts)
  InnerTiling  : 0.156789
  SWT          : 0.012345  ✅ BEST (minimal artifacts)
```

## Python API Example

```python
from pathlib import Path
from analysis_pipeline.core.analysis import run_gradient_analysis_multi
from analysis_pipeline.utils import load_prediction, ensure_4d, remove_padding

# Load and preprocess
predictions = []
for file, pad in zip(["pred1.tiff", "pred2.tiff"], [48, 48]):
    pred = load_prediction(file)
    pred = ensure_4d(remove_padding(pred, pad))
    predictions.append(pred)

# Analyze
run_gradient_analysis_multi(
    predictions_list=predictions,
    method_names=["Original", "Modified"],
    save_dir=Path("./results"),
    inner_tile_size=[32, 32],
    bins=200,
    channel=0,
)
```

## Common Issues

| Error | Solution |
|-------|----------|
| Length mismatch | Ensure equal number of predictions, names, and paddings |
| File not found | Use absolute paths |
| Wrong gradients | Verify tile sizes match training configuration |

## Tips

1. Start with 100-200 bins
2. Use absolute paths for long file paths
3. Extract tile sizes from filenames
4. Check `summary_kl_divergence.txt` for quantitative comparison
5. View `gradient_histograms_all_methods.png` for visual inspection
