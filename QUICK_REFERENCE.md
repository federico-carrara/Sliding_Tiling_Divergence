# Quick Reference

## Command Format

```bash
analyze-experiment \
    --dataset DATASET_NAME \
    --model_name {usplit|microsplit|HDN} \
    --predictions "file1.tiff,file2.pkl,file3.tiff" \
    --method_names "Name1,Name2,Name3" \
    --save_dir ./results \
    --inner_tile_size TILE_SPEC \
    --bins 200 \
    --padding PADDING_SPEC \
    [--channel CHANNEL]
```

## Critical Rules

1. **NO SPACES** in comma-separated lists
   - ✅ `"file1.tiff,file2.tiff"`
   - ❌ `"file1.tiff, file2.tiff"`

2. **Match counts**: predictions = method_names = padding (if per-method)

3. **Tile sizes must match** your prediction configuration

## Tile Size Formats

| Scenario | Format | Example |
|----------|--------|---------|
| 2D, same for all | Single int | `32` |
| 2D, per method | Semicolon-separated | `"32;64;32"` |
| 3D, same for all | Comma-separated | `"5,32,32"` |
| 3D, per method | Semicolon + comma | `"4,32,32;5,32,32;6,32,32"` |

## Padding Formats

| Scenario | Format | Example |
|----------|--------|---------|
| Same for all | Single int | `48` |
| No padding | Zero | `0` |
| Per method | Comma-separated | `"32,16,16"` |

## Output Files

```
results/Gradient_Analysis/
├── gradient_histograms_all_methods.png  # Visual comparison
└── summary_kl_divergence.txt            # KL scores (lower = better)
```

## Interpreting Results

```
KL Divergence (Lower is Better):
  Method1  : 0.234567  ❌ WORST
  Method2  : 0.098765  ✅ BEST
```

Lower KL divergence = more consistent gradients = fewer tiling artifacts

## Common Examples

### 2D Analysis
```bash
analyze-experiment \
    --dataset Flywing \
    --model_name microsplit \
    --predictions "pred_og.tiff,pred_sw.tiff" \
    --method_names "Original,SlidingWindow" \
    --save_dir ./results \
    --inner_tile_size 32 \
    --bins 200 \
    --padding 48
```

### 3D Analysis (Different Z-dimensions)
```bash
analyze-experiment \
    --dataset Care3D \
    --model_name HDN \
    --predictions "pred_z4.tiff,pred_z5.tiff,pred_z6.tiff" \
    --method_names "Z4,Z5,Z6" \
    --save_dir ./results \
    --inner_tile_size "4,32,32;5,32,32;6,32,32" \
    --bins 200 \
    --padding 16
```

### Multi-Channel
```bash
analyze-experiment \
    --dataset TwoChannel \
    --model_name microsplit \
    --predictions "pred1.tiff,pred2.tiff" \
    --method_names "OG,SW" \
    --save_dir ./results \
    --inner_tile_size 32 \
    --bins 200 \
    --padding 48 \
    --channel 2  # Analyzes channels 0 and 1 separately
```

## Decoding Filenames

```
prediction_G5-32-32.tiff → --inner_tile_size "5,32,32"
Test_G9-32-32/pred.tif → --inner_tile_size "9,32,32"
pred_og_128.pkl → --inner_tile_size 128
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| Length mismatch | Count items in predictions, names, padding |
| File not found | Use absolute paths |
| Wrong gradients | Check training tile size |
| Spaces in args | Remove all spaces from comma-separated lists |

## Quick Start

```bash
# Install
pip install -e .

# Run
analyze-experiment --help

# Example
analyze-experiment \
    --dataset MyDataset \
    --model_name microsplit \
    --predictions "pred1.tiff,pred2.tiff" \
    --method_names "OG,SW" \
    --save_dir ./results \
    --inner_tile_size 32 \
    --bins 200 \
    --padding 48
```

For more examples, see [EXAMPLES.md](EXAMPLES.md)
