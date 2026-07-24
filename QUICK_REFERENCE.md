# Quick Reference

Both CLIs run **one method per invocation** and read **`.npz` archives keyed by
image name** (arrays squeeze to channel-first `(C, H, W)` / `(C, D, H, W)`). Run a
command again per method to cover several.

## Gradient test (reference-free)

```bash
run-gradient-test \
    --predictions predictions.npz \
    --method_name inner_tiling \
    --tile_size 64,64 --overlap 32,32 \
    --statistic js \
    --output_dir results/gradient_test
```

- `--tile_size` / `--overlap`: comma-separated per spatial axis. The count of
  `--tile_size` entries sets the dimensionality (`64,64` → 2-D, `16,64,64` → 3-D).
- `--statistic`: one of `kl`, `js`, `ks`, `wasserstein`, `mean_abs_ratio`.
- `--channels 0,1` restricts to channels (default: all).
- No ground truth needed. To test a GT as a seam-free null, run again with the GT
  `.npz` as `--predictions` and `--method_name GT`.

**Outputs** (`--output_dir`): `gradient_test_config.json`,
`{method_name}_gradient_report.json`, `{method_name}_summary.csv`.

## FRC (reference-based)

```bash
compute-frc \
    --predictions predictions.npz \
    --ground_truth ground_truths.npz \
    --method_name inner_tiling \
    --ndim 2 \
    --step 32 \
    --output_dir results/frc
```

- `--ground_truth`: `.npz` keyed by the **same image names** as `--predictions`.
- `--ndim {2,3}`: 3-D volumes are scored per z-slice.
- `--step`: seam interval in px (e.g. `tile_size - overlap`); draws dashed harmonic
  verticals `k/step` on the plots. Omit for no verticals.
- `--no_window` disables the 2-D Hamming window (real images need it — keep on).

**Outputs** (`--output_dir`): `{method_name}_frc_report.json`,
`{method_name}_summary.csv`, `{method_name}_frc_curves_ch{c}.pdf` per channel.

## Interpreting results

- **Gradient test** — `frac_rejected` is the fraction of tiles whose across-seam
  gradients differ significantly from their control strip at level α. Higher →
  more seam artifacts. `median_T` summarises the effect size.
- **FRC** — dips in the mean curve at the seam harmonics `k/step` indicate
  periodic stitching artifacts; the resolution readout is where the curve crosses
  the 1/7 threshold.

## Tips

```bash
run-gradient-test --help
compute-frc --help
```

Save your arrays to `.npz` keyed by image name:

```python
import numpy as np
np.savez("predictions.npz", **{name: array for name, array in images})
```
