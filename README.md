# Tilartmetrics

Metrics for **quantifying tiling (stitching) artifacts** in large images produced
by tiled inference. Two complementary metrics ship today:

- **Gradient test** — a reference-free, per-tile permutation hypothesis test. For
  each kept region of the TiledPatching grid it compares across-seam gradients to
  a local control strip, yielding a per-tile statistic `T_tile` and p-value. Two
  per-image scalars are reported: `median(T_tile)` and `frac_rejected` at α.
- **FRC** — reference-based Fourier Ring Correlation against a ground truth. Per
  image it computes the 2-D FRC curve and aggregates a per-frequency mean + 95%
  CI across the test set; stitching artifacts show up as dips at the seam
  harmonics `k / step`.

## Installation

```bash
pip install -e .
```

## Data format

Both CLIs consume **`.npz` archives whose keys are image names** and whose arrays
squeeze to channel-first layout — `(C, H, W)` for 2-D or `(C, D, H, W)` for 3-D.
Save your predictions (and, for FRC, your ground truths) that way, for example:

```python
import numpy as np
np.savez("predictions.npz", **{image_name: array for image_name, array in images})
```

For FRC the ground-truth archive must be keyed by the **same image names** as the
predictions (each prediction is paired with the ground truth under the same key).

Each command runs **one method at a time** and writes one report; run it again per
method to cover several.

## Usage — gradient test

```bash
run-gradient-test \
    --predictions predictions.npz \
    --method_name inner_tiling \
    --tile_size 64,64 --overlap 32,32 \
    --statistic js \
    --output_dir results/gradient_test
```

The spatial dimensionality is inferred from the number of `--tile_size` entries
(`64,64` → 2-D, `16,64,64` → 3-D). The test is reference-free; to test a ground
truth as a seam-free null baseline, run the command again with the ground-truth
`.npz` as `--predictions` and `--method_name GT`.

Outputs under `--output_dir`: `gradient_test_config.json`,
`{method_name}_gradient_report.json`, and `{method_name}_summary.csv`
(one row per image × channel).

## Usage — FRC

```bash
compute-frc \
    --predictions predictions.npz \
    --ground_truth ground_truths.npz \
    --method_name inner_tiling \
    --ndim 2 \
    --step 32 \
    --output_dir results/frc
```

3-D volumes (`--ndim 3`) are scored per z-slice. `--step` is the seam interval in
pixels (e.g. `tile_size - overlap`); when given, dashed harmonic verticals `k/step`
are drawn on the curve plots.

Outputs under `--output_dir`: `{method_name}_frc_report.json`,
`{method_name}_summary.csv`, and `{method_name}_frc_curves_ch{c}.pdf` per channel.

Run either command with `--help` for the full list of parameters.

## Python API

```python
from pathlib import Path
from tilartmetrics.gradient_test import run_gradient_analysis_dataset
from tilartmetrics.utils import iter_npz_images, read_image_names

names = read_image_names("predictions.npz")          # keys = image names
images = iter_npz_images("predictions.npz", names, n_spatial=2)  # lazy (C, H, W)

report = run_gradient_analysis_dataset(
    images,
    tile_size=[64, 64],
    overlap=[32, 32],
    method_name="inner_tiling",
    save_dir=Path("./results"),
)
```

## Documentation

- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Command syntax cheat sheet
- **[AGENTS.md](AGENTS.md)**: Architecture & internals
