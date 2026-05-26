# AGENTS.md

Guidance for AI coding agents working in this repository.

## Test / verification environment

A dedicated conda environment is pre-built. **Use this env for all tests, verifications, and CLI smoke runs** — do not waste tokens hunting for one that has the right deps:

```
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python
```

Created via `conda create -n sliding_tiling_env python=3.12 -y` then `pip install -e .` from the repo root, so the package is editable. Imports work without `PYTHONPATH=src`.

Quick commands:

```bash
# Run all verification scripts (geometry, null calibration, artifact injection)
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_geometry.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_null_calibration.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_artifact_injection.py

# CLI smoke
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python -m analysis_pipeline.cli.analyze --help
```

If a dep is missing, `pip install <pkg>` inside the env — don't switch to another env.

## What this repo is

A small Python library for **quantifying tiling artifacts** in stitched large-scale images produced by tiled inference. As of the per-tile rewrite, it tests **each kept region of the TiledPatching grid** against its locally-adjacent gradients with a block permutation test, yielding a per-tile statistic `T_tile` and p-value `p_tile`. Two per-image scalars are reported: `median(T_tile)` and `frac_rejected` at α=0.05.

Reference-free: only the stitched prediction and the TiledPatching `(tile_size, overlap)` are needed. Conceptual anchor: the JPEG-blockiness IQA literature (Wang/Sheikh/Bovik 2002; Pan et al. 2004; Liu & Heynderickx 2009). Full design lives at [agents_artifacts/per_tile_metric_design.md](agents_artifacts/per_tile_metric_design.md).

Entry points:

- CLI: `analyze-experiment` → [src/analysis_pipeline/cli/analyze.py](src/analysis_pipeline/cli/analyze.py)
- Python API: `run_gradient_analysis_multi` in [src/analysis_pipeline/gradient_test/analysis.py](src/analysis_pipeline/gradient_test/analysis.py)

## Layout

```
src/analysis_pipeline/
├── cli/analyze.py            # argparse CLI; per-tile flags only
├── config/settings.py        # PerTileConfig + AnalysisConfig (pydantic)
├── gradient_test/            # per-tile permutation hypothesis test (this metric)
│   ├── seams.py              # closed-form seam positions from TiledPatching
│   ├── tiles.py              # kept-region enumeration + owned-seam list
│   ├── sampling.py           # per-tile seam_sample / control_sample
│   ├── statistics.py         # kl / js / ks / wasserstein / mean_abs_ratio
│   ├── permutation.py        # vectorized block-permutation engine
│   ├── aggregation.py        # TileResult / ImageReport / MethodReport dataclasses
│   ├── per_tile.py           # per-image orchestrator (one slice in, one report out)
│   ├── analysis.py           # multi-method orchestrator (legacy entry name)
│   └── gradient_analysis.py  # compute_gradients_{2d,3d} helpers only
├── legacy/                   # quarantined — pre-rewrite helpers, kept for v2 reuse
│   ├── metrics.py            # legacy KL/Wiener helpers
│   └── plotting.py           # legacy histogram/KL-heatmap figures
└── utils/
    ├── array_utils.py        # ensure_4d (channel-first)
    └── file_utils.py         # load_prediction (.pkl / .tiff)
tests/
├── test_geometry.py          # tile enumeration + owned-seam classification
├── test_null_calibration.py  # flat-field null check (frac_rejected ≈ α)
└── test_artifact_injection.py # synthetic seam shift → tiles near seam reject
```

## How the pipeline works

### 1. Geometry (no detection)

The TiledPatching geometry is supplied by the user (`--tile_size 64,64 --overlap 32,32` etc.); seams are derived analytically in [src/analysis_pipeline/gradient_test/seams.py](src/analysis_pipeline/gradient_test/seams.py) — closed form lifted from `careamics/dataset/patching/tiled_patching.py::_compute_1d_coords`:

- `step = tile_size - overlap`, `M = overlap // 2`
- `N = ceil((axis_size - overlap) / step)` tiles per axis
- Seams at `{k * step + M : k = 1, …, N - 1}` (interior only)

Kept-region tiles span the slabs between consecutive seams; the first and last regions include the image boundary. Per-tile owned seams: 4 (2D interior) / 3 (2D edge) / 2 (2D corner); 6 / 4-5 / 3 in 3D.

### 2. Per-tile two-sample test

For each owned seam in a tile:

- **Seam sample**: the 1-D across-seam gradient slice *on* the seam, restricted to the tile's parallel range.
- **Control sample**: ``2 * strip_width`` strips at gradient-index offsets `{−N, …, −1, +1, …, +N}` from the seam, same parallel range. Strips span both adjacent kept regions — both sides of the seam contribute.

All owned-seam slices and all control strips pool into one `seam_sample` and one `control_sample` per tile, regardless of seam axis (isotropy is assumed; ablations are out of scope for v1).

**Geometric constraint:** `step ≥ 2*strip_width + 2` so opposite-side strips can't overlap the opposite seam. Enforced at config validation and again in `per_image_tile_scan`.

### 3. Block permutation

Each per-seam / per-strip slice is partitioned into blocks of size `B` (default 3). Blocks never span slice boundaries — they only carry intra-slice spatial coherence along the parallel axis. The combined block list is permuted `R` times (default 1000) preserving seam/control block counts; `T_null` is recomputed each time.

`p = (1 + #{T_null ≥ T_obs}) / (1 + R)` (Phipson–Smyth — avoids a hard zero).

Fast paths:
- Binned (KL, JS): per-block histogram contributions are pre-computed once on per-tile joint bin edges, then summed under each permutation in one vectorized NumPy call.
- `mean_abs_ratio`: per-block abs sum + length pre-computed; same idea.
- KS / Wasserstein fall back to a Python loop calling the registered statistic.

### 4. Aggregation

Per image: `median(T_tile)` and `frac_rejected = mean(p_tile < α)`. Per method: mean of both across images. NaN tiles (fewer than 2 owned seams) excluded.

The orchestrator returns a `MultiMethodReport` dataclass and, if `save_dir` is set, pickles it to `save_dir/per_tile_report.pkl`. No CSVs / npy / summary text files are emitted — that's v2 work.

## Conventions

- Arrays are **channel-first**: 2-D `(N, C, H, W)`, 3-D `(N, C, D, H, W)`. `ensure_4d` inserts the channel axis at position 1 for 3-D inputs. Dimensionality is inferred from `predictions.ndim` (4 → 2-D, 5 → 3-D).
- `tile_size` / `overlap` may be a flat per-axis list (`[64, 64]`) applied to every method, or a per-method nested list (`[[64, 64], [32, 32]]`) — broadcast in `_broadcast_per_method_spec` ([gradient_test/analysis.py](src/analysis_pipeline/gradient_test/analysis.py)).
- 3-D z-direction is pooled with x/y by default. `pool_z_with_xy=False` is reserved for v2 and currently emits a warning while still pooling.
- Single channel only per run; the legacy `--channel 2` magic-iterate-over-channels behaviour is removed. Call once per channel if you need multiple.

## Gotchas

- **Calibration under residual dependence** is the canary. If `tests/test_null_calibration.py` ever fails (`frac_rejected` significantly above α on a flat field), raise `block_size` from 3 to 5–7 before chasing anything else.
- **Strip width vs. step**: `tile_size - overlap` must be ≥ `2*strip_width + 2`. Lower the strip width if you want overlap-heavy tilings.
- **Dilution at low bias**: each interior tile pools across 4 (2-D) or 6 (3-D) owned seams; an artifact present on only one of them produces a signal diluted by ~3-5×. `tests/test_artifact_injection.py` works with a +2.0 bias for this reason.
- **`legacy/plotting.py` and `legacy/metrics.py` are orphaned**, not deleted. They retain helpers (`plot_multiple_hist`, `compute_kl_matrix`, `wiener_entropy`, …) that may be reused when per-tile heatmaps and boxplots land in v2. No v1 code imports them; keep them quarantined.
- **Pydantic 2 is required** (not v1 syntax). `config/settings.py` uses `model_validator(mode="after")`.

## Running it

```bash
pip install -e .

analyze-experiment \
  --model_name microsplit \
  --dataset MyDataset \
  --predictions "pred1.tiff,pred2.tiff" \
  --method_names "OG,SW" \
  --save_dir ./results \
  --tile_size 64,64 \
  --overlap 32,32 \
  --statistic kl \
  --n_permutations 1000 \
  --strip_width 4 \
  --block_size 3 \
  --channel 0
```

Output: `./results/per_tile_report.pkl` with a `MultiMethodReport`, plus a console-printed summary table.

## Verification scripts

`tests/` has three smoke scripts; none are wired into pytest yet. Run them directly with the env above:

- `test_geometry.py` — tile counts, owned-seam classification (2-D and 3-D).
- `test_null_calibration.py` — `frac_rejected ≈ α` on a flat Gaussian field.
- `test_artifact_injection.py` — injected +2.0 seam shift detected by the right tiles, not by far ones.

Add new tests next to these. Each script uses bare `assert` + exit-code, no pytest needed.
