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
# Run all verification scripts (per-tile + FRC)
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_geometry.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_null_calibration.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_artifact_injection.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_calibration.py
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_frc.py

# Console CLIs (general, one method per invocation over a single .npz)
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python -m tilartmetrics.cli.analyze --help
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python -m tilartmetrics.cli.frc_analyze --help

# Dataset-driver smoke (experiment sweeps)
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python scripts/run_gradient_test_on_dataset.py --help
/localscratch/miniforge3/envs/sliding_tiling_env/bin/python scripts/run_frc_analysis_on_dataset.py --help
```

If a dep is missing, `pip install <pkg>` inside the env — don't switch to another env.

## What this repo is

A small Python library for **quantifying tiling artifacts** in stitched large-scale images produced by tiled inference. As of the per-tile rewrite, it tests **each kept region of the TiledPatching grid** against its locally-adjacent gradients with a block permutation test, yielding a per-tile statistic `T_tile` and p-value `p_tile`. Two per-image scalars are reported: `median(T_tile)` and `frac_rejected` at α=0.05.

Reference-free: only the stitched prediction and the TiledPatching `(tile_size, overlap)` are needed. Conceptual anchor: the JPEG-blockiness IQA literature (Wang/Sheikh/Bovik 2002; Pan et al. 2004; Liu & Heynderickx 2009). Full design lives at [agents_artifacts/per_tile_metric_design.md](agents_artifacts/per_tile_metric_design.md).

Entry points:

Two metrics ship today. There are two ways to run each: the general **console CLIs**
(`analyze-experiment` / `frc-experiment`), which run **one method per invocation** over a
single `.npz` archive keyed by image name; and the dataset-specific **argparse drivers in
`scripts/`** (see the `scripts/` + `hpc/` layout below and "Running on a dataset"), which sweep a
whole experiment. Both are built on the same streaming Python APIs.

- **Gradient test** (reference-free, per-tile permutation hypothesis test):
  - CLI (general): `analyze-experiment` → [cli/analyze.py](src/tilartmetrics/cli/analyze.py). One method, one prediction `.npz`.
  - Dataset driver (experiment sweeps): [scripts/run_gradient_test_on_dataset.py](scripts/run_gradient_test_on_dataset.py) → `run_gradient_analysis_dataset`.
  - Python API: `run_gradient_analysis` (single image) / `run_gradient_analysis_dataset` (lazy dataset) in [src/tilartmetrics/gradient_test/analysis.py](src/tilartmetrics/gradient_test/analysis.py).
- **FRC** (reference-based, 2-D Fourier Ring Correlation vs. ground truth):
  - CLI (general): `frc-experiment` → [cli/frc_analyze.py](src/tilartmetrics/cli/frc_analyze.py). One method; prediction + ground-truth `.npz` keyed by the same image names; 3-D scored per z-slice.
  - Dataset driver (experiment sweeps): [scripts/run_frc_analysis_on_dataset.py](scripts/run_frc_analysis_on_dataset.py) → `run_frc_analysis_dataset`. 3-D volumes are scored per z-slice.
  - Python API: `run_frc_analysis` (stacked `(N,C,H,W)`) / `run_frc_analysis_dataset` (lazy iterator of `(id, pred, gt)`) in [src/tilartmetrics/frc/analysis.py](src/tilartmetrics/frc/analysis.py). Spec: [agents_artifacts/FRC_metric.md](agents_artifacts/FRC_metric.md).

## Layout

```
src/tilartmetrics/
├── cli/
│   ├── analyze.py            # per-tile CLI (primary API)
│   └── frc_analyze.py        # FRC CLI (primary API)
├── config/
│   └── gradient.py           # GradientTestConfig (per-tile permutation params)
├── gradient_test/            # per-tile permutation hypothesis test (reference-free)
│   ├── seams.py              # closed-form seam positions from TiledPatching
│   ├── tiles.py              # kept-region enumeration + owned-seam list
│   ├── sampling.py           # per-tile seam_sample / control_sample
│   ├── statistics.py         # kl / js / ks / wasserstein / mean_abs_ratio
│   ├── permutation.py        # vectorized block-permutation engine
│   ├── aggregation.py        # TileResult / ImageReport / MethodReport dataclasses
│   ├── per_tile.py           # per-image orchestrator (one slice in, one report out)
│   ├── analysis.py           # dataset orchestrator (primary API)
│   └── gradient_analysis.py  # compute_gradients_{2d,3d} helpers only
├── frc/                      # Fourier Ring Correlation vs. GT (reference-based)
│   ├── windowing.py          # 2-D Hamming window — mandatory pre-FFT taper
│   ├── frc.py                # per-image FRC curve (FFT + radial bincount)
│   ├── aggregation.py        # FRCChannelResult / FRCImageReport / FRCMethodReport
│   ├── analysis.py           # run_frc_analysis (stacked) + run_frc_analysis_dataset (lazy iterator)
│   ├── plotting.py           # mean curve + 95% CI bands + harmonic verticals
│   ├── reduction.py          # frc_to_scalar — stub, raises NotImplementedError (§3.7)
│   └── fsc.py                # 3-D FSC — stub only (§5)
└── utils/
    ├── array_utils.py        # ensure_4d (channel-first)
    └── file_utils.py         # load_prediction (.pkl / .tiff)
tests/
├── test_geometry.py          # tile enumeration + owned-seam classification
├── test_null_calibration.py  # flat-field null check (frac_rejected ≈ α)
├── test_artifact_injection.py # synthetic seam shift → tiles near seam reject
├── test_calibration.py       # block-size calibration smoke
└── test_frc.py               # self-FRC = 1, indep-noise ≈ 0, harmonic dip, aggregation
scripts/                      # dataset-sweep drivers (argparse, SLURM-friendly)
├── run_gradient_test_on_dataset.py  # gradient test over a whole dataset, per method
├── run_frc_analysis_on_dataset.py   # FRC over a whole dataset, per method (2-D; 3-D scored per z-slice)
├── run_anisotropy_diagnostic.py     # per-axis anisotropy diagnostic (3-D)
└── flateness_gradient_test_analyses.py  # flatness-stratified rejection analysis
hpc/                          # SLURM array wrappers around the dataset drivers
├── run_gradient_test.sbatch
└── run_frc_analysis.sbatch
```

## How the pipeline works

### 1. Geometry (no detection)

The TiledPatching geometry is supplied by the user (`--tile_size 64,64 --overlap 32,32` etc.); seams are derived analytically in [src/tilartmetrics/gradient_test/seams.py](src/tilartmetrics/gradient_test/seams.py) — closed form lifted from `careamics/dataset/patching/tiled_patching.py::_compute_1d_coords`:

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

Reports are pydantic models with NaN-safe JSON `save`/`load` (`model_dump_json`), **not** pickle. The orchestrator returns a `MethodReport` and, if `save_dir` is set, writes `save_dir/{method_name}_per_tile_report.json`. The `scripts/` dataset drivers additionally emit `{method}_gradient_report.json` + a `summary.csv` (`to_records()`).

## How FRC works

2-D only. Per (prediction, GT) pair: Hamming window → `fft2` → `fftshift` → bin Fourier-space pixels by integer radius from DC → for each ring `r`, compute

```
FRC(r) = Σ F_P(k) · conj(F_G(k))  /  sqrt( Σ |F_P(k)|² · Σ |F_G(k)|² )
```

The ring sums are vectorised with three `np.bincount` calls over a precomputed radial-index map; the cross-sum imaginary part cancels by Hermitian symmetry for real inputs and is dropped. Frequency-bin centres are `r / min(H, W)` in cycles/pixel; `r_max = min(H, W) // 2`.

Per method, images are aggregated per-frequency-bin with `np.nanmean` and a 95% CI of `± 1.96 · std / sqrt(n)`. The Fisher-z transform is left as a deferred optional refinement (see `agents_artifacts/FRC_metric.md` §3.5).

Stitching artifacts show up as dips at the seam harmonics `k/step`, where `step` is the method's seam interval in pixels (`tile_size - overlap` for inner tiling; the sliding stride for SWITi, whose seams are dimmer but still periodic) and `k = 1, 2, …, step//2` is the harmonic index — a periodic train of sharp seams puts energy at every integer multiple of the fundamental `1/step`, up to Nyquist. The headline plot in [src/tilartmetrics/frc/plotting.py](src/tilartmetrics/frc/plotting.py) draws dashed verticals at those locations, per method, when the driver's `--step` is provided (one entry per `--methods`, `none` for a seam-free method).

**Resolution readout.** `reduction.frc_resolution` returns the frequency where the curve first falls below the conventional `1/7 ≈ 0.143` threshold (DC bin skipped, NaN rings ignored, crossing refined by linear interpolation); `frc_resolution_period` is its reciprocal in pixels. Higher crossing frequency = finer detail still faithful to GT. `NaN` means either "never crosses" (correlated to the band edge) or "already below at the first non-DC bin" (no correlated band at all — e.g. pure noise). Caveat: the 1/7 criterion is derived for FRC between two *independent equally-noisy half-datasets*; here we correlate prediction vs. a clean GT (a fidelity curve), so treat the crossing as a **conventional cutoff for ranking methods on the same data**, not a physical instrument resolution.

The seam-artifact scalar `frc_to_scalar` and the 3-D `fsc` module remain intentional stubs (see [src/tilartmetrics/frc/reduction.py](src/tilartmetrics/frc/reduction.py) and [src/tilartmetrics/frc/fsc.py](src/tilartmetrics/frc/fsc.py)). Reports save as NaN-safe JSON (`FRCMethodReport.save`/`.load`), not pickle: `run_frc_analysis` writes `save_dir/{method_name}_frc_report.json`, and the `scripts/run_frc_analysis_on_dataset.py` driver adds a `summary.csv` (`to_records()`: per `(image, channel)` `frc_mean_excl_dc`, `frc_res_cyc_per_px`, `frc_res_period_px`, `n_bins`).

## Conventions

- Arrays are **channel-first**: 2-D `(N, C, H, W)`, 3-D `(N, C, D, H, W)`. `ensure_4d` inserts the channel axis at position 1 for 3-D inputs. Dimensionality is inferred from `predictions.ndim` (4 → 2-D, 5 → 3-D).
- `tile_size` / `overlap` are flat per-axis lists (`[64, 64]`), one entry per spatial axis.
- 3-D z-direction is pooled with x/y by default. `pool_z_with_xy=False` is reserved for v2 and currently emits a warning while still pooling.
- Single channel only per run; the legacy `--channel 2` magic-iterate-over-channels behaviour is removed. Call once per channel if you need multiple.

## Gotchas

- **Calibration under residual dependence** is the canary. If `tests/test_null_calibration.py` ever fails (`frac_rejected` significantly above α on a flat field), raise `block_size` from 3 to 5–7 before chasing anything else.
- **Strip width vs. step**: `tile_size - overlap` must be ≥ `2*strip_width + 2`. Lower the strip width if you want overlap-heavy tilings.
- **Dilution at low bias**: each interior tile pools across 4 (2-D) or 6 (3-D) owned seams; an artifact present on only one of them produces a signal diluted by ~3-5×. `tests/test_artifact_injection.py` works with a +2.0 bias for this reason.
- **`legacy/` was deleted** (pre-rewrite `plotting.py`/`metrics.py` helpers: `plot_multiple_hist`, `compute_kl_matrix`, `wiener_entropy`, …). Recover them from git history if per-tile heatmaps or boxplots need them in v2.
- **Pydantic 2 is required** (not v1 syntax). The `config/` modules use `model_validator(mode="after")`.

## Running it

For a single method, use the general console CLIs (`analyze-experiment` / `frc-experiment`,
one `.npz` per invocation — see Entry points). For whole-experiment sweeps, the
**dataset-sweep drivers in `scripts/`** loop over several methods + GT. Both drivers share a
data layout: per method a single
`{results_root}/{dataset}/{predictions_subdir}/{method_subdir}/predictions.npz` (keys are image
names, arrays squeeze to channel-first `(C,H,W)` / `(C,D,H,W)`) plus matching ground truths at
`{data_root}/{dataset}/targets/test/{image}.tif`. Method→subdir map:
`inner_tiling → inner_tiling`, `SWITi → sw_inner_tiling`.

```bash
pip install -e .

# Gradient test over a dataset, per method (GT tested as a seam-free null unless --no_gt)
python scripts/run_gradient_test_on_dataset.py \
  --dataset PaviaATN --predictions_subdir predictions_MMSE64 \
  --methods inner_tiling SWITi --tile_size 64 64 --overlap 32 32 \
  --statistic js --strip_width 3 --output_root ./results/gradient_test

# FRC over a dataset, per method (2-D; pass --ndim 3 to score a 3-D volume per z-slice)
python scripts/run_frc_analysis_on_dataset.py \
  --dataset PaviaATN --predictions_subdir predictions_MMSE64 \
  --methods inner_tiling SWITi --ndim 2 --output_root ./results/frc
```

Each driver writes, under `--output_root`:
- one `{method}_gradient_report.json` / `{method}_frc_report.json` per method (a pydantic
  `MethodReport` / `FRCMethodReport`, reload via `.load(...)`; NaN-safe JSON, **not** pickle);
- a `summary.csv` with one row per `(method, image, channel)` from `to_records()`;
- the gradient driver also dumps `gradient_test_config.json` (`GradientTestConfig`). The FRC driver
  takes no config object (nothing config-worthy to persist).

For SLURM, the `hpc/*.sbatch` files wrap these drivers as `--array` jobs over a dataset list.

## Verification scripts

`tests/` has three smoke scripts; none are wired into pytest yet. Run them directly with the env above:

- `test_geometry.py` — tile counts, owned-seam classification (2-D and 3-D).
- `test_null_calibration.py` — `frac_rejected ≈ α` on a flat Gaussian field.
- `test_artifact_injection.py` — injected +2.0 seam shift detected by the right tiles, not by far ones.

Add new tests next to these. Each script uses bare `assert` + exit-code, no pytest needed.
