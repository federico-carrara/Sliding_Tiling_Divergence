# The Sliding-Tiling Gradient Permutation Test — Implementation Report

## 0. Purpose and one-line summary

The test asks, for each stitched prediction, whether **finite-difference
gradients that fall exactly on the tiling seams are systematically larger /
distributed differently than gradients in the immediate neighbourhood of those
seams**. If a tiled-inference method leaves stitching artifacts, the
across-seam gradients carry an excess discontinuity that a matched-control
two-sample permutation test can detect. The unit of testing is the individual
*kept region* ("tile"), and per-tile outcomes are aggregated to image and
method level.

Module map (`src/tilartmetrics/gradient_test/`):

| Stage | Module | Role |
|---|---|---|
| Seam geometry | `seams.py` | Closed-form seam positions from `(axis_size, tile_size, overlap)` |
| Tile enumeration | `tiles.py` | Kept regions + the seams each owns |
| Gradients | `gradient_analysis.py` | Per-axis finite differences |
| Sampling | `sampling.py` | Seam-line and control-strip 1-D samples per tile |
| Statistics | `statistics.py` | Two-sample discrepancy functions + registry |
| Permutation | `permutation.py` | Block-permutation null + p-value |
| Per-image scan | `per_tile.py` | Glue: gradients → tiles → sample → test |
| Aggregation | `aggregation.py` | Tile → image → method roll-up |
| Orchestration | `analysis.py` (+ `comparison.py`) | Public entry point over a set of predictions |

---

## 1. Seam geometry (`seams.py`)

Seams are derived **analytically** from the CAREamics `TiledPatching`
parameters — no runtime dependency on the patcher. Per spatial axis, with
`step = tile_size − overlap` and `M = overlap // 2`:

- **Single-tile fallback:** if `axis_size ≤ tile_size`, the axis emits one tile
  and *no* seams (warn-and-skip).
- **Otherwise:** number of tiles `N = ceil((axis_size − overlap) / step)`, and
  seam pixel positions are

  ```
  seam_k = k · step + M,   k = 1, …, N − 1.
  ```

  A seam is the first pixel of kept tile `k` in the stitched image, i.e. the
  boundary between two adjacent kept regions (`seams.py:57-60`).

`assert_shape_consistent` enforces the **clean-tiling regime**
(`axis_size == (N−1)·step + tile_size`); irregular last-tile tilings are
rejected because the metric is only meaningful when these parameters actually
produced the prediction (`seams.py:63-102`).

**Gradient-index convention** (used everywhere downstream): finite differences
are `grad[p] = img[p+1] − img[p]`, so the step *into* pixel `j` lives at
gradient index `j − 1`. A seam at pixel `j` therefore maps to gradient index
`j − 1` (`seams.py:105-123`; `Seam.grad_idx = pixel − 1`).

---

## 2. Tiles = kept regions (`tiles.py`)

A **tile** here is *not* an inference patch — it is one kept region of the
stitched output: the slab between two consecutive seams along each axis (or
between an image edge and the first/last seam).

- Per axis, boundaries are `[0, seam_1, …, seam_{N-1}, axis_size]`; the
  Cartesian product over axes enumerates all kept regions in C-order
  (`tiles.py:122-153`).
- Each tile **owns** the seams that bound it: for axis `a`, if it is not the
  first region along `a` it owns the seam on its low face, and if not the last
  it owns the seam on its high face. So an interior 2-D tile owns 4 seams (2 per
  axis), an edge tile 3, a corner tile 2.
- Each `Seam` records `(axis, pixel, grad_idx)` (`tiles.py:24-40`).

---

## 3. Gradients (`gradient_analysis.py`)

For each image, one **first-order forward finite-difference array per spatial
axis** is computed (`gradient_analysis.py:18-74`):

- 2-D: `g_y` shape `(H−1, W)`, `g_x` shape `(H, W−1)`.
- 3-D: `g_z, g_y, g_x` analogously.

Only the component **perpendicular to a seam** is ever used for that seam (see
§4). There is no gradient-magnitude / sum / root-sum-of-squares step — this is
deliberate: a stitching seam is a jump *across* the seam, so the across-seam
directional derivative is the signal; the parallel component would only add
image-structure noise.

---

## 4. Per-tile sampling: seam line vs. matched control strips (`sampling.py`)

This is the statistical core. For each seam owned by a tile (axis `a`, gradient
index `g_idx`):

- **Seam sample** — one 1-D array: take gradient array `g_a`, fix the
  across-seam index at `g_idx`, and slice the full tile range along **every
  parallel axis**, then ravel. This is the strip of across-seam gradients lying
  *on* the seam line (`sampling.py:148-153`; `_slice_along` at
  `sampling.py:74-111`).
- **Control sample** — `2·strip_width` separate 1-D arrays: parallel lines at
  across-seam offsets `{−N,…,−1,+1,…,+N}` (offset 0 is the seam itself,
  skipped), each sliced to the *same* parallel range as the seam line
  (`sampling.py:155-160`).

Key design points:

1. **The control "strip of width N" is never a 2-D block.** The across-seam
   width dimension is decomposed into `2N` independent 1-D lines, each
   geometrically identical to the seam line. Blocking (§5) then only ever
   happens along the seam-parallel direction.
2. **Slices are kept separate, not pre-concatenated** (`sampling.py:11-17`), so
   the permutation engine can block each slice independently and blocks never
   span slice boundaries — otherwise the test would fabricate coherence between
   unrelated seams.
3. **Multi-axis tiles pool per-component but component-matched:** a corner
   tile's vertical seam contributes `g_x` lines and its horizontal seam
   contributes `g_y` lines, both pooled into the one seam sample; the respective
   controls pool into the one control sample. Seam and control pools carry the
   same axis mixture, so the two-sample comparison stays fair.
4. **Pre-condition** (validated upstream in `per_tile.py`): along every seam
   axis, `step ≥ 2·strip_width + 2`, guaranteeing all `2N` control offsets land
   inside the gradient array (`per_tile.py:23-49`).

### 4.1 Control offsets do not re-include the seam

The seam discontinuity is contained entirely in the single gradient value at
`g_idx` (`g[j−1] = img[j] − img[j−1]`, the step between tile `k−1` and tile
`k`). The control offsets `g_idx ± 1, …, g_idx ± N` are all **within-region**
finite differences (e.g. `offset=+1` → `img[j+1] − img[j]`, both in tile `k`),
so they never span the seam. `offset == 0` — the only gradient that straddles
the boundary — is explicitly skipped. Endpoint image-pixel sharing between the
seam gradient and the `±1` controls is intrinsic to overlapping finite
differences and does not leak the discontinuity into the control.

---

## 5. Block-permutation two-sample test (`permutation.py`)

Given the per-tile seam slices and control slices:

1. **Blocking** — each slice (seam line, and each of the `2N` control lines) is
   cut into contiguous blocks of length `B = block_size` along its raveled
   order; trailing partial blocks are kept to preserve all data; blocks never
   cross slice boundaries (`_split_into_blocks`, `permutation.py:35-61`).
   Contiguous blocks absorb **along-seam spatial autocorrelation** so the
   exchangeability assumption is defensible.
2. **Labels** — the observed split is "first `n_seam_blocks` blocks = seam, rest
   = control." A permutation reshuffles block labels while keeping the
   seam/control block *counts* fixed (`_build_permutations`,
   `permutation.py:64-84`). Because every block has the same 1-D geometry, seam
   and control blocks are exchangeable under the null.
3. **Statistic** — `T_obs` on the true split, `T_null` (length `R`) over `R`
   permutations.
4. **P-value (Phipson–Smyth)** — right-tailed, guarded against a hard zero:

   ```
   p = (1 + #{T_null ≥ T_obs}) / (1 + R).
   ```

   (`permutation.py:355`)
5. **Skip rule** — if either side has zero blocks, returns `(nan, nan, empty)`
   and the tile is recorded as skipped (`permutation.py:335-336`).

**Vectorized fast paths** (same numerical result, faster): `binned` for KL/JS
(per-block histograms on shared per-tile edges, summed under each permutation),
`abs_ratio` for the mean-abs-ratio (per-block absolute sums and lengths).
Everything else uses a Python loop over permutations (`_scalar_path`)
(`permutation.py:343-353`).

---

## 6. Two-sample statistics (`statistics.py`)

Registry of `(seam, control) → scalar` discrepancies (`statistics.py:237-245`):

| Name | Definition | Path |
|---|---|---|
| `js` **(default)** | Jensen–Shannon divergence, natural log, histograms | binned |
| `kl` | `KL(seam ‖ control)` on histograms | binned |
| `ks` | Two-sample Kolmogorov–Smirnov `D` | scalar fallback¹ |
| `wasserstein` | 1-D Wasserstein-1 (scipy) | scalar |
| `mean_abs_ratio` | `mean(|seam|)/mean(|control|)`, Pan et al. (2004) | abs_ratio |

For KL/JS, bin edges are **joint per-tile** — built from
`concat(seam, control)` and reused for the observed statistic and every
permutation in that tile, preserving comparability (`statistics.py:25-46`;
`permutation.py:122-123`). Default bin count 32. `EPS = 1e-12` guards all
logs/divisions.

¹ `ks` carries a `"ks"` `vec_kind` label but the permutation engine has no
dedicated KS branch, so it currently runs through the scalar Python-loop path.

> **Note (see TODO.md → Cleanups):** the KL/JS/mean-abs-ratio math is
> duplicated between these reference callables and the vectorized paths in
> `permutation.py`; for those three, `stat_spec.fn` is never actually called.
> Only `wasserstein` and `ks` route through `fn`. The duplication is a
> speed-for-clarity trade and should be pinned with a parity test.

---

## 7. Aggregation (`aggregation.py`)

- **Per tile** (`TileResult`): `coord, n_seams, T_obs, p, n_seam_samples,
  n_control_samples`.
- **Per image** (`ImageReport`): over tiles with valid (non-NaN) results —
  `median_T` = median of `T_obs`, `frac_rejected` = fraction with `p < alpha`.
  Skipped tiles are excluded from both (`aggregation.py:99-134`).
- **Per method** (`MethodReport`): `mean_median_T` and `mean_frac_rejected` =
  means of the valid per-image scalars (`aggregation.py:137-166`).

---

## 8. Orchestration and control flow (`per_tile.py`, `analysis.py`)

Per single-channel image (`per_tile.py:52-164`):

1. Validate ndim (2-D or 3-D) and `step ≥ 2N+2`.
2. Compute per-axis gradients; enumerate tiles.
3. For each tile: **if `n_seams < 2`, skip** (record NaN) — degenerate tiles
   with 0–1 seams are excluded (`per_tile.py:128-139`); otherwise sample and run
   the permutation test.
4. Aggregate to `ImageReport`.

Public entry point `run_gradient_analysis` (`analysis.py:22-162`) takes
channel-first predictions (`(N,C,H,W)` or `(N,C,D,H,W)`), one `channel`,
per-axis `tile_size`/`overlap`, loops images, and returns a `MethodReport`
(optionally pickled). A single shared `np.random.default_rng(random_seed)`
drives all permutations for reproducibility.

**Default configuration** (the numbers to cite): `statistic="js"`,
`strip_width=4` (⇒ 8 control lines per seam), `block_size=3`,
`n_permutations=1000`, `alpha=0.05`, `num_bins_per_tile=32`, `random_seed=0`. In
3-D, v1 **always pools z with xy**; `pool_z_with_xy=False` is reserved for a
future revision and currently warns (`analysis.py:115-121`).

---

## 9. Compact methods-paragraph skeleton

> For each stitched prediction we located the tiling seams analytically from the
> `TiledPatching` parameters (`step = tile_size − overlap`; seams at
> `k·step + overlap//2`). We partitioned the output into kept regions and, for
> every seam owned by a region, extracted the across-seam first-difference
> gradients lying on the seam line together with `2N` matched control lines
> drawn parallel to the seam at across-seam offsets `±1,…,±N` (`N = 4`) over the
> same span. We tested the seam sample against the pooled control sample with a
> block-permutation two-sample test: each 1-D line was split into contiguous
> blocks of length `B = 3` to preserve along-seam autocorrelation, block labels
> were permuted `R = 1000` times, and significance was assessed by a
> right-tailed Jensen–Shannon divergence statistic with a Phipson–Smyth p-value
> `(1 + #{T_null ≥ T_obs})/(1 + R)`. Per image we report the median
> seam-vs-control divergence and the fraction of regions rejecting at
> `α = 0.05`, averaged across the test set per method.
