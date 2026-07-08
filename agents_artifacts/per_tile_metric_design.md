# Per-tile statistical metric for stitching artifacts — design document

**Purpose.** This document records the methodological design of the per-tile statistical metric for detecting stitching artifacts in tiled-inference predictions. It serves two purposes:

1. **Methods section source**: provides the conceptual framing, statistical justification, and design choices to be written up in the paper's methods section.
2. **Implementation guidelines**: provides the technical specification for updating the existing `Sliding_Tiling_Divergence` codebase to implement the new metric.

---

## 1. Conceptual framing

### 1.1. Problem statement

We aim to quantify stitching artifacts in tiled-inference predictions in a reference-free, single-prediction setting. The metric must operate on the final stitched prediction without requiring ground truth or a non-tiled baseline (both often infeasible for the gigapixel images that motivate tiling in the first place).

### 1.2. Core idea — two-sample hypothesis test

We treat artifact detection as a **two-sample testing problem**. Under a well-stitched prediction, the distribution of pixel gradients sampled at known seam locations should be statistically indistinguishable from the distribution of pixel gradients sampled from the surrounding image. Artifacts manifest as a statistically detectable shift between these two empirical distributions.

The seam locations are known analytically from the inference patching scheme, so the sampling regions are determined deterministically.

### 1.3. Conceptual ancestry

The construction is the direct analog, for tiled-inference seams, of the **JPEG-blockiness metrics** developed in the no-reference IQA literature:

- Wang, Sheikh & Bovik 2002 — canonical inter-block vs. intra-block pixel-difference construction.
- Pan et al. 2004 — ratio of inter-block to intra-block pixel differences.
- Liu & Heynderickx 2009 — local-feature-aware blockiness with explicit texture-masking handling.

The reason this literature is the right anchor: known artifact locations + reference-free statistical comparison between artifact-prone and artifact-free regions. We generalize this from pixel-difference statistics on fixed-size JPEG blocks to gradient-distribution statistics on inference tiles, and from heuristic ratio scores to a principled two-sample testing framework.

### 1.4. What this design replaces

The previous implementation (`Sliding_Tiling_Divergence`) computes a single global KL divergence between gradient distributions pooled across all seam locations and all tile interiors of an image. This has two known weaknesses:

- **Texture masking**: seams crossing textured regions get confounded with structural edges; global pooling washes out local effects.
- **No spatial localization**: a single scalar per image, no way to identify where artifacts occur.

The per-tile statistical test addresses both: each tile is evaluated against its own local context, yielding a per-tile test statistic and enabling spatial visualization.

---

## 2. Per-tile construction

### 2.1. Unit of analysis

The unit is **one tile in the stitched prediction**. For each tile T, we compute a single test statistic `T_tile` and an associated p-value `p_tile` that measure whether the tile's boundary gradients are statistically distinguishable from its locally-adjacent gradients.

### 2.2. Seam sample

For an interior tile with four boundaries (top, bottom, left, right), the seam sample comprises the across-seam directional gradients at all four boundaries:

- **Left/right (vertical) seams** at columns `X_L` and `X_R`, spanning rows in the tile's row range:
  - `g_h[y, X_L+1] − g_h[y, X_L]` for y in tile rows (across-seam horizontal gradient at the left seam).
  - Analogous for the right seam.
- **Top/bottom (horizontal) seams** at rows `Y_T` and `Y_B`, spanning columns in the tile's column range:
  - `g_v[Y_T+1, x] − g_v[Y_T, x]` for x in tile columns (across-seam vertical gradient at the top seam).
  - Analogous for the bottom seam.

The seam sample is the pooled set of all these gradients. For a tile of side S, the seam sample size is approximately `4S` for an interior tile, `3S` for an edge tile, `2S` for a corner tile.

### 2.3. Control sample

The control sample comprises gradients in **strips of width N immediately adjacent** to each seam, sampled on **both sides** (inside the tile and inside the neighbouring tile).

For each seam, take 2N adjacent strips (N on each side). Each strip contributes (per row, or per column, in the direction parallel to the seam) the gradient between two consecutive pixels within the strip. The control sample pools gradients from all 2N strips × all 4 seams of the tile.

Rationale for using both sides as control:
- Under H₀ (no artifact), a seam pixel transition should be indistinguishable from any other local pixel transition, regardless of which tile the transition lies in.
- Pooling both sides keeps the control as local as possible to each seam, preserving the locality benefit of the per-tile construction.
- A boundary between tile predictions is *itself* part of the stitched image; both sides' adjacent strips are legitimate samples of "the local image away from the seam".

**Parameter choice**: N = 3 to 8 pixels. Default N = 4. Robustness to N should be checked in an ablation. Larger N is acceptable in principle (more samples), but smaller N keeps the control more local.

### 2.4. Anisotropy / direction pooling

We pool horizontal-direction and vertical-direction gradients into the same sample, both for seam and control sets. This assumes the underlying gradient statistics are approximately isotropic, which holds for typical fluorescence microscopy data after PSF blurring.

**Required diagnostic**: before reporting results, run a quick anisotropy check on a few representative tiles: separately compute the horizontal and vertical gradient distributions for the control sample and confirm they are statistically indistinguishable. If not, direction pooling must be revisited.

### 2.5. Edge and corner tiles

Edge tiles (3 seams) and corner tiles (2 seams) are **kept** in the analysis. The two-sample test handles unequal sample sizes natively. Per-tile statistics for these will be noisier due to smaller sample size; this is an honest reflection of the available data.

No special weighting or normalization at the per-tile level.

---

## 3. Statistical test

### 3.1. Test choice — block permutation

We use a **block permutation test** rather than a closed-form analytical test (KS, etc.) for two reasons:

- **Dependence handling**: gradients within a tile are not independent — neighbouring pixels share content. Analytical KS p-values would be anti-conservative under this dependence. Block permutation, by permuting groups of spatially-adjacent gradients together, respects local spatial structure.
- **Flexible statistic choice**: permutation lets us pick whatever discrepancy statistic best captures the artifact signal, without re-deriving asymptotics for each choice.

### 3.2. Block permutation procedure

1. Concatenate the seam sample and the control sample into a single combined set, tagging the original labels.
2. Partition the combined set into spatially-coherent blocks of size `B` (default B = 3 pixels — match the local correlation length; tune if needed).
3. Compute the observed test statistic on the original labelling.
4. For `R` permutations (default R = 1000):
   - Randomly reassign the block labels (preserving the total number of seam-blocks and control-blocks).
   - Compute the test statistic on the permuted labelling.
5. p-value = fraction of permuted statistics ≥ the observed statistic.

### 3.3. Test statistic — KL divergence (current choice)

The current implementation uses KL divergence between histograms of the two samples, which we keep for backwards compatibility. The permutation framework handles the lack of analytical asymptotics for binned KL.

**Note on direction**: KL is asymmetric. We use `KL(seam || control)`, interpreted as "how surprising are the seam gradients under the control distribution". Document this direction explicitly in the methods section.

### 3.4. Alternative statistics (robustness checks for the supplementary)

The permutation framework supports easy swapping of the test statistic. Useful alternatives to report as robustness checks:

- **KS statistic**: `max_t |F_seam(t) − F_control(t)|`. Bin-free; standard non-parametric distributional comparison.
- **Mean absolute gradient ratio**: `mean(|seam gradients|) / mean(|control gradients|)`. Direct analog of Pan et al.'s 2004 inter/intra-block ratio. Highly interpretable.
- **Jensen–Shannon divergence**: symmetric variant of KL; bounded.
- **Wasserstein-1 / Earth Mover's Distance**: geometrically meaningful on the gradient-magnitude space.

Robustness checks should demonstrate that method rankings are stable across choices of statistic.

---

## 4. Aggregation and reporting

### 4.1. Per-image scalars

Each image yields a set of `T_tile` values (one per tile). Aggregate into per-image scalars:

- **Primary scalar — median `T_tile` across tiles**. Robust to outliers; natural for method comparison ("method X reduces median T by 50%").
- **Secondary scalar — fraction of rejected tiles at α = 0.05**. Interpretable as "fraction of tiles with detectable artifacts". Calibration is approximate due to within-tile pixel dependence; report with an explicit caveat. Despite this, the *relative* ordering between methods is robust.

Avoid combining per-tile p-values into a single per-image p-value (Fisher's method etc.). One step further into inferential territory than necessary for this paper.

### 4.2. Per-method scalars

Average per-image scalars across the test set. Report median (across tiles) → mean (across images), and fraction-rejected (across tiles) → mean (across images).

### 4.3. Figures

- **Per-tile heatmap**: spatial map of `T_tile` overlaid on the stitched prediction. Single image, multiple methods side-by-side. The most informative figure for the reader.
- **Boxplot of `T_tile` distribution per method**: across all tiles of all test images. Shows the full distribution of artifact scores, not just the median.

### 4.4. Null reference distribution (optional but recommended)

To calibrate "what does an artifact-free `T_tile` look like", compute `T_tile` on either:

- **Ground-truth images** (when available), using the same patching scheme.
- **Non-tiled predictions** on small enough images that tiling is unnecessary.

This gives a null reference distribution against which method `T_tile` distributions can be compared. Not essential to the metric definition, but strongly improves interpretability of reported numbers.

---

## 5. Open methods-section questions to resolve before submission

The following are *not* blockers for implementation, but should be resolved before the methods section is finalized:

1. **Direction of KL divergence** (`KL(seam || control)` vs. `KL(control || seam)`) — document the chosen direction and rationale.
2. **Block size B** — current default 3. May need tuning per dataset based on local correlation length; could be derived empirically from autocorrelation analysis.
3. **Anisotropy diagnostic results** — confirm direction pooling is justified for each dataset, or report results with separate horizontal/vertical pooling if not.
4. **Null reference distribution** — decide whether to include for the final paper. Strengthens interpretation.
5. **Failure mode of "uniformly smooth predictions"** — a model that smooths everything will produce low `T_tile` everywhere. This is an honest limitation of any seam-vs-interior comparison and should be acknowledged. Pair with a standard task metric (PSNR / MicroMS-SSIM) so the joint reporting shows that smoothing is not the way to score well.

---

## 6. References for the methods section

See `references_metrics.md`. Priority citations for the methods section specifically:

- Wang, Sheikh & Bovik 2002 — conceptual ancestor.
- Pan et al. 2004 — inter/intra-block ratio analog.
- Liu & Heynderickx 2009 — texture-masking justification for using local controls.
- Buglakova et al. 2025 — closest existing tiling-artifact metric; contrast with our per-tile reference-free construction.
- Demidenko 2004 — two-sample testing for image comparison.
- Liu et al. 2010 — gradient-domain IQA with distributional distances; closest existing precedent for distributional distance on image gradients.
