# Metrics for stitching-artifact evaluation — implementation handover

**Purpose.** This document is the implementation specification for the metric portfolio used to evaluate stitching artifacts in tiled-inference predictions, for the Bioimaging Computing (BIC) ECCV 2026 workshop submission. It consolidates the design decisions reached during methods-section planning and is intended as context for a coding agent implementing the metrics.

**Scope.** Reference-free and reference-based metrics that, together, characterize stitching artifacts in 2D predictions from tiled inference. Comparison target: inner tiling (deterministic seams) vs. SWiTi (no fixed seams).

**Out of scope** (placeholders only): 3D / volumetric extensions, finalization of the FRC curve-to-scalar reduction, and the discarded alternatives listed at the end of this document.

---

## 1. Metric portfolio

Four metrics, complementary roles:

| Metric | Type | Spatial localization | Frequency-resolved | Role |
|---|---|---|---|---|
| Per-tile hypothesis test | Reference-free | Yes (per-tile heatmap) | No | Spatial story; detects where artifacts occur |
| Fourier Ring Correlation vs. GT | Reference-based | No | Yes | Spectral story; harmonic dips at seam frequencies |
| PSNR | Reference-based | No | No | Overall reconstruction quality |
| MicroMS-SSIM | Reference-based | No | No | Perceptual reconstruction quality |

All four are reported together for every method × dataset combination.

---

## 2. Per-tile hypothesis test (reference-free)

Already designed and partially implemented; see `per_tile_metric_design.md` for the full specification. No revisions to that design in this round.

Status: implementation work is tracked separately from this document. The handover for that metric is `per_tile_metric_design.md`.

---

## 3. Fourier Ring Correlation against ground truth

### 3.1. What it computes

Given a prediction `P` and matching ground-truth image `G` of the same field of view, FRC measures the normalized cross-correlation between their 2D Fourier transforms as a function of radial spatial frequency `r`:

```
FRC(r) = sum_{k in ring(r)} F_P(k) * conj(F_G(k))
       / sqrt( sum_{k in ring(r)} |F_P(k)|^2 * sum_{k in ring(r)} |F_G(k)|^2 )
```

Output: a 1D curve, x-axis = spatial frequency (cycles/pixel, from 0 to Nyquist = 0.5), y-axis = correlation in `[-1, 1]`.

For our use case, stitching artifacts should manifest as **dips in the FRC curve at frequencies `k/S` for `k = 1, 2, 3, ...`**, where `S` is the inner tile size in pixels. SWiTi predictions, lacking a fixed seam grid, should show a smooth curve with no such dips.

### 3.2. Variant to implement

**Standard two-image FRC**, prediction vs. matching GT. No one-image FRC variants (see §6 for why).

### 3.3. Preprocessing — windowing

**Required:** apply a 2D Hamming window (or equivalent smooth taper — Hann, Tukey all acceptable) to both prediction and GT before FFT.

Why: a finite-size image viewed by the FFT is treated as one period of an infinite periodic signal. Real images do not wrap around cleanly, so the FFT sees a sharp discontinuity at the image boundary and produces spurious high-frequency content along the `u` and `v` axes of the spectrum. This cross-shaped leakage lies exactly where seam harmonics are expected to appear (`(k/S, 0)` and `(0, k/S)`), so without windowing the harmonic dips are contaminated by boundary artifacts.

This is not a tunable hyperparameter; treat it as a fixed preprocessing step. Both Koho et al. 2019 (MIPLIB) and NanoJ-SQUIRREL implement it as standard.

### 3.4. Per-image curve computation

Standard implementation:

1. Apply Hamming window to `P` and `G`.
2. Compute 2D FFT of each → `F_P`, `F_G`.
3. Define radial frequency bins. For an `N × N` image, bin edges at integer pixel distances from the DC bin, from `r = 0` to `r = N/2` (Nyquist). Bin centers in cycles/pixel are `r_i / N`.
4. For each bin, compute the FRC ratio using all Fourier-space pixels whose distance from the origin falls within the bin.
5. Return the array of FRC values and the array of frequency-bin centers in cycles/pixel.

### 3.5. Cross-image aggregation

Assumption: all images in a given dataset share the same size, so the native frequency grid is identical across images. No interpolation required.

For each (method × dataset) combination:

1. Compute the FRC curve for every (prediction, GT) pair.
2. Per-frequency-bin mean across the dataset: `μ(r_i) = mean_n FRC_n(r_i)`.
3. Per-frequency-bin 95% confidence interval: `± 1.96 · SE(r_i)` where `SE(r_i) = std_n FRC_n(r_i) / sqrt(N_images)`.

Optional (nice-to-have, not blocking): apply Fisher z-transform `z = atanh(FRC)` before averaging, compute CI in z-space, then transform back via `tanh` for plotting. For typical FRC values (not close to ±1) the difference vs. raw CI is small. Default to raw CI; offer Fisher z as an optional flag.

### 3.6. Headline plot

For each dataset, one figure with:
- Mean FRC curves for inner tiling and SWiTi on the same axes.
- 95% CI shown as shaded bands around each curve.
- X-axis: spatial frequency in cycles/pixel, from 0 to Nyquist (0.5).
- Vertical reference lines (dashed, lightly drawn) at expected harmonic locations `k/S` for `k = 1, 2, ..., S/2`.
- Legend: method name + `N_images` per band.

Expected result: visible dips in inner tiling's curve at the vertical reference lines; smooth SWiTi curve. Non-overlapping CI bands at dip locations are themselves visual evidence of significance.

### 3.7. Curve-to-scalar reduction — DEFERRED, placeholder

The reduction from per-method averaged curve to a single scalar is left as an open empirical question. Two candidates were discussed:

- **Dip depth at expected harmonics:** for each `f_k = k/S`, compute `D_k = baseline(f_k) − FRC_avg(f_k)` where `baseline(f_k)` is the median of `FRC_avg` in a small window around `f_k` excluding `f_k` itself. Aggregate as `mean_k D_k` or `max_k D_k`.
- **FRC-AUC:** integral of the FRC curve from 0 to Nyquist. Globally defined but conflates seam-specific signal with general reconstruction quality.

The choice between these (or a third alternative) depends on what the actual curves look like on the actual datasets. Implementer should:

1. Implement the per-image curve + per-frequency mean + CI plot first.
2. Inspect the resulting plots.
3. Defer scalar-extraction implementation until empirical evidence indicates which form is appropriate.

Leave a stub function `frc_to_scalar(curve, S)` returning `NaN` with a `NotImplementedError` and a clear comment noting this is deferred.

### 3.8. Useful implementation resources

- **MIPLIB** (https://github.com/sakoho81/miplib): open-source Python implementation of FRC by Koho et al. Contains windowing, radial binning, and the standard two-image FRC. Check license before copying code; if it's compatible, reuse rather than reimplement.
- **NanoJ-SQUIRREL** (https://github.com/superresolusian/NanoJ-SQUIRREL): ImageJ implementation of FRC mapping. Useful as a reference for behaviour; reimplementing the Java in Python is not recommended.

---

## 4. PSNR and MicroMS-SSIM

Standard restoration metrics, reference-based, scalar per image.

- **PSNR:** standard implementation. Use the same intensity normalization as in the MicroSplit paper (Ashesh et al. 2026, Nat. Methods).
- **MicroMS-SSIM:** as defined in the MicroSplit paper; use their implementation if available, otherwise reimplement following their specification.

Per-method scalar: mean ± std across images in the dataset. Report alongside FRC scalar.

These metrics are not the focus of the paper — they exist to ensure that the headline message (artifact reduction by SWiTi) is not confounded with overall quality degradation. If a method has worse PSNR/MS-SSIM, that is a finding that must be acknowledged regardless of how well it scores on artifact metrics.

---

## 5. 3D / FSC — DEFERRED, placeholder

Fourier Shell Correlation is the 3D analog of FRC: spherical shells in 3D Fourier space instead of rings in 2D. The MicroSplit datasets include some 3D data, so an FSC extension would broaden the applicability of the metrics.

**Why deferred:** fluorescence microscopy has anisotropic resolution along the Z axis (PSF elongated along the optical axis). Plain isotropic FSC averages across all directions and therefore (i) underrepresents the lateral resolution and (ii) dilutes any Z-direction seam signal across all Fourier-space directions. The principled solution is direction-resolved FSC (Koho et al. 2019 propose SFSC, with angular wedges), which is non-trivial to implement correctly and requires its own validation. This is too much complexity for the workshop submission.

**Placeholder action:** create a stub module `fsc.py` with a top-of-file comment stating the deferral reason and pointing to Koho et al. 2019 (Nat. Commun. 10:3103) for the direction-resolved approach. Implementer should not attempt FSC in this round.

---

## 6. Out-of-scope decisions — curiosities, do NOT implement

This section documents alternatives that were considered and explicitly rejected during methods design. **The coding agent should not implement any of these.** They are listed only so that future readers (including future versions of the agent) understand why the chosen approach is what it is, and do not "helpfully" add these as extras.

### 6.1. Per-pixel directional gradient asymmetry `m = grad_R − grad_B`

A spatial-domain metric comparing the right-direction gradient to the down-direction gradient at each pixel. Rejected because: (i) structurally a rediscovery of Wang–Bovik–Evans 2000 (ICIP), not novel; (ii) the per-tile hypothesis test already covers the spatial-domain story; (iii) FRC covers the spectral story with stronger credibility in the bioimaging community. Several variants were considered (anisotropy normalization, null calibration via random walk noise, Sobel-filter replacement of finite differences) — all rejected with the metric itself.

### 6.2. Spectral harmonic-to-background ratio on the 2D power spectrum

A direct measurement of seam-harmonic energy in the power spectrum of the prediction, with local background subtraction around expected harmonic locations. Rejected because the same harmonic signature is captured by FRC dips, with FRC additionally providing frequency-resolved comparison against ground truth. No methodological gain from implementing both.

### 6.3. One-image FRC (Koho et al. 2019 checkerboard sub-sampling)

A variant of FRC computed on a single image by splitting it into even/odd sub-images. Rejected because: (i) it measures self-consistency / effective resolution, not the question we're asking; (ii) the checkerboard sub-sampling is fragile in the presence of localized seam structure — seams get split across sub-images in a structured rather than random way, producing spurious decorrelation at seam frequencies that contaminates the harmonic-dip signal we want to detect. Standard two-image FRC against GT is the right tool for our use case.

---

## 7. Implementation order

Suggested order, from highest-priority to optional:

1. FRC per-image computation with Hamming windowing (§3.3–§3.4).
2. Per-method cross-image aggregation with 95% CI (§3.5).
3. Headline plot (§3.6).
4. PSNR and MicroMS-SSIM wrappers (§4) — likely a thin layer over existing implementations.
5. Stubs for the deferred FRC scalar (§3.7) and FSC (§5).

The per-tile hypothesis test (§2) is tracked separately under `per_tile_metric_design.md`.

---

## 8. Testing

Minimum tests for the FRC implementation:

- **Sanity:** FRC of a noise image against itself = 1 at all frequencies; FRC of independent noise images ≈ 0 at all frequencies above DC.
- **Windowing effect:** verify that the Hamming-windowed FRC of a non-tiled natural image shows no spurious peaks along the `u`/`v` axes of the power spectrum (compare a small power-spectrum visualization with and without windowing).
- **Harmonic detection on synthetic seams:** construct a synthetic image with a known periodic seam pattern (e.g., add a small intensity offset every `S` columns), compute FRC against the unseamed version, verify that dips appear at the expected harmonic locations.
- **Cross-image aggregation:** verify per-frequency mean and CI computation on a toy dataset (e.g., 10 FRC curves with known mean and variance).
