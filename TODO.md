# TODO

Running list of features, bugfixes, and cleanups for this repo.

## Features

- [ ] **Absolute-value gradient mode.** Add an option (e.g. `--abs_gradients` / `abs_gradients: bool` in `run_gradient_analysis_multi`) that takes `|grad_x|`, `|grad_y|` (and `|grad_z|` in 3D) before histogramming. Rationale: with signed differences, seams where the right tile is brighter (`+`) and seams where it is darker (`−`) cancel in the histogram, making the edge distribution look symmetric and "clean" even when every seam has a real jump. Magnitude-only collapses that cancellation and isolates the artifact strength. Touchpoints: [_extract_gradients_multi](src/analysis_pipeline/gradient_test/analysis.py#L128-L156) (apply `np.abs` after extraction, before normalization), CLI flag in [cli/analyze.py](src/analysis_pipeline/cli/analyze.py), config in [config/settings.py](src/analysis_pipeline/config/settings.py). Note: bin-edge computation in [_compute_histograms_multi](src/analysis_pipeline/gradient_test/analysis.py#L159-L185) will then span `[0, max]` instead of `[−max, max]`, so existing thresholds/plots may need adjustment.

- [ ] **Properly divided gradients (scale-invariant convention).** Currently the code computes raw first differences `img[k+1] − img[k]` and calls them "gradients." This is fine for ε = 1 (the `/1` is a no-op) but breaks down once long-range diffs are introduced: an ε-step un-divided value scales linearly with ε in the middle distribution while staying constant at the edge, making KL across ε values incomparable. Switch to the proper finite-difference convention `g_ε(X) := (img[X + ε] − img[X]) / ε`, i.e. divide by the step size. Identity (proven by telescoping): `g_ε(X) = (1/ε) · Σ_{k=0}^{ε−1} g_1(X+k)`, the **arithmetic mean** of nearest-neighbor gradients in the band — discrete analogue of the mean value theorem. Touchpoints: [compute_gradients](src/analysis_pipeline/gradient_test/gradient_analysis.py#L223-L239) (2D) and [compute_gradients](src/analysis_pipeline/gradient_test/gradient_analysis.py#L333-L351) (3D), plus wherever the ε-step sampler is added. With `ε = 1` the result is bit-identical to today, so this is a no-op for existing runs but makes ε > 1 dimensionally consistent.

- [ ] **Replace KL with symmetric Jensen-Shannon Divergence or Wasserstein-1.**
Also consider Kolmogorov-Smirnov distance (max CDF difference) for a non-parametric option. Rationale: KL is asymmetric and can be infinite if the edge distribution has heavier tails than the middle, which is common in real data and makes it hard to interpret the scores. JSD is a symmetrized and smoothed version of KL that is always finite and more interpretable as a distance metric. Wasserstein-1 (Earth Mover's Distance) captures the "effort" to transform one distribution into the other and is also symmetric and finite. Touchpoints: [compute_kl_divergence](src/analysis_pipeline/gradient_test/analysis.py#L187-L221) would be replaced with `compute_jsd` or `compute_wasserstein_distance`, and the CLI/config would need an option to select the metric.


## Bugs

- [ ] **`--bins` default mismatch between CLI and Python API.** CLI default is `100` ([cli/analyze.py:57](src/analysis_pipeline/cli/analyze.py#L57)); `run_gradient_analysis_multi` and `get_bin_edges` default to `200` ([gradient_analysis.py:62](src/analysis_pipeline/gradient_test/gradient_analysis.py#L62), README also documents `200`). Two runs invoked through the two entry points are silently incomparable. Pick one (likely 200, matching the README) and align.

## Cleanups

_(none yet)_
