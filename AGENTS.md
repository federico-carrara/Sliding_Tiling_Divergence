# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

A small Python library for **quantifying tiling artifacts** in stitched large-scale images produced by tiled inference. It compares the distribution of finite-difference gradients sampled **across tile seams** to those sampled **inside tiles**: the closer the two distributions, the smoother the stitching. The headline metric is an asymmetric KL divergence `KL(mid ‖ edge)` — lower is better.

Entry points:
- CLI: `analyze-experiment` → [src/analysis_pipeline/cli/analyze.py](src/analysis_pipeline/cli/analyze.py)
- Python API: `run_gradient_analysis_multi` in [src/analysis_pipeline/core/analysis.py](src/analysis_pipeline/core/analysis.py)

## Layout

```
src/analysis_pipeline/
├── cli/analyze.py          # argparse CLI, prediction loading, channel loop
├── config/settings.py      # config dataclass + arg validation
├── core/
│   ├── gradient_analysis.py  # GradientUtils2D / GradientUtils3D
│   ├── metrics.py            # KL divergence, peakiness, Wiener entropy
│   ├── analysis.py           # multi-method orchestrator + summary writer
│   └── plotting.py           # histogram & KL-heatmap figures
└── utils/                   # I/O, padding, 4D reshaping
```

## How the pipeline works

### 1. Tile edges / stitching seams

Seams are **not detected** from pixel content — the inner tile size `T` is supplied by the user (`--inner_tile_size 32`, or `"4,32,32;5,32,32"` per method, parsed in [analyze.py:91-119](src/analysis_pipeline/cli/analyze.py#L91-L119)).

Pipeline:
1. Trim outer borders via [border_free](src/analysis_pipeline/core/gradient_analysis.py#L205-L221) (2D) / [border_free](src/analysis_pipeline/core/gradient_analysis.py#L310-L331) (3D).
2. Compute first-difference gradients along each axis ([compute_gradients](src/analysis_pipeline/core/gradient_analysis.py#L223-L239) for 2D; [compute_gradients](src/analysis_pipeline/core/gradient_analysis.py#L333-L351) for 3D).
3. Subsample on the tile lattice via strided slicing in [_gradients_along_tile_grid](src/analysis_pipeline/core/gradient_analysis.py#L241-L273): `grad_x[:, :, ox::tile_sz_x, :]`, `grad_y[:, oy::tile_sz_y, :, :]` (and `oz::tile_sz_z` for 3D in [_gradients_along_tile_grid](src/analysis_pipeline/core/gradient_analysis.py#L353-L382)).
4. The **edge** offset is `tile_size − 1` (last pixel of each tile = the seam); the **middle** offset is `tile_size // 2 − 1` (interior reference). See [get_gradients_at](src/analysis_pipeline/core/gradient_analysis.py#L275-L301).

So seams are implicitly the periodic lattice of pixel columns/rows/planes spaced `T` apart, anchored at the last pixel of each tile.

### 2. The metric (asymmetric KL)

The KL implementation in [metrics.py:91-105](src/analysis_pipeline/core/metrics.py#L91-L105) is the plain asymmetric form:

```python
sum(p * log((p+eps)/(q+eps)))
```

with both histograms normalized to sum to 1 ([normalize_histogram](src/analysis_pipeline/core/metrics.py#L76-L88)).

The reported summary is `KL(mid ‖ edge)` per method — computed in [_write_summary_multi](src/analysis_pipeline/core/analysis.py#L300-L324) as `compute_kl_matrix([h["mid_i"], h["edge_i"]])[0, 1]` and written to `summary_kl_divergence.txt`. **Not symmetric** (no JS, no symmetrized KL); `compute_kl_matrix` builds the full pairwise asymmetric table but only the mid→edge direction is surfaced.

Histograms share bin edges across all methods so KL values are comparable: see [_compute_histograms_multi](src/analysis_pipeline/core/analysis.py#L159-L185), which calls [get_bin_edges](src/analysis_pipeline/core/gradient_analysis.py#L61-L75) (`np.histogram` with `bins=num_bins`, default 200 — CLI default is 100).

### 3. Pixel counts feeding the distributions

Two distinct populations:

- **Gradient field**: computed on every pixel of the border-trimmed image. For an `(N, H, W, C)` array that's ~`N·H·(W−1)·C` x-gradients plus ~`N·(H−1)·W·C` y-gradients (analogous for z in 3D). See [compute_gradients](src/analysis_pipeline/core/gradient_analysis.py#L237-L239).

- **Edge / middle distributions** (what actually enters the histograms): strided subsamples on the tile lattice ([_gradients_along_tile_grid](src/analysis_pipeline/core/gradient_analysis.py#L241-L273)). For tile size `T` in 2D each distribution is roughly:

  ```
  N · H · ⌈W/T⌉ · C    (grad_x at columns offset by T)
  + N · ⌈H/T⌉ · W · C  (grad_y at rows offset by T)
  ≈ 2 · N·H·W·C / T
  ```

  With `T=32` that's ~1/16 of all gradient pixels per distribution. In 3D it's ~`3·N·D·H·W·C/T` (with equal `T` per axis).

Before histogramming, both edge and middle gradients are z-score normalized using the **middle's** mean/std for that method (per-method self-normalization), in [_extract_gradients_multi](src/analysis_pipeline/core/analysis.py#L128-L156) via [_normalize_gradients](src/analysis_pipeline/core/gradient_analysis.py#L155-L177).

Channel control: `--channel <int>` restricts to one channel; `--channel 2` is a special CLI flag meaning "loop over channels 0 and 1 separately" ([analyze.py:172-184](src/analysis_pipeline/cli/analyze.py#L172-L184)).

## Conventions

- 2D arrays are `(N, H, W, C)`; 3D arrays are `(N, D, H, W, C)`. Dimensionality is inferred by `len(shape)` in [run_gradient_analysis_multi](src/analysis_pipeline/core/analysis.py#L54-L62); 6D inputs are squeezed once in the CLI.
- `inner_tile_size` may be `int`, single tile spec (`[32]`, `[4,32,32]`), or per-method list of specs; normalization happens in [run_gradient_analysis_multi](src/analysis_pipeline/core/analysis.py#L38-L51).
- Output directory: `<save_dir>/Gradient_Analysis/` with `gradient_histograms_all_methods.png` and `summary_kl_divergence.txt`.
- `GradientUtils` is an abstract base; the 2D/3D subclasses implement `border_free`, `compute_gradients`, and `get_gradients_at`.

## Gotchas

- Changing the histogram normalization rule (currently each method self-normalizes by its own middle stats) breaks the reported KL — a "combined-normalization" path exists in [_plot_combined_histogram_multi](src/analysis_pipeline/core/analysis.py#L217-L270) but is **not** wired into the summary. The comment at [analysis.py:319-321](src/analysis_pipeline/core/analysis.py#L319-L321) flags this as intentional.
- KL direction matters: only `KL(mid ‖ edge)` is reported. If you flip it, the ranking can change.
- `border_free` in 2D uses a symmetric int crop; in 3D it accepts `[z, y, x]`. The pipeline passes `[0, border, border]` for 3D (no z-cropping by default — see [run_gradient_analysis_multi](src/analysis_pipeline/core/analysis.py#L56-L62)).
- The CLI sets `border=0` when calling `run_gradient_analysis_multi` ([analyze.py:183, 193](src/analysis_pipeline/cli/analyze.py#L183)) — outer cropping is expected to be done already via `--padding` (`remove_padding`). Don't double-crop.
- `--bins` CLI default (100) differs from the API default (200). Use the same value across runs you want to compare.

## Running it

```bash
pip install -e .

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

There is no test suite at present.
