"""Synthetic artifact-injection check.

Inject a constant additive shift on one side of a single seam (so the
across-seam gradient there is biased) and verify that the kept-region tiles
owning that seam concentrate the rejections — even though each tile pools
samples across 3-4 owned seams (only one of which carries the bias), the
affected columns should still show a clear elevation in ``T_obs`` and
rejection rate vs. tiles far from the seam.

Run with ``PYTHONPATH=src python tests/test_artifact_injection.py``.
"""

from __future__ import annotations

import sys

import numpy as np

from analysis_pipeline.core.per_tile import per_image_tile_scan
from analysis_pipeline.core.seams import compute_seam_positions


def main() -> None:
    H = W = 256
    TS = 32
    OV = 0
    ALPHA = 0.05

    rng_image = np.random.default_rng(0)
    image = rng_image.standard_normal((H, W)).astype(np.float64)

    seams_x = compute_seam_positions(W, TS, OV)
    assert seams_x.size >= 2, "need at least 2 interior seams along x"
    x_seam = int(seams_x[1])
    image[:, x_seam:] += 2.0

    report = per_image_tile_scan(
        image,
        tile_size=[TS, TS],
        overlap=[OV, OV],
        strip_width=4,
        block_size=3,
        n_permutations=400,
        statistic="kl",
        alpha=ALPHA,
        num_bins_per_tile=32,
        rng=np.random.default_rng(2),
    )

    affected_columns = (1, 2)  # seams_x[1] separates region 1 from region 2

    aff_T = [
        t.T_obs
        for t in report.tiles
        if t.coord[1] in affected_columns and not np.isnan(t.T_obs)
    ]
    far_T = [
        t.T_obs
        for t in report.tiles
        if t.coord[1] not in affected_columns and not np.isnan(t.T_obs)
    ]
    aff_rej = sum(
        1
        for t in report.tiles
        if t.coord[1] in affected_columns
        and not np.isnan(t.p)
        and t.p < ALPHA
    )
    far_rej = sum(
        1
        for t in report.tiles
        if t.coord[1] not in affected_columns
        and not np.isnan(t.p)
        and t.p < ALPHA
    )
    aff_total = len([t for t in report.tiles if t.coord[1] in affected_columns])
    far_total = len([t for t in report.tiles if t.coord[1] not in affected_columns])

    print(f"affected columns (j in {affected_columns}):")
    print(f"  median T = {np.median(aff_T):.3f} | rejected {aff_rej}/{aff_total}")
    print("far columns:")
    print(f"  median T = {np.median(far_T):.3f} | rejected {far_rej}/{far_total}")

    aff_rate = aff_rej / aff_total
    far_rate = far_rej / far_total
    assert aff_rate >= 0.5, f"affected rejection rate {aff_rate:.2f} < 0.5"
    assert far_rate <= 0.15, f"far rejection rate {far_rate:.2f} > 0.15"
    assert np.median(aff_T) > 3 * np.median(far_T), (
        "median T in affected columns not clearly elevated above far columns"
    )

    print("OK: artifact injection")


if __name__ == "__main__":
    main()
