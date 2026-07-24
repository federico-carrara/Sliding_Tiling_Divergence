"""Flat-field null calibration check.

The strongest correctness signal for the per-tile permutation test: on an
artifact-free flat field, ``p_tile`` should be approximately uniform on
[0, 1] and ``frac_rejected`` should be near ``alpha``.

Run with ``PYTHONPATH=src python tests/test_null_calibration.py``.
"""

from __future__ import annotations

import sys

import numpy as np

from tilartmetrics.gradient_test.per_tile import per_image_tile_scan


def main() -> None:
    rng = np.random.default_rng(0)

    # 256x256 flat field; tile_size=32 / overlap=0 → 8 seams per axis → 9×9 grid
    image = rng.standard_normal((256, 256)).astype(np.float64)
    H, W = image.shape
    TS = 32
    OV = 0
    ALPHA = 0.05

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
        rng=np.random.default_rng(1),
    )

    p_values = np.array(
        [t.p for t in report.tiles if not np.isnan(t.p)], dtype=np.float64
    )
    print(f"n_tiles_with_test = {p_values.size}")
    print(f"median p          = {np.median(p_values):.3f}")
    print(f"frac_rejected     = {report.frac_rejected:.3f}  (alpha={ALPHA})")

    # With ~64 valid tiles (interior 7×7 = 49; plus edge/corner) and alpha=0.05,
    # frac_rejected should comfortably sit in [0.0, 0.15]. A failure here means
    # the permutation calibration is broken.
    assert report.frac_rejected <= 0.20, (
        f"frac_rejected={report.frac_rejected:.3f} too high under H0 — "
        "permutation calibration is suspect."
    )
    # Median p under H0 should be near 0.5.
    median_p = float(np.median(p_values))
    assert 0.3 <= median_p <= 0.7, f"median p={median_p:.3f} not near 0.5"

    print("OK: null calibration")


if __name__ == "__main__":
    main()
