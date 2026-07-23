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

from tilartmetrics.gradient_test.per_tile import per_image_tile_scan
from tilartmetrics.gradient_test.seams import compute_seam_positions


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

    affected_columns = (1, 2)  # seams_x[1] separates region 1 from region 2

    def _metrics(normalize: bool, balance: bool):
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
            normalize_per_axis=normalize,
            balance_axis_counts=balance,
        )
        aff_T, far_T, aff_rej, far_rej, aff_total, far_total = [], [], 0, 0, 0, 0
        for t in report.tiles:
            in_aff = t.coord[1] in affected_columns
            if in_aff:
                aff_total += 1
            else:
                far_total += 1
            if not np.isnan(t.T_obs):
                (aff_T if in_aff else far_T).append(t.T_obs)
            if not np.isnan(t.p) and t.p < ALPHA:
                if in_aff:
                    aff_rej += 1
                else:
                    far_rej += 1
        return (
            float(np.median(aff_T)),
            float(np.median(far_T)),
            aff_rej / aff_total,
            far_rej / far_total,
        )

    # Raw engine (anisotropy corrections off): the original, strict claim — the
    # injected seam elevates affected-column T by > 3x and concentrates rejections.
    med_aff, med_far, aff_rate, far_rate = _metrics(normalize=False, balance=False)
    print(f"[raw]      median T aff/far = {med_aff:.3f}/{med_far:.3f} "
          f"| rej aff/far = {aff_rate:.2f}/{far_rate:.2f}")
    assert aff_rate >= 0.5, f"[raw] affected rejection rate {aff_rate:.2f} < 0.5"
    assert far_rate <= 0.15, f"[raw] far rejection rate {far_rate:.2f} > 0.15"
    assert med_aff > 3 * med_far, "[raw] median T aff not > 3x far"

    # Anisotropy defaults on: per-axis normalization + count balancing slightly
    # dilute a single-axis 2-D artifact (balancing subsamples the artifact axis in
    # edge tiles), so the magnitude gap shrinks — but localization must survive.
    med_aff, med_far, aff_rate, far_rate = _metrics(normalize=True, balance=True)
    print(f"[defaults] median T aff/far = {med_aff:.3f}/{med_far:.3f} "
          f"| rej aff/far = {aff_rate:.2f}/{far_rate:.2f}")
    assert aff_rate >= 0.5, f"[defaults] affected rejection rate {aff_rate:.2f} < 0.5"
    assert far_rate <= 0.15, f"[defaults] far rejection rate {far_rate:.2f} > 0.15"
    assert med_aff > 2 * med_far, "[defaults] median T aff not > 2x far"

    print("OK: artifact injection")


if __name__ == "__main__":
    main()
