"""Smoke test for the block-size calibration pipeline.

On a flat-field reference (no spatial correlation), every candidate ``B``
should pass the selection rule and the recommended ``B`` should be the
smallest candidate. We also flip the tolerance negative to force a
"no candidate passes" branch and confirm the report handles it cleanly.

Run with ``PYTHONPATH=src python tests/test_calibration.py``.
"""

from __future__ import annotations

import sys

import numpy as np

from analysis_pipeline.core.calibration import calibrate_block_size


def main() -> None:
    rng = np.random.default_rng(0)
    # 512×512 flat field → ~225 tiles with tile_size=32, overlap=0
    image = rng.standard_normal((512, 512)).astype(np.float64)

    candidates = (1, 2, 4, 8)
    report = calibrate_block_size(
        [image],
        tile_size=[32, 32],
        overlap=[0, 0],
        strip_width=4,
        statistic="kl",
        n_permutations=200,
        num_bins_per_tile=32,
        alpha=0.05,
        tolerance=0.05,  # generous so noise can't flip a true PASS to FAIL
        candidate_block_sizes=candidates,
        n_seeds=1,
        base_seed=0,
        verbose=False,
    )

    assert len(report.candidates) == len(candidates), (
        f"expected {len(candidates)} candidates, got {len(report.candidates)}"
    )
    bs = [c.block_size for c in report.candidates]
    assert bs == list(candidates), f"candidate order: {bs}"

    # Pure white noise should sit close to alpha for every B, well under
    # the generous threshold (0.10).
    for c in report.candidates:
        assert c.frac_rejected_mean < 0.10, (
            f"B={c.block_size}: frac_rejected={c.frac_rejected_mean:.3f} too high "
            "on flat-field reference"
        )
        assert c.n_tiles_total > 0, f"B={c.block_size}: zero tiles counted"

    assert report.recommended_block_size == 1, (
        f"expected recommended B=1 on white noise, got {report.recommended_block_size}"
    )

    # Negative-tolerance branch: even alpha itself is unattainable in
    # practice with binomial noise, so nothing should pass.
    strict = calibrate_block_size(
        [image],
        tile_size=[32, 32],
        overlap=[0, 0],
        strip_width=4,
        statistic="kl",
        n_permutations=200,
        num_bins_per_tile=32,
        alpha=0.05,
        tolerance=-0.049,  # threshold = 0.001 — essentially unattainable
        candidate_block_sizes=candidates,
        n_seeds=1,
        base_seed=0,
        verbose=False,
    )
    assert strict.recommended_block_size is None, (
        f"expected no recommendation under strict tolerance, got "
        f"{strict.recommended_block_size}"
    )

    print(f"OK: calibration ({len(report.candidates)} candidates, "
          f"recommended B={report.recommended_block_size})")


if __name__ == "__main__":
    main()
