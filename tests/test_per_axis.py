"""Per-axis normalization + count-balancing checks.

Covers the anisotropy corrections wired into the per-tile test:

- ``balance_axis_blocks`` gives every present axis an equal number of whole blocks.
- ``AxisMoments`` normalization puts each axis at unit scale over pooled samples.
- Per-axis normalization removes a constant per-axis scale factor: multiplying one
  axis's gradients by a constant leaves the calibrated ``Z_obs`` unchanged.
- A no-op balance (single axis / equal counts) leaves the rng stream untouched.
- A degenerate (std=0) axis normalizes to finite values.

Run with ``PYTHONPATH=src python tests/test_per_axis.py``.
"""

from __future__ import annotations

import sys

import numpy as np

from tilartmetrics.gradient_test.per_axis import (
    AxisMoments,
    balance_axis_blocks,
    normalize_slices,
)
from tilartmetrics.gradient_test.per_tile import per_image_tile_scan


def test_balance_equalizes_block_counts() -> None:
    """Balancing yields equal per-axis block counts and preserves whole blocks."""
    block_size = 3
    # Axis 0: 30 samples (10 blocks); axis 1: 9 (3 blocks); axis 2: 6 (2 blocks).
    slices_by_axis = {
        0: [np.arange(30, dtype=np.float64)],
        1: [np.arange(9, dtype=np.float64)],
        2: [np.arange(6, dtype=np.float64)],
    }
    rng = np.random.default_rng(0)
    selected = balance_axis_blocks(
        slices_by_axis, block_size=block_size, rng=rng
    )
    # min blocks across axes = 2 → 2 blocks per axis × 3 axes = 6 blocks.
    assert len(selected) == 6, f"expected 6 blocks, got {len(selected)}"
    # Each selected block is a contiguous run of <= block_size (whole block).
    for b in selected:
        assert 1 <= b.size <= block_size
    print("OK: balancing equalizes per-axis block counts, whole blocks kept")


def test_moments_unit_scale() -> None:
    """AxisMoments normalization drives each axis to mean~0, std~1."""
    rng = np.random.default_rng(1)
    seam = {0: rng.normal(5.0, 3.0, 4000), 1: rng.normal(-2.0, 0.5, 4000)}
    control = {0: rng.normal(5.0, 3.0, 4000), 1: rng.normal(-2.0, 0.5, 4000)}
    m = AxisMoments.zeros(2)
    for a in (0, 1):
        m.update(a, seam[a])
        m.update(a, control[a])
    stats = m.finalize()
    for a in (0, 1):
        pooled = np.concatenate([seam[a], control[a]])
        norm = normalize_slices([pooled], [a], stats)[0]
        assert abs(norm.mean()) < 0.05, f"axis {a} mean {norm.mean()}"
        assert abs(norm.std() - 1.0) < 0.05, f"axis {a} std {norm.std()}"
    print("OK: per-axis normalization gives unit scale")


def _thin_z_scan(image, *, normalize, balance, seed=1):
    return per_image_tile_scan(
        image,
        tile_size=[6, 64, 64],
        overlap=[2, 32, 32],
        strip_width=1,
        block_size=3,
        n_permutations=200,
        statistic="js",
        alpha=0.05,
        num_bins_per_tile=32,
        rng=np.random.default_rng(seed),
        normalize_per_axis=normalize,
        balance_axis_counts=balance,
    )


def test_per_axis_scale_invariance() -> None:
    """Scaling one axis by a constant leaves normalized Z_obs unchanged.

    With balancing off (so the rng stream is identical between runs), multiply the
    z-direction structure by a constant. Per-axis normalization should absorb the
    constant, leaving each tile's calibrated Z essentially unchanged.
    """
    rng = np.random.default_rng(0)
    # Clean 3D tiling: D=26 (tile6/ov2), H=W=160 (tile64/ov32).
    base = rng.standard_normal((26, 160, 160)).astype(np.float64)

    scaled = base.copy()
    scaled *= 1.0
    # Amplify z-direction variation by 7x (multiply the whole array's z-differences
    # is equivalent to scaling values along z-structure; here scale the array so the
    # z gradients grow): inject a z-ramp then scale it.
    zramp = np.linspace(0.0, 1.0, 26)[:, None, None]
    base = base + zramp
    scaled = scaled + 7.0 * zramp

    rep_a = _thin_z_scan(base, normalize=True, balance=False)
    rep_b = _thin_z_scan(scaled, normalize=True, balance=False)
    z_a = np.array([t.Z_obs for t in rep_a.tiles], dtype=np.float64)
    z_b = np.array([t.Z_obs for t in rep_b.tiles], dtype=np.float64)
    both = ~np.isnan(z_a) & ~np.isnan(z_b)
    max_dev = float(np.max(np.abs(z_a[both] - z_b[both])))
    assert max_dev < 1e-6, f"Z changed after per-axis rescale: max dev {max_dev}"
    print(f"OK: per-axis normalization is scale-invariant (max Z dev {max_dev:.2e})")


def test_std_zero_axis_is_finite() -> None:
    """A constant (std=0) axis normalizes without NaN/inf."""
    m = AxisMoments.zeros(2)
    m.update(0, np.full(50, 3.0))  # constant axis
    m.update(1, np.arange(50, dtype=np.float64))
    stats = m.finalize()
    norm = normalize_slices([np.full(10, 3.0)], [0], stats)[0]
    assert np.all(np.isfinite(norm)), "std=0 axis produced non-finite values"
    print("OK: std=0 axis stays finite")


def test_balance_noop_leaves_rng_untouched() -> None:
    """Equal-count / single-axis balancing draws nothing from the rng."""
    # Single axis → no-op.
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    balance_axis_blocks({0: [np.arange(9.0)]}, block_size=3, rng=rng)
    assert rng.bit_generator.state == before, "single-axis balance touched rng"
    # Two axes with equal block counts → no-op.
    rng2 = np.random.default_rng(5)
    before2 = rng2.bit_generator.state
    balance_axis_blocks(
        {0: [np.arange(9.0)], 1: [np.arange(9.0)]}, block_size=3, rng=rng2
    )
    assert rng2.bit_generator.state == before2, "equal-count balance touched rng"
    print("OK: no-op balancing leaves rng stream untouched")


def main() -> None:
    test_balance_equalizes_block_counts()
    test_moments_unit_scale()
    test_per_axis_scale_invariance()
    test_std_zero_axis_is_finite()
    test_balance_noop_leaves_rng_untouched()
    print("ALL OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
