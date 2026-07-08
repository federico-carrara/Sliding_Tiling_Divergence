"""FRC correctness checks (handout §8).

Run with::

    /localscratch/miniforge3/envs/sliding_tiling_env/bin/python tests/test_frc.py

Exits non-zero on any failure; no pytest required.
"""

from __future__ import annotations

import sys

import numpy as np

from analysis_pipeline.frc import (
    FRCImageResult,
    aggregate_method,
    per_image_frc,
)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def test_self_frc_is_one() -> None:
    """FRC of a noise image against itself must be 1 at every frequency."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((128, 128))
    res = per_image_frc(img, img, apply_window=False)
    valid = ~np.isnan(res.frc)
    deviation = np.max(np.abs(res.frc[valid] - 1.0))
    _expect(
        deviation < 1e-10,
        f"self-FRC deviates from 1: max|FRC - 1| = {deviation:.3e}",
    )


def test_independent_noise_frc_is_zero() -> None:
    """FRC of two independent noise images must be ~0 above DC.

    For a ring of ``N`` pixels in 2-D Fourier space the FRC of independent
    Gaussian noise has magnitude ~ 1/sqrt(N). Inner rings (small r) have
    few pixels so we skip the first few bins; the RMS bound below is a
    comfortable multiple of the theoretical noise floor.
    """
    rng = np.random.default_rng(1)
    a = rng.standard_normal((1024, 1024))
    b = rng.standard_normal((1024, 1024))
    res = per_image_frc(a, b, apply_window=False)
    skip = 10
    body = res.frc[skip:-1]
    body = body[~np.isnan(body)]
    rms = float(np.sqrt(np.mean(body**2)))
    max_abs = float(np.max(np.abs(body)))
    _expect(
        rms < 0.06,
        f"indep-noise FRC RMS too high: {rms:.3f} (expected < 0.06)",
    )
    _expect(
        max_abs < 0.20,
        f"indep-noise FRC max|.| too high: {max_abs:.3f} (expected < 0.20)",
    )


def test_freqs_range_and_count() -> None:
    """Frequency grid spans [0, ~Nyquist] in 1/N steps with N=min(H,W)."""
    img = np.zeros((64, 96), dtype=np.float64)
    res = per_image_frc(img, img + 1.0, apply_window=False)
    n = 64
    expected_bins = n // 2 + 1
    _expect(
        res.freqs.shape == (expected_bins,),
        f"freqs shape {res.freqs.shape} != ({expected_bins},)",
    )
    _expect(
        abs(res.freqs[0]) < 1e-12,
        f"freqs[0] should be 0 (DC), got {res.freqs[0]}",
    )
    _expect(
        abs(res.freqs[-1] - 0.5) < 1e-12,
        f"freqs[-1] should be 0.5 (Nyquist), got {res.freqs[-1]}",
    )


def test_aggregation_toy() -> None:
    """Per-bin mean and 95% CI match closed-form values on a toy dataset."""
    freqs = np.linspace(0.0, 0.5, 6)
    rng = np.random.default_rng(42)
    n_images = 20
    # Construct curves with known per-bin mean = bin_index * 0.1 and
    # std = 0.05 (independent Gaussian noise per image per bin).
    base = np.tile(np.arange(6) * 0.1, (n_images, 1))
    noise = 0.05 * rng.standard_normal(base.shape)
    curves = base + noise

    images = [
        FRCImageResult(freqs=freqs.copy(), frc=curves[i], image_shape=(64, 64))
        for i in range(n_images)
    ]
    rep = aggregate_method(images)

    _expect(rep.n_images == n_images, f"n_images={rep.n_images} != {n_images}")
    _expect(
        np.array_equal(rep.freqs, freqs),
        "aggregated freqs grid does not match input",
    )

    # Mean should be very close to bin_index * 0.1 with 20 samples and std 0.05.
    expected_mean = np.arange(6) * 0.1
    mean_err = np.max(np.abs(rep.mean_frc - expected_mean))
    _expect(
        mean_err < 0.05,
        f"aggregated mean deviates: max|mean - expected| = {mean_err:.3f}",
    )

    # CI half-width should be 1.96 * 0.05 / sqrt(20) ≈ 0.0219, within
    # sampling noise of that.
    expected_hw = 1.96 * 0.05 / np.sqrt(n_images)
    hw = (rep.ci95_hi - rep.ci95_lo) / 2.0
    hw_err = float(np.max(np.abs(hw - expected_hw)))
    _expect(
        hw_err < 0.015,
        f"CI half-width deviates: max|hw - {expected_hw:.4f}| = {hw_err:.4f}",
    )


def test_synthetic_seam_harmonic() -> None:
    """Periodic seam pattern -> dip at the seam fundamental frequency.

    Construct an image with a small additive offset every ``S`` columns.
    FRC against the unseamed version should drop at frequency ``1/S``
    relative to nearby non-harmonic bins.
    """
    rng = np.random.default_rng(7)
    h = w = 256
    s = 32
    base = rng.standard_normal((h, w))
    seamed = base.copy()
    # Pulse at every column k*s: bias = +0.5 on that single column.
    cols = np.arange(s, w, s)
    seamed[:, cols] += 0.5

    res = per_image_frc(seamed, base, apply_window=True)
    # Convert harmonic freq to nearest bin index. With windowing the leakage
    # is bounded, but the dominant dip should still be within +/-1 bin of 1/s.
    n = min(h, w)
    target_idx = int(round((1.0 / s) * n))
    window_lo = max(target_idx - 2, 1)
    window_hi = min(target_idx + 3, len(res.frc) - 1)

    # Reference baseline: median FRC in a wide neighbourhood that excludes
    # all harmonics k/s for k >= 1.
    harmonic_idxs = {int(round((k / s) * n)) for k in range(1, n // (2 * s) + 1)}
    nonharmonic = np.array(
        [
            res.frc[i]
            for i in range(5, len(res.frc) - 5)
            if not any(abs(i - h_i) <= 2 for h_i in harmonic_idxs)
            and not np.isnan(res.frc[i])
        ]
    )
    baseline = float(np.median(nonharmonic))
    dip = float(np.min(res.frc[window_lo:window_hi]))
    _expect(
        baseline - dip > 0.05,
        f"no visible dip at f=1/{s}: baseline={baseline:.3f}, "
        f"min in window=[{window_lo}:{window_hi}]={dip:.3f}",
    )


def main() -> None:
    test_self_frc_is_one()
    print("OK: self-FRC == 1")
    test_independent_noise_frc_is_zero()
    print("OK: independent-noise FRC ~ 0")
    test_freqs_range_and_count()
    print("OK: frequency grid")
    test_aggregation_toy()
    print("OK: aggregation (mean + 95% CI)")
    test_synthetic_seam_harmonic()
    print("OK: synthetic seam harmonic dip")


if __name__ == "__main__":
    main()
