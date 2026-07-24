"""Calibrated per-tile Z-score checks.

``Z_obs = (T_obs - null_mean) / max(null_std, eps)`` expresses each tile's
statistic in units of its own permutation-null spread. This file checks:

- Under H0 (flat field) ``Z_obs`` is centred near 0 with order-1 spread across
  many tiles, and is always finite (the variance floor holds).
- A near-constant tile, where the null collapses, does not blow ``Z_obs`` up.
- The report models carry the new fields, round-trip through ``save``/``load``,
  and old reports lacking the fields still load (NaN defaults).

Run with ``PYTHONPATH=src python tests/test_z_score.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from tilartmetrics.gradient_test.aggregation import (
    MethodReport,
    TileResult,
    aggregate_channel,
    aggregate_image,
    aggregate_method,
)
from tilartmetrics.gradient_test.per_tile import per_image_tile_scan


def _valid_Z(report) -> np.ndarray:
    return np.array(
        [t.Z_obs for t in report.tiles if not np.isnan(t.Z_obs)],
        dtype=np.float64,
    )


def test_z_under_h0() -> None:
    """Flat field: Z_obs centred near 0, order-1 spread, all finite."""
    image = np.random.default_rng(0).standard_normal((256, 256)).astype(
        np.float64
    )
    report = per_image_tile_scan(
        image,
        tile_size=[32, 32],
        overlap=[0, 0],
        strip_width=4,
        block_size=3,
        n_permutations=400,
        statistic="kl",
        alpha=0.05,
        num_bins_per_tile=32,
        rng=np.random.default_rng(1),
    )

    z = _valid_Z(report)
    print(f"n_valid_Z = {z.size}")
    print(f"mean_Z    = {report.mean_Z:.3f}")
    print(f"median_Z  = {report.median_Z:.3f}")
    print(f"p90_Z     = {report.p90_Z:.3f}")
    print(f"std(Z)    = {float(np.std(z)):.3f}")

    assert z.size > 0, "no valid Z_obs produced"
    assert np.all(np.isfinite(z)), "Z_obs contains non-finite values under H0"
    # Sampling error of the mean over ~64 tiles is ~std/8, so |mean| < 0.5 is a
    # comfortable band. The null of a divergence stat is right-skewed, so the
    # spread is not exactly 1; require order-1.
    assert abs(report.mean_Z) < 0.5, f"mean_Z={report.mean_Z:.3f} not near 0"
    std_z = float(np.std(z))
    assert 0.3 <= std_z <= 2.5, f"std(Z)={std_z:.3f} not order-1"

    print("OK: Z under H0")


def test_null_collapse_guard() -> None:
    """Near-constant field collapses the null; Z_obs must stay finite."""
    # Constant plus vanishing noise: gradients (hence the statistic and its
    # null) are ~0, so null_std ~ 0 and the variance floor governs the ratio.
    rng = np.random.default_rng(2)
    image = np.ones((256, 256), dtype=np.float64)
    image += 1e-12 * rng.standard_normal((256, 256))
    report = per_image_tile_scan(
        image,
        tile_size=[32, 32],
        overlap=[0, 0],
        strip_width=4,
        block_size=3,
        n_permutations=200,
        statistic="kl",
        alpha=0.05,
        num_bins_per_tile=32,
        rng=np.random.default_rng(3),
    )

    z = _valid_Z(report)
    print(f"n_valid_Z (near-constant) = {z.size}")
    assert np.all(np.isfinite(z)), (
        "Z_obs exploded on a near-constant field — variance floor not applied"
    )
    print("OK: null-collapse guard")


def test_report_models_roundtrip() -> None:
    """New fields populate, round-trip, and old reports load with NaN defaults."""
    tiles = [
        TileResult(
            coord=(i,),
            n_seams=3,
            T_obs=float(i),
            p=0.5,
            null_mean=0.0,
            null_std=1.0,
            Z_obs=float(i),
            n_seam_samples=10,
            n_control_samples=10,
        )
        for i in range(5)
    ]
    channel = aggregate_channel(tiles, alpha=0.05, channel=0)
    assert channel.mean_Z == float(np.mean(range(5)))
    assert channel.p90_Z == float(np.percentile(range(5), 90))

    image = aggregate_image("img0", {0: channel}, alpha=0.05)
    method = aggregate_method({"img0": image}, method_name="m", dataset="d")
    assert 0 in method.mean_mean_Z
    assert method.mean_p90_Z[0] == channel.p90_Z

    # to_records exposes the Z summaries for the CLI's summary.csv.
    rec = method.to_records()[0]
    for key in ("mean_Z", "median_Z", "p90_Z"):
        assert key in rec, f"{key} missing from to_records()"

    with tempfile.TemporaryDirectory() as d:
        path = method.save(Path(d) / "report.json")
        loaded = MethodReport.load(path)
    assert loaded.mean_mean_Z == method.mean_mean_Z
    assert loaded.images["img0"].channels[0].mean_Z == channel.mean_Z

    # An old report row lacking the new keys must still validate (NaN defaults).
    legacy = TileResult.model_validate(
        {
            "coord": (0,),
            "n_seams": 3,
            "T_obs": 1.0,
            "p": 0.5,
            "n_seam_samples": 10,
            "n_control_samples": 10,
        }
    )
    assert np.isnan(legacy.Z_obs) and np.isnan(legacy.null_std)

    print("OK: report models round-trip and back-compat")


def main() -> None:
    test_z_under_h0()
    test_null_collapse_guard()
    test_report_models_roundtrip()
    print("ALL OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
