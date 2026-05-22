"""Public orchestrator for the per-tile stitching-artifact metric.

Loops methods → images → channel slice → :func:`per_image_tile_scan` and
returns a :class:`MultiMethodReport`. The signature preserves the legacy
``run_gradient_analysis_multi(predictions, names, save_dir, tile_size,
overlap, ...)`` argument order so the CLI keeps working; everything else
moves behind keyword-only flags.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

from .aggregation import MultiMethodReport, aggregate_method
from .per_tile import per_image_tile_scan


def _broadcast_per_method_spec(
    spec: list, n_methods: int, name: str
) -> list[list[int]]:
    """Normalise a per-axis or per-method tiling spec to a per-method list.

    Accepts:
      - a single per-axis spec (e.g. ``[64, 64]`` or ``[4, 64, 64]``) — applied
        to every method;
      - a per-method list of per-axis specs (e.g. ``[[64, 64], [32, 32]]``).
    """
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        raise ValueError(f"{name} must be a non-empty list/tuple, got {spec!r}")
    if all(isinstance(v, int) for v in spec):
        return [list(spec) for _ in range(n_methods)]
    if len(spec) != n_methods:
        raise ValueError(
            f"{name} has {len(spec)} per-method entries; expected {n_methods} "
            "to match the number of predictions"
        )
    return [list(s) for s in spec]


def run_gradient_analysis_multi(
    predictions_list: list[np.ndarray],
    method_names: list[str],
    save_dir: Path,
    tile_size: list,
    overlap: list,
    *,
    statistic: str = "kl",
    strip_width: int = 4,
    block_size: int = 3,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    num_bins_per_tile: int = 32,
    random_seed: int = 0,
    pool_z_with_xy: bool = True,
    channel: int = 0,
) -> MultiMethodReport:
    """Run the per-tile metric on N predictions and return a report.

    Predictions are channel-first: ``(N, C, H, W)`` for 2-D or
    ``(N, C, D, H, W)`` for 3-D. ``tile_size`` and ``overlap`` are per
    spatial axis and must match the TiledPatching configuration used to
    produce the predictions.

    ``save_dir``, if not None, receives a pickled ``MultiMethodReport`` at
    ``save_dir / per_tile_report.pkl``. No other files are written — raw
    per-tile dumps and visualizations are deferred to v2.
    """
    n_methods = len(predictions_list)
    if len(method_names) != n_methods:
        raise ValueError(
            f"method_names ({len(method_names)}) must match predictions "
            f"({n_methods})"
        )

    tile_size_per = _broadcast_per_method_spec(tile_size, n_methods, "tile_size")
    overlap_per = _broadcast_per_method_spec(overlap, n_methods, "overlap")

    is_2d = predictions_list[0].ndim == 4  # (N, C, H, W)
    n_spatial = 2 if is_2d else 3

    if not is_2d and not pool_z_with_xy:
        warnings.warn(
            "pool_z_with_xy=False is reserved for a future revision; the v1 "
            "implementation always pools all spatial axes. Result reflects "
            "pool_z_with_xy=True.",
            stacklevel=2,
        )

    rng = np.random.default_rng(random_seed)

    report = MultiMethodReport(
        methods={},
        config_summary={
            "statistic": statistic,
            "strip_width": strip_width,
            "block_size": block_size,
            "n_permutations": n_permutations,
            "alpha": alpha,
            "num_bins_per_tile": num_bins_per_tile,
            "random_seed": random_seed,
            "pool_z_with_xy": pool_z_with_xy,
            "channel": channel,
            "tile_size": tile_size_per,
            "overlap": overlap_per,
        },
    )

    for pred, name, ts, ov in zip(
        predictions_list, method_names, tile_size_per, overlap_per, strict=True
    ):
        if pred.ndim != n_spatial + 2:
            raise ValueError(
                f"{name}: expected ndim={n_spatial + 2} (N,C,...); got {pred.ndim}"
            )
        if len(ts) != n_spatial:
            raise ValueError(
                f"{name}: tile_size has {len(ts)} entries, expected {n_spatial}"
            )
        if len(ov) != n_spatial:
            raise ValueError(
                f"{name}: overlap has {len(ov)} entries, expected {n_spatial}"
            )
        if not (0 <= channel < pred.shape[1]):
            raise ValueError(
                f"{name}: channel={channel} out of range for C={pred.shape[1]}"
            )

        print(f"  [{name}] {pred.shape[0]} images × channel {channel}")

        image_reports = []
        for n in range(pred.shape[0]):
            image = pred[n, channel]
            ir = per_image_tile_scan(
                image,
                tile_size=ts,
                overlap=ov,
                strip_width=strip_width,
                block_size=block_size,
                n_permutations=n_permutations,
                statistic=statistic,
                alpha=alpha,
                num_bins_per_tile=num_bins_per_tile,
                rng=rng,
            )
            image_reports.append(ir)
            print(
                f"    image {n}: median_T={ir.median_T:.4g} "
                f"frac_rejected={ir.frac_rejected:.3f}"
            )

        method_report = aggregate_method(image_reports)
        report.methods[name] = method_report
        print(
            f"  -> {name}: mean_median_T={method_report.mean_median_T:.4g} "
            f"mean_frac_rejected={method_report.mean_frac_rejected:.3f}"
        )

    _print_summary(report, method_names, statistic)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "per_tile_report.pkl", "wb") as f:
            pickle.dump(report, f)
        print(f"\nReport pickled to: {save_dir / 'per_tile_report.pkl'}")

    return report


def _print_summary(
    report: MultiMethodReport, method_names: list[str], statistic: str
) -> None:
    bar = "=" * 60
    print()
    print(bar)
    print(f"PER-TILE METRIC SUMMARY (statistic={statistic})")
    print(bar)
    print(f"{'Method':<25s} {'mean_median_T':>15s} {'mean_frac_rejected':>20s}")
    print("-" * 60)
    for name in method_names:
        m = report.methods[name]
        print(
            f"{name:<25s} {m.mean_median_T:>15.4g} "
            f"{m.mean_frac_rejected:>20.3f}"
        )
    print(bar)
