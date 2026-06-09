"""Multi-method wrapper for the per-tile metric.

Loops methods → :func:`run_gradient_analysis` and returns a
:class:`MultiMethodReport`. Kept separate from
:mod:`.analysis` so the single-method orchestrator remains the primary API.
The signature preserves the legacy ``run_gradient_analysis_multi(predictions,
names, save_dir, tile_size, overlap, ...)`` argument order.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from analysis_pipeline.gradient_test.aggregation import MultiMethodReport
from analysis_pipeline.gradient_test.analysis import run_gradient_analysis


def _broadcast_per_method_spec(
    spec: list, n_methods: int, name: str
) -> list[list[int]]:
    """Normalise a per-axis or per-method tiling spec to a per-method list.

    Accepts a single per-axis spec (e.g. ``[64, 64]`` or ``[4, 64, 64]``),
    applied to every method, or a per-method list of per-axis specs
    (e.g. ``[[64, 64], [32, 32]]``).

    Parameters
    ----------
    spec : list
        Per-axis or per-method spec.
    n_methods : int
        Number of methods to broadcast to.
    name : str
        Spec name used in error messages.

    Returns
    -------
    list of list of int
        Per-method list of per-axis integer specs (length ``n_methods``).

    Raises
    ------
    ValueError
        If ``spec`` is empty or has the wrong per-method length.
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
    save_dir: Optional[Path],
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
    """Run the per-tile metric on N predictions and return a multi-method report.

    Predictions are channel-first: ``(N, C, H, W)`` for 2-D or
    ``(N, C, D, H, W)`` for 3-D. ``tile_size`` and ``overlap`` are per
    spatial axis and must match the TiledPatching configuration used to
    produce the predictions.

    If ``save_dir`` is not None, the report is pickled to
    ``save_dir / per_tile_report.pkl``. No other files are written.

    Parameters
    ----------
    predictions_list : list of np.ndarray
        One channel-first prediction array per method.
    method_names : list of str
        Method names matching ``predictions_list`` one-to-one.
    save_dir : pathlib.Path, optional
        Directory for the pickled report (created if missing); pass ``None``
        to skip writing.
    tile_size : list
        Either a single per-axis spec, or a per-method list of per-axis specs.
    overlap : list
        Either a single per-axis spec, or a per-method list of per-axis specs.
    statistic : str, default="kl"
        Two-sample discrepancy statistic name.
    strip_width : int, default=4
        Control-strip half-width ``N``.
    block_size : int, default=3
        Contiguous-block size ``B`` for permutation.
    n_permutations : int, default=1000
        Number of permutations ``R`` per tile.
    alpha : float, default=0.05
        Rejection threshold for the per-tile test.
    num_bins_per_tile : int, default=32
        Histogram bin count for binned statistics (KL, JS).
    random_seed : int, default=0
        RNG seed.
    pool_z_with_xy : bool, default=True
        If False (reserved for v2), run separate xy and z tests in 3D.
    channel : int, default=0
        Channel index to analyse.

    Returns
    -------
    MultiMethodReport
        Aggregated multi-method report.

    Raises
    ------
    ValueError
        If shapes or per-method specs are inconsistent with the predictions.
    """
    n_methods = len(predictions_list)
    if len(method_names) != n_methods:
        raise ValueError(
            f"method_names ({len(method_names)}) must match predictions "
            f"({n_methods})"
        )

    tile_size_per = _broadcast_per_method_spec(tile_size, n_methods, "tile_size")
    overlap_per = _broadcast_per_method_spec(overlap, n_methods, "overlap")

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
        report.methods[name] = run_gradient_analysis(
            pred,
            save_dir=None,
            tile_size=ts,
            overlap=ov,
            method_name=name,
            statistic=statistic,
            strip_width=strip_width,
            block_size=block_size,
            n_permutations=n_permutations,
            alpha=alpha,
            num_bins_per_tile=num_bins_per_tile,
            random_seed=random_seed,
            pool_z_with_xy=pool_z_with_xy,
            channel=channel,
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
    """Print a human-readable summary of a multi-method report.

    Parameters
    ----------
    report : MultiMethodReport
        Report to summarise.
    method_names : list of str
        Methods to display, in display order.
    statistic : str
        Statistic name shown in the header.
    """
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
