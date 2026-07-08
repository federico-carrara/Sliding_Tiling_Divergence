"""Multi-method wrapper for the FRC metric.

Loops methods → :func:`run_frc_analysis` and returns a
:class:`FRCMultiMethodReport`. Kept separate from :mod:`.analysis` so the
single-method orchestrator remains the primary API. Mirrors the shape of
``analysis_pipeline.gradient_test.comparison.run_gradient_analysis_multi``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from analysis_pipeline.frc.aggregation import FRCMultiMethodReport
from analysis_pipeline.frc.analysis import run_frc_analysis


def run_frc_analysis_multi(
    predictions_list: list[np.ndarray],
    ground_truths_list: list[np.ndarray],
    method_names: list[str],
    save_dir: Optional[Path],
    *,
    channel: int = 0,
    apply_window: bool = True,
) -> FRCMultiMethodReport:
    """Run the FRC metric on N (prediction, ground-truth) sets.

    Predictions and ground truths are channel-first ``(N, C, H, W)``. The
    per-method ``ground_truths_list[i]`` must have the same ``(N, H, W)``
    layout as ``predictions_list[i]`` so each prediction has a matching GT.

    If ``save_dir`` is not None, the report is serialized as JSON to
    ``save_dir / frc_report.json``.

    Parameters
    ----------
    predictions_list : list of np.ndarray
        One channel-first prediction array per method.
    ground_truths_list : list of np.ndarray
        One channel-first ground-truth array per method, in 1:1 correspondence
        with ``predictions_list``.
    method_names : list of str
        Method names matching ``predictions_list`` one-to-one.
    save_dir : pathlib.Path, optional
        Directory for the JSON report (created if missing); pass ``None``
        to skip writing.
    channel : int, default=0
        Channel index to analyse.
    apply_window : bool, default=True
        Apply a 2-D Hamming window before the FFT. Disable only for sanity
        tests; mandatory for real images (see ``frc.windowing``).

    Returns
    -------
    FRCMultiMethodReport
        Aggregated multi-method report.

    Raises
    ------
    ValueError
        If list lengths disagree, shapes are wrong, or shapes between a
        prediction and its ground truth do not match.
    """
    n_methods = len(predictions_list)
    if len(ground_truths_list) != n_methods:
        raise ValueError(
            f"ground_truths_list ({len(ground_truths_list)}) must match "
            f"predictions_list ({n_methods})"
        )
    if len(method_names) != n_methods:
        raise ValueError(
            f"method_names ({len(method_names)}) must match predictions_list "
            f"({n_methods})"
        )

    report = FRCMultiMethodReport(
        methods={},
        config_summary={
            "channel": channel,
            "apply_window": apply_window,
        },
    )

    for pred, gt, name in zip(
        predictions_list, ground_truths_list, method_names, strict=True
    ):
        report.methods[name] = run_frc_analysis(
            pred,
            gt,
            save_dir=None,
            method_name=name,
            channel=channel,
            apply_window=apply_window,
        )

    _print_summary(report, method_names)

    if save_dir is not None:
        out_path = report.save(Path(save_dir) / "frc_report.json")
        print(f"\nReport written to: {out_path}")

    return report


def _print_summary(
    report: FRCMultiMethodReport, method_names: list[str]
) -> None:
    """Print a human-readable summary of a multi-method FRC report.

    Parameters
    ----------
    report : FRCMultiMethodReport
        Report to summarise.
    method_names : list of str
        Methods to display, in display order.
    """
    bar = "=" * 60
    print()
    print(bar)
    print("FRC METRIC SUMMARY")
    print(bar)
    print(
        f"{'Method':<25s} {'n_images':>10s} {'mean FRC':>12s} {'FRC[Nyq]':>10s}"
    )
    print("-" * 60)
    for name in method_names:
        m = report.methods[name]
        body_mean = (
            float(np.nanmean(m.mean_frc[1:])) if m.mean_frc.size > 1 else float("nan")
        )
        nyq = float(m.mean_frc[-1]) if m.mean_frc.size else float("nan")
        print(
            f"{name:<25s} {m.n_images:>10d} {body_mean:>12.4f} {nyq:>10.4f}"
        )
    print(bar)
