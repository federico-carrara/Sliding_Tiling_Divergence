"""Public orchestrator for the FRC stitching-artifact metric.

Loops methods → images → channel slice → :func:`per_image_frc` and returns
a :class:`FRCMultiMethodReport`. Mirrors the shape of
``analysis_pipeline.gradient_test.analysis.run_gradient_analysis_multi``
so the two metrics feel parallel from a user's perspective.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .aggregation import FRCMultiMethodReport, aggregate_method
from .frc import per_image_frc


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

    If ``save_dir`` is not None, the report is pickled to
    ``save_dir / frc_report.pkl``.

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
        Directory for the pickled report (created if missing); pass ``None``
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
        if pred.ndim != 4 or gt.ndim != 4:
            raise ValueError(
                f"{name}: expected 4-D arrays (N, C, H, W); got pred ndim="
                f"{pred.ndim}, gt ndim={gt.ndim}. FRC is 2-D only."
            )
        if pred.shape[0] != gt.shape[0]:
            raise ValueError(
                f"{name}: prediction N={pred.shape[0]} != ground-truth "
                f"N={gt.shape[0]}"
            )
        if pred.shape[-2:] != gt.shape[-2:]:
            raise ValueError(
                f"{name}: spatial shape mismatch pred {pred.shape[-2:]} vs "
                f"gt {gt.shape[-2:]}"
            )
        if not (0 <= channel < pred.shape[1]):
            raise ValueError(
                f"{name}: channel={channel} out of range for C={pred.shape[1]}"
            )
        if not (0 <= channel < gt.shape[1]):
            raise ValueError(
                f"{name}: channel={channel} out of range for GT C={gt.shape[1]}"
            )

        print(f"  [{name}] {pred.shape[0]} images × channel {channel}")

        image_results = []
        for n in range(pred.shape[0]):
            res = per_image_frc(
                pred[n, channel],
                gt[n, channel],
                apply_window=apply_window,
            )
            image_results.append(res)

        method_report = aggregate_method(image_results)
        report.methods[name] = method_report
        body_mean = float(np.nanmean(method_report.mean_frc[1:]))
        print(
            f"  -> {name}: n_images={method_report.n_images}, "
            f"mean FRC (excl. DC)={body_mean:.4f}, "
            f"FRC[Nyquist]={method_report.mean_frc[-1]:.4f}"
        )

    _print_summary(report, method_names)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "frc_report.pkl", "wb") as f:
            pickle.dump(report, f)
        print(f"\nReport pickled to: {save_dir / 'frc_report.pkl'}")

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
