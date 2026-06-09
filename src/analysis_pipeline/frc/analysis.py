"""Public orchestrator for the FRC stitching-artifact metric.

Loops images → channel slice → :func:`per_image_frc` and returns a
:class:`FRCMethodReport`. This is the primary public API: one method, a set
of ``(prediction, ground_truth)`` pairs. Multi-method comparison lives in
:mod:`.comparison`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from analysis_pipeline.frc.aggregation import FRCMethodReport, aggregate_method
from analysis_pipeline.frc.frc import per_image_frc


def run_frc_analysis(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    save_dir: Optional[Path],
    *,
    method_name: str = "method",
    channel: int = 0,
    apply_window: bool = True,
) -> FRCMethodReport:
    """Run the FRC metric on a set of (prediction, ground-truth) pairs.

    Both arrays are channel-first ``(N, C, H, W)`` with matching ``(N, H, W)``
    layout so each prediction has a corresponding ground truth.

    If ``save_dir`` is not None, the report is pickled to
    ``save_dir / f"{method_name}_frc_report.pkl"``.

    Parameters
    ----------
    predictions : np.ndarray
        Channel-first prediction array.
    ground_truths : np.ndarray
        Channel-first ground-truth array, same ``(N, H, W)`` as ``predictions``.
    save_dir : pathlib.Path, optional
        Directory for the pickled report (created if missing); pass ``None``
        to skip writing.
    method_name : str, default="method"
        Display name used in logs and the pickle filename.
    channel : int, default=0
        Channel index to analyse.
    apply_window : bool, default=True
        Apply a 2-D Hamming window before the FFT. Disable only for sanity
        tests; mandatory for real images (see ``frc.windowing``).

    Returns
    -------
    FRCMethodReport
        Aggregated per-method report.

    Raises
    ------
    ValueError
        If shapes are wrong, or prediction and ground truth shapes do not
        match.
    """
    if predictions.ndim != 4 or ground_truths.ndim != 4:
        raise ValueError(
            f"{method_name}: expected 4-D arrays (N, C, H, W); got pred ndim="
            f"{predictions.ndim}, gt ndim={ground_truths.ndim}. FRC is 2-D only."
        )
    if predictions.shape[0] != ground_truths.shape[0]:
        raise ValueError(
            f"{method_name}: prediction N={predictions.shape[0]} != "
            f"ground-truth N={ground_truths.shape[0]}"
        )
    if predictions.shape[-2:] != ground_truths.shape[-2:]:
        raise ValueError(
            f"{method_name}: spatial shape mismatch pred "
            f"{predictions.shape[-2:]} vs gt {ground_truths.shape[-2:]}"
        )
    if not (0 <= channel < predictions.shape[1]):
        raise ValueError(
            f"{method_name}: channel={channel} out of range for "
            f"C={predictions.shape[1]}"
        )
    if not (0 <= channel < ground_truths.shape[1]):
        raise ValueError(
            f"{method_name}: channel={channel} out of range for "
            f"GT C={ground_truths.shape[1]}"
        )

    print(f"  [{method_name}] {predictions.shape[0]} images × channel {channel}")

    image_results = []
    for n in range(predictions.shape[0]):
        res = per_image_frc(
            predictions[n, channel],
            ground_truths[n, channel],
            apply_window=apply_window,
        )
        image_results.append(res)

    method_report = aggregate_method(image_results)
    body_mean = float(np.nanmean(method_report.mean_frc[1:]))
    print(
        f"  -> {method_name}: n_images={method_report.n_images}, "
        f"mean FRC (excl. DC)={body_mean:.4f}, "
        f"FRC[Nyquist]={method_report.mean_frc[-1]:.4f}"
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{method_name}_frc_report.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(method_report, f)
        print(f"\nReport pickled to: {out_path}")

    return method_report
