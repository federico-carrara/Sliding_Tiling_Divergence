"""Public orchestrator for the FRC stitching-artifact metric.

Loops images → channel slices → :func:`per_image_frc` and returns a
:class:`FRCMethodReport`. This is the primary public API: one method, a set
of ``(prediction, ground_truth)`` pairs. Multi-method comparison lives in
:mod:`.comparison`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from analysis_pipeline.frc.aggregation import (
    FRCChannelResult,
    FRCImageReport,
    FRCMethodReport,
    aggregate_image,
    aggregate_method,
)
from analysis_pipeline.frc.frc import per_image_frc


def run_frc_analysis(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    save_dir: Optional[Path],
    *,
    method_name: str = "method",
    channels: Optional[Sequence[int]] = None,
    image_ids: Optional[Sequence[str]] = None,
    dataset: Optional[str] = None,
    apply_window: bool = True,
) -> FRCMethodReport:
    """Run the FRC metric on a set of (prediction, ground-truth) pairs.

    Both arrays are channel-first ``(N, C, H, W)`` with matching ``(N, H, W)``
    layout so each prediction has a corresponding ground truth.

    If ``save_dir`` is not None, the report is serialized as JSON to
    ``save_dir / f"{method_name}_frc_report.json"``.

    Parameters
    ----------
    predictions : np.ndarray
        Channel-first prediction array.
    ground_truths : np.ndarray
        Channel-first ground-truth array, same ``(N, H, W)`` as ``predictions``.
    save_dir : pathlib.Path, optional
        Directory for the JSON report (created if missing); pass ``None``
        to skip writing.
    method_name : str, default="method"
        Display name used in logs and the report filename.
    channels : sequence of int, optional
        Channel indices to analyse. If ``None`` (default), every channel is
        analysed. Each image's report then holds one ``FRCChannelResult`` per
        requested channel.
    image_ids : sequence of str, optional
        Identifier for each image, one per ``N``; used as the keys of the
        returned ``images`` dict. Defaults to ``"0" .. "N-1"``.
    dataset : str, optional
        Name of the dataset the predictions were drawn from; stamped onto the
        report for downstream analysis.
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
    n_channels = predictions.shape[1]
    if channels is None:
        channels = list(range(n_channels))
    else:
        channels = list(channels)
        if not channels:
            raise ValueError(
                f"{method_name}: channels must be a non-empty sequence or None"
            )
        for c in channels:
            if not (0 <= c < n_channels):
                raise ValueError(
                    f"{method_name}: channel={c} out of range for "
                    f"C={n_channels}"
                )
            if not (0 <= c < ground_truths.shape[1]):
                raise ValueError(
                    f"{method_name}: channel={c} out of range for "
                    f"GT C={ground_truths.shape[1]}"
                )

    n_images = predictions.shape[0]
    if image_ids is None:
        image_ids = [str(n) for n in range(n_images)]
    elif len(image_ids) != n_images:
        raise ValueError(
            f"{method_name}: image_ids has {len(image_ids)} entries, "
            f"expected {n_images} (one per image)"
        )

    print(f"  [{method_name}] {n_images} images × channels {channels}")

    images: dict[str, FRCImageReport] = {}
    for n, image_id in enumerate(image_ids):
        channel_results: dict[int, FRCChannelResult] = {}
        for c in channels:
            channel_results[c] = per_image_frc(
                predictions[n, c],
                ground_truths[n, c],
                apply_window=apply_window,
                channel=c,
            )
        images[image_id] = aggregate_image(image_id, channel_results)

    method_report = aggregate_method(images, method_name, dataset)
    for c in channels:
        mean_frc = method_report.mean_frc[c]
        body_mean = float(np.nanmean(mean_frc[1:])) if mean_frc.size > 1 else float("nan")
        nyquist = float(mean_frc[-1]) if mean_frc.size else float("nan")
        print(
            f"  -> {method_name} [c{c}]: n_images={method_report.n_images}, "
            f"mean FRC (excl. DC)={body_mean:.4f}, FRC[Nyquist]={nyquist:.4f}"
        )

    if save_dir is not None:
        out_path = method_report.save(
            Path(save_dir) / f"{method_name}_frc_report.json"
        )
        print(f"\nReport written to: {out_path}")

    return method_report
