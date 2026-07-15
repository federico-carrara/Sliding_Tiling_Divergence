"""Public orchestrators for the FRC stitching-artifact metric.

Two entrypoints, mirroring the gradient-test package:

- :func:`run_frc_analysis_dataset` — the lazy, streaming primitive. It consumes
  an iterable of ``(image_id, prediction, ground_truth)`` triples (each a
  channel-first ``(C, H, W)`` slice) so callers can feed one image into memory at
  a time (e.g. from a ``.npz`` archive). Direct analog of
  ``gradient_test.analysis.run_gradient_analysis_dataset``.
- :func:`run_frc_analysis` — a convenience wrapper over stacked ``(N, C, H, W)``
  arrays that validates shapes and delegates to the streaming primitive. Used by
  the CLIs and notebooks.

Multi-method comparison lives in :mod:`.comparison`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from analysis_pipeline.frc.aggregation import (
    FRCChannelResult,
    FRCImageReport,
    FRCMethodReport,
    aggregate_image,
    aggregate_method,
)
from analysis_pipeline.frc.frc import per_image_frc
from analysis_pipeline.frc.reduction import frc_resolution


def _resolve_channels(
    n_channels: int,
    gt_channels: int,
    channels: Optional[Sequence[int]],
    context: str,
) -> list[int]:
    """Resolve and validate the channel indices to analyse for one image.

    ``channels=None`` selects every channel of the image; otherwise each index
    must be in range for both the prediction and its ground truth.
    """
    if channels is None:
        return list(range(n_channels))
    resolved = list(channels)
    if not resolved:
        raise ValueError(f"{context}: channels must be a non-empty sequence or None")
    for c in resolved:
        if not (0 <= c < n_channels):
            raise ValueError(
                f"{context}: channel={c} out of range for C={n_channels}"
            )
        if not (0 <= c < gt_channels):
            raise ValueError(
                f"{context}: channel={c} out of range for GT C={gt_channels}"
            )
    return resolved


def run_frc_analysis_dataset(
    images: Iterable[tuple[str, np.ndarray, np.ndarray]],
    *,
    method_name: str = "method",
    dataset: Optional[str] = None,
    channels: Optional[Sequence[int]] = None,
    save_dir: Optional[Path] = None,
    apply_window: bool = True,
    verbose: bool = True,
) -> FRCMethodReport:
    """Run the FRC metric over a stream of (prediction, ground-truth) pairs.

    ``images`` is an iterable of ``(image_id, prediction, ground_truth)`` triples,
    consumed lazily so callers can stream one image into memory at a time (e.g.
    from a ``.npz`` archive) rather than materialising the whole dataset. Each
    ``prediction`` / ``ground_truth`` is a channel-first 2-D slice ``(C, H, W)``
    (FRC is 2-D only; 3-D volumes are z-sliced by the caller into extra images).

    All images must share the same spatial size: the per-method mean curve + CI
    are pooled per frequency bin, which requires a common frequency grid.
    :func:`analysis_pipeline.frc.aggregation.aggregate_method` raises a clear
    ``ValueError`` if the sizes differ.

    If ``save_dir`` is not None, the report is serialized as JSON to
    ``save_dir / f"{method_name}_frc_report.json"``.

    Parameters
    ----------
    images : iterable of (str, np.ndarray, np.ndarray)
        ``(image_id, prediction, ground_truth)`` triples; each array is
        channel-first ``(C, H, W)`` and a prediction shares its ground truth's
        spatial shape.
    method_name : str, default="method"
        Display name; also the report filename stem.
    dataset : str, optional
        Dataset name stamped onto the report.
    channels : sequence of int, optional
        Channel indices to analyse (``None`` = all channels of each image).
    save_dir : pathlib.Path, optional
        Directory for the JSON report (created if missing); ``None`` skips writing.
    apply_window : bool, default=True
        Apply a 2-D Hamming window before the FFT. Disable only for sanity tests;
        mandatory for real images (see ``frc.windowing``).
    verbose : bool, default=True
        Print a per-method per-channel summary.

    Returns
    -------
    FRCMethodReport
        Aggregated per-method report keyed by ``image_id``.

    Raises
    ------
    ValueError
        If a slice is not 3-D channel-first, a prediction and its ground truth
        disagree in spatial shape, a requested channel is out of range, or the
        images do not share a common frequency grid.
    """
    if verbose:
        print(f"  [{method_name}] running (channels={channels})")

    image_reports: dict[str, FRCImageReport] = {}
    for image_id, prediction, ground_truth in images:
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)
        if prediction.ndim != 3 or ground_truth.ndim != 3:
            raise ValueError(
                f"{method_name} [{image_id}]: expected 3-D channel-first slices "
                f"(C, H, W); got pred ndim={prediction.ndim}, gt ndim="
                f"{ground_truth.ndim}. FRC is 2-D only."
            )
        if prediction.shape[-2:] != ground_truth.shape[-2:]:
            raise ValueError(
                f"{method_name} [{image_id}]: spatial shape mismatch pred "
                f"{prediction.shape[-2:]} vs gt {ground_truth.shape[-2:]}"
            )
        chans = _resolve_channels(
            prediction.shape[0],
            ground_truth.shape[0],
            channels,
            f"{method_name} [{image_id}]",
        )
        channel_results: dict[int, FRCChannelResult] = {
            c: per_image_frc(
                prediction[c],
                ground_truth[c],
                apply_window=apply_window,
                channel=c,
            )
            for c in chans
        }
        image_reports[image_id] = aggregate_image(image_id, channel_results)

    method_report = aggregate_method(image_reports, method_name, dataset)

    if verbose:
        for c in sorted(method_report.mean_frc):
            mean_frc = method_report.mean_frc[c]
            body_mean = (
                float(np.nanmean(mean_frc[1:])) if mean_frc.size > 1 else float("nan")
            )
            f_c = frc_resolution(method_report.freqs, mean_frc)
            print(
                f"  -> {method_name} [c{c}]: n_images={method_report.n_images}, "
                f"mean FRC (excl. DC)={body_mean:.4f}, "
                f"resolution (FRC=1/7)={f_c:.4f} cyc/px"
            )

    if save_dir is not None:
        out_path = method_report.save(
            Path(save_dir) / f"{method_name}_frc_report.json"
        )
        if verbose:
            print(f"\nReport written to: {out_path}")

    return method_report


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
    verbose: bool = True,
) -> FRCMethodReport:
    """Run the FRC metric on a stacked set of (prediction, ground-truth) pairs.

    A thin convenience wrapper over :func:`run_frc_analysis_dataset`: it validates
    stacked channel-first ``(N, C, H, W)`` arrays and streams them as
    ``(image_id, prediction, ground_truth)`` triples.

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
    verbose : bool, default=True
        Print a per-method per-channel summary.

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

    n_images = predictions.shape[0]
    if image_ids is None:
        image_ids = [str(n) for n in range(n_images)]
    elif len(image_ids) != n_images:
        raise ValueError(
            f"{method_name}: image_ids has {len(image_ids)} entries, "
            f"expected {n_images} (one per image)"
        )

    triples = (
        (image_ids[n], predictions[n], ground_truths[n]) for n in range(n_images)
    )
    return run_frc_analysis_dataset(
        triples,
        method_name=method_name,
        dataset=dataset,
        channels=channels,
        save_dir=save_dir,
        apply_window=apply_window,
        verbose=verbose,
    )
