"""Fourier Ring Correlation (FRC) metric package.

Reference-based spectral metric for stitching-artifact evaluation: for each
``(prediction, ground_truth)`` pair, compute the 2-D FRC curve and aggregate
per-bin mean + 95% CI across the test set. Stitching artifacts in
inner-tiling predictions appear as dips at frequencies ``k/S`` where ``S``
is the inner tile size.

See ``agents_artifacts/FRC_metric.md`` for the full spec.
"""

from analysis_pipeline.frc.aggregation import (
    FRCChannelResult,
    FRCImageReport,
    FRCMethodReport,
    FRCMultiMethodReport,
    aggregate_image,
    aggregate_method,
)
from analysis_pipeline.frc.analysis import (
    run_frc_analysis,
    run_frc_analysis_dataset,
)
from analysis_pipeline.frc.comparison import run_frc_analysis_multi
from analysis_pipeline.frc.frc import per_image_frc
from analysis_pipeline.frc.reduction import (
    FRC_THRESHOLD_1_7,
    frc_resolution,
    frc_resolution_period,
)
from analysis_pipeline.frc.windowing import apply_hamming_window_2d

__all__ = [
    "FRCChannelResult",
    "FRCImageReport",
    "FRCMethodReport",
    "FRCMultiMethodReport",
    "FRC_THRESHOLD_1_7",
    "aggregate_image",
    "aggregate_method",
    "apply_hamming_window_2d",
    "frc_resolution",
    "frc_resolution_period",
    "per_image_frc",
    "run_frc_analysis",
    "run_frc_analysis_dataset",
    "run_frc_analysis_multi",
]
