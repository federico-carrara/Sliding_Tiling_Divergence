"""Fourier Ring Correlation (FRC) metric package.

Reference-based spectral metric for stitching-artifact evaluation: for each
``(prediction, ground_truth)`` pair, compute the 2-D FRC curve and aggregate
per-bin mean + 95% CI across the test set. Stitching artifacts in
inner-tiling predictions appear as dips at frequencies ``k/S`` where ``S``
is the inner tile size.

See ``agents_artifacts/FRC_metric.md`` for the full spec.
"""

from .aggregation import (
    FRCImageResult,
    FRCMethodReport,
    FRCMultiMethodReport,
    aggregate_method,
)
from .analysis import run_frc_analysis
from .comparison import run_frc_analysis_multi
from .frc import per_image_frc
from .windowing import apply_hamming_window_2d

__all__ = [
    "FRCImageResult",
    "FRCMethodReport",
    "FRCMultiMethodReport",
    "aggregate_method",
    "apply_hamming_window_2d",
    "per_image_frc",
    "run_frc_analysis",
    "run_frc_analysis_multi",
]
