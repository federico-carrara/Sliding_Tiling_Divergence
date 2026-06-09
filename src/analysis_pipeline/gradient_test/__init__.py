"""Core analysis modules."""

from analysis_pipeline.gradient_test.aggregation import (
    ImageReport,
    MethodReport,
    MultiMethodReport,
    TileResult,
)
from analysis_pipeline.gradient_test.analysis import run_gradient_analysis
from analysis_pipeline.gradient_test.comparison import run_gradient_analysis_multi
from analysis_pipeline.gradient_test.gradient_analysis import (
    compute_gradients,
    compute_gradients_2d,
    compute_gradients_3d,
)
from analysis_pipeline.gradient_test.per_tile import per_image_tile_scan
from analysis_pipeline.gradient_test.statistics import STATISTICS, StatisticSpec, get_statistic

__all__ = [
    "ImageReport",
    "MethodReport",
    "MultiMethodReport",
    "STATISTICS",
    "StatisticSpec",
    "TileResult",
    "compute_gradients",
    "compute_gradients_2d",
    "compute_gradients_3d",
    "get_statistic",
    "per_image_tile_scan",
    "run_gradient_analysis",
    "run_gradient_analysis_multi",
]
