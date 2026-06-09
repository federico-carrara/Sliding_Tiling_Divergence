"""Core analysis modules."""

from .aggregation import (
    ImageReport,
    MethodReport,
    MultiMethodReport,
    TileResult,
)
from .analysis import run_gradient_analysis
from .comparison import run_gradient_analysis_multi
from .gradient_analysis import (
    compute_gradients,
    compute_gradients_2d,
    compute_gradients_3d,
)
from .per_tile import per_image_tile_scan
from .statistics import STATISTICS, StatisticSpec, get_statistic

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
