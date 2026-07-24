"""Core analysis modules."""

from tilartmetrics.gradient_test.aggregation import (
    ChannelReport,
    ImageReport,
    MethodReport,
    TileResult,
)
from tilartmetrics.gradient_test.analysis import (
    run_gradient_analysis,
    run_gradient_analysis_dataset,
)
from tilartmetrics.gradient_test.gradient_analysis import (
    compute_gradients,
    compute_gradients_2d,
    compute_gradients_3d,
)
from tilartmetrics.gradient_test.per_tile import per_image_tile_scan
from tilartmetrics.gradient_test.plotting import (
    plot_gradient_comparison,
    plot_pvalue_distribution,
    plot_significance_overlay,
    plot_significance_overlay_grid,
)
from tilartmetrics.gradient_test.statistics import STATISTICS, StatisticSpec, get_statistic

__all__ = [
    "ChannelReport",
    "ImageReport",
    "MethodReport",
    "STATISTICS",
    "StatisticSpec",
    "TileResult",
    "compute_gradients",
    "compute_gradients_2d",
    "compute_gradients_3d",
    "get_statistic",
    "per_image_tile_scan",
    "plot_gradient_comparison",
    "plot_pvalue_distribution",
    "plot_significance_overlay",
    "plot_significance_overlay_grid",
    "run_gradient_analysis",
    "run_gradient_analysis_dataset",
]
