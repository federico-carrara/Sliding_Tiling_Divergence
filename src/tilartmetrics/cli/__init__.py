"""Command-line interface modules."""

from analysis_pipeline.cli.analyze import main as analyze_main
from analysis_pipeline.cli.analyze_multi import main as analyze_multi_main
from analysis_pipeline.cli.calibrate import main as calibrate_main
from analysis_pipeline.cli.frc_analyze import main as frc_analyze_main
from analysis_pipeline.cli.frc_analyze_multi import main as frc_analyze_multi_main

__all__ = [
    "analyze_main",
    "analyze_multi_main",
    "calibrate_main",
    "frc_analyze_main",
    "frc_analyze_multi_main",
]
