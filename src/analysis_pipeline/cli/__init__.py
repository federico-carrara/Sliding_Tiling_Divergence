"""Command-line interface modules."""

from .analyze import main as analyze_main
from .analyze_multi import main as analyze_multi_main
from .calibrate import main as calibrate_main
from .frc_analyze import main as frc_analyze_main
from .frc_analyze_multi import main as frc_analyze_multi_main

__all__ = [
    "analyze_main",
    "analyze_multi_main",
    "calibrate_main",
    "frc_analyze_main",
    "frc_analyze_multi_main",
]
