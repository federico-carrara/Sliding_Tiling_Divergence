"""Configuration settings for analysis pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class GradientConfig:
    """Configuration for gradient analysis."""

    inner_tile_size: List[int] = field(default_factory=lambda: [32])
    bins: int = 100
    channel: Optional[int] = 0
    border_size: int = 0
    kl_start: int = 29
    kl_end: int = 33
    eps: float = 1e-12


@dataclass
class PlotConfig:
    """Configuration for plotting."""

    dpi: int = 300
    figsize_per_method: tuple = (8, 4)
    colormap: str = "coolwarm"
    colors: List[str] = field(
        default_factory=lambda: ["blue", "red", "green", "orange", "purple"]
    )


@dataclass
class AnalysisConfig:
    """Main configuration for analysis pipeline."""

    model_name: str
    dataset: str
    save_dir: Path
    predictions: List[str]
    method_names: List[str]
    padding: List[int] = field(default_factory=lambda: [48])

    # Sub-configs
    gradient: GradientConfig = field(default_factory=GradientConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    # Analysis flags
    run_gradient_analysis: bool = True
    run_qualitative_analysis: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.save_dir = Path(self.save_dir)

        if len(self.predictions) != len(self.method_names):
            raise ValueError(
                f"Number of predictions ({len(self.predictions)}) must match "
                f"method names ({len(self.method_names)})"
            )

        if len(self.predictions) < 2:
            raise ValueError("At least 2 predictions required")

        if len(self.predictions) > 5:
            raise ValueError("Maximum 5 predictions supported")

        # Ensure padding list matches predictions
        if len(self.padding) == 1:
            self.padding = self.padding * len(self.predictions)
        elif len(self.padding) != len(self.predictions):
            raise ValueError(
                f"Padding length ({len(self.padding)}) must be 1 or match "
                f"predictions ({len(self.predictions)})"
            )


def load_config_from_args(args) -> AnalysisConfig:
    """
    Create AnalysisConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        AnalysisConfig instance
    """
    pred_files = [p.strip() for p in args.predictions.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]

    gradient_config = GradientConfig(
        inner_tile_size=args.inner_tile_size,
        bins=args.bins,
        channel=args.channel,
        border_size=getattr(args, "border_size", 0),
    )

    return AnalysisConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        save_dir=args.save_dir,
        predictions=pred_files,
        method_names=method_names,
        padding=args.padding,
        gradient=gradient_config,
        run_gradient_analysis=args.gradient_analysis
        and not getattr(args, "skip_gradient_analysis", False),
        run_qualitative_analysis=getattr(args, "qualitative_analysis", True),
    )
