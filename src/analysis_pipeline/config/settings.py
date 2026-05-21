"""Configuration settings for analysis pipeline."""

from pathlib import Path
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GradientConfig(BaseModel):
    """Configuration for gradient analysis."""

    tile_size: list[int]
    overlap: list[int]
    bins: int = 100
    channel: Optional[int]

    @model_validator(mode="after")
    def _check_tiling(self) -> Self:
        if len(self.tile_size) != len(self.overlap):
            raise ValueError(
                f"tile_size and overlap must have the same number of axes "
                f"(got {len(self.tile_size)} vs {len(self.overlap)})"
            )
        for i, (t, o) in enumerate(zip(self.tile_size, self.overlap)):
            if t <= 0:
                raise ValueError(f"tile_size[{i}]={t} must be > 0")
            if o < 0:
                raise ValueError(f"overlap[{i}]={o} must be >= 0")
            if o >= t:
                raise ValueError(
                    f"overlap[{i}]={o} must be < tile_size[{i}]={t}"
                )
        return self


class PlotConfig(BaseModel):
    """Configuration for plotting."""

    dpi: int = 300
    figsize_per_method: tuple[float, float] = (8.0, 4.0)
    colormap: str = "Greys_r"
    colors: list[str] = Field(
        default_factory=lambda: ["blue", "red", "green", "orange", "purple"]
    )


class AnalysisConfig(BaseModel):
    """Main configuration for analysis pipeline."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    dataset: str
    save_dir: str | Path
    predictions: list[str]
    method_names: list[str]
    gradient_config: GradientConfig
    plot_config: PlotConfig = Field(default_factory=PlotConfig)
    run_gradient_analysis: bool = True
    run_qualitative_analysis: bool = True

    @model_validator(mode="after")
    def _check_predictions_and_methods(self) -> "AnalysisConfig":
        if len(self.predictions) != len(self.method_names):
            raise ValueError(
                f"Number of predictions ({len(self.predictions)}) must match "
                f"method names ({len(self.method_names)})"
            )
        if len(self.predictions) < 2:
            raise ValueError("At least 2 predictions required")
        if len(self.predictions) > 5:
            raise ValueError("Maximum 5 predictions supported")
        return self


def load_config_from_args(args) -> AnalysisConfig:
    """
    Create AnalysisConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        AnalysisConfig instance
    """
    # TODO: rewire CLI args (--tile_size, --overlap) — the CLI still exposes
    # legacy --inner_tile_size / --padding / --border_size flags which this
    # loader intentionally ignores after the seam-locator refactor. Until the
    # CLI is updated, callers should drive the pipeline from Python with an
    # explicit GradientConfig.
    pred_files = [p.strip() for p in args.predictions.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]

    gradient_config = GradientConfig(
        bins=args.bins,
        channel=args.channel,
    )

    return AnalysisConfig(
        model_name=args.model_name,
        dataset=args.dataset,
        save_dir=args.save_dir,
        predictions=pred_files,
        method_names=method_names,
        gradient=gradient_config,
        run_gradient_analysis=args.gradient_analysis
        and not getattr(args, "skip_gradient_analysis", False),
        run_qualitative_analysis=getattr(args, "qualitative_analysis", True),
    )
