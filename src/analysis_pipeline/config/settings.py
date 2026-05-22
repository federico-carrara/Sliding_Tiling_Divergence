"""Configuration models for the per-tile analysis pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

# Names that ``statistics.STATISTICS`` exposes. Kept as a Literal for
# pydantic to validate without importing core/ from config/.
Statistic = Literal["kl", "js", "ks", "wasserstein", "mean_abs_ratio"]


class PerTileConfig(BaseModel):
    """Parameters of the per-tile two-sample test."""

    tile_size: list[PositiveInt]
    overlap: list[NonNegativeInt]
    statistic: Statistic = "kl"
    strip_width: int = Field(default=4, ge=1)
    block_size: int = Field(default=3, ge=1)
    n_permutations: int = Field(default=1000, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    num_bins_per_tile: int = Field(default=32, ge=2)
    random_seed: int = 0
    diagnostic_n_tiles: int = Field(default=16, ge=0)
    pool_z_with_xy: bool = True
    channel: int = Field(default=0, ge=0)

    @field_validator("n_permutations")
    @classmethod
    def _warn_low_n_permutations(cls, v: int) -> int:
        if v < 100:
            warnings.warn(
                f"n_permutations={v} is below the recommended 100; "
                "p-value resolution will be poor.",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def _check_axis_consistency(self) -> Self:
        """Cross-field checks: matching lengths and the step / strip-width geometry."""
        if len(self.tile_size) != len(self.overlap):
            raise ValueError(
                f"tile_size and overlap must have the same number of axes "
                f"(got {len(self.tile_size)} vs {len(self.overlap)})"
            )
        min_step = 2 * self.strip_width + 2
        for i, (t, o) in enumerate(zip(self.tile_size, self.overlap)):
            if o >= t:
                raise ValueError(
                    f"overlap[{i}]={o} must be < tile_size[{i}]={t}"
                )
            step = t - o
            if step < min_step:
                raise ValueError(
                    f"axis {i}: step = tile_size - overlap = {step} < "
                    f"{min_step} = 2*strip_width + 2. Lower strip_width or "
                    f"reduce overlap[{i}]."
                )
        return self


class AnalysisConfig(BaseModel):
    """Top-level configuration consumed by the CLI."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    dataset: str
    save_dir: Path
    predictions: list[str]
    method_names: list[str]
    per_tile: PerTileConfig

    @model_validator(mode="after")
    def _check_predictions_and_methods(self) -> Self:
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
    """Build an ``AnalysisConfig`` from parsed CLI ``args``.

    Expects ``args.tile_size`` and ``args.overlap`` to be flat per-axis
    integer lists (the multi-method per-prediction broadcasting is handled
    later inside ``run_gradient_analysis_multi``).
    """
    pred_files = [p.strip() for p in args.predictions.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]

    per_tile = PerTileConfig(
        tile_size=list(args.tile_size),
        overlap=list(args.overlap),
        statistic=args.statistic,
        strip_width=args.strip_width,
        block_size=args.block_size,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        num_bins_per_tile=args.num_bins_per_tile,
        random_seed=args.random_seed,
        diagnostic_n_tiles=args.diagnostic_n_tiles,
        pool_z_with_xy=args.pool_z_with_xy,
        channel=args.channel,
    )

    return AnalysisConfig(
        name=args.model_name,
        dataset=args.dataset,
        save_dir=Path(args.save_dir),
        predictions=pred_files,
        method_names=method_names,
        per_tile=per_tile,
    )
