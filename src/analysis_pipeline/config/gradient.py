"""Configuration model for the gradient-test (per-tile permutation) metric."""

from __future__ import annotations

import warnings
from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

# Names that ``gradient_test.statistics.STATISTICS`` exposes. Kept as a
# Literal so pydantic can validate without importing the gradient-test
# package from ``config/``.
Statistic = Literal["kl", "js", "ks", "wasserstein", "mean_abs_ratio"]


class GradientTestConfig(BaseModel):
    """Parameters of the per-tile two-sample test.

    Attributes
    ----------
    tile_size : list of int
        TiledPatching tile size per spatial axis (image-pixel units).
    overlap : list of int
        TiledPatching overlap per spatial axis (image-pixel units).
    statistic : {"kl", "js", "ks", "wasserstein", "mean_abs_ratio"}, default="kl"
        Two-sample discrepancy statistic.
    strip_width : int, default=4
        Half-width ``N`` of the control strip around each seam.
    block_size : int, default=3
        Contiguous-block size ``B`` for the permutation engine.
    n_permutations : int, default=1000
        Number of permutations ``R`` per tile.
    alpha : float, default=0.05
        Rejection threshold for the per-tile test.
    num_bins_per_tile : int, default=32
        Number of histogram bins for binned statistics (KL, JS).
    random_seed : int, default=0
        RNG seed.
    normalize_per_axis : bool, default=True
        Standardize gradients per axis by image-wide ``(mean, std)`` (seam+control
        pooled) so gradients from all axes can be pooled into one test.
    balance_axis_counts : bool, default=True
        Subsample per tile so every owned-seam axis contributes an equal number of
        blocks. Only statistically valid alongside ``normalize_per_axis``.
    channel : int, default=0
        Channel index to analyse.
    """

    tile_size: list[PositiveInt]
    overlap: list[NonNegativeInt]
    statistic: Statistic = "kl"
    strip_width: int = Field(default=4, ge=1)
    block_size: int = Field(default=3, ge=1)
    n_permutations: int = Field(default=1000, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    num_bins_per_tile: int = Field(default=32, ge=2)
    random_seed: int = 0
    normalize_per_axis: bool = True
    balance_axis_counts: bool = True
    channel: int = Field(default=0, ge=0)

    @field_validator("n_permutations")
    @classmethod
    def _warn_low_n_permutations(cls, v: int) -> int:
        """Warn if ``n_permutations`` is too low for usable p-value resolution.

        Parameters
        ----------
        v : int
            Candidate ``n_permutations`` value.

        Returns
        -------
        int
            The validated value (unchanged).
        """
        if v < 100:
            warnings.warn(
                f"n_permutations={v} is below the recommended 100; "
                "p-value resolution will be poor.",
                stacklevel=2,
            )
        return v

    @model_validator(mode="after")
    def _check_axis_consistency(self) -> Self:
        """Cross-field validation of axis lengths and step / strip-width geometry.

        Returns
        -------
        Self
            The validated ``GradientTestConfig`` instance.

        Raises
        ------
        ValueError
            If ``tile_size`` and ``overlap`` have different lengths, if any
            ``overlap[i] >= tile_size[i]``, or if any axis step is too small
            for ``2 * strip_width + 2``.
        """
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
