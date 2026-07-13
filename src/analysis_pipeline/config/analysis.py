"""Top-level analysis configurations (one per metric) and their CLI loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, model_validator

from analysis_pipeline.config.frc import FRCConfig
from analysis_pipeline.config.gradient import GradientTestConfig


class AnalysisConfig(BaseModel):
    """Top-level configuration consumed by the gradient-test CLI.

    Attributes
    ----------
    name : str
        Model name (used in the report header).
    dataset : str
        Dataset name.
    save_dir : pathlib.Path
        Directory where the JSON ``MultiMethodReport`` is written.
    predictions : list of str
        Prediction file paths (one per method).
    method_names : list of str
        Method names matching ``predictions`` one-to-one.
    gradient_test : GradientTestConfig
        Gradient-test (per-tile two-sample) parameters.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    dataset: str
    save_dir: Path
    predictions: list[str]
    method_names: list[str]
    gradient_test: GradientTestConfig

    @model_validator(mode="after")
    def _check_predictions_and_methods(self) -> Self:
        """Validate matching counts and the supported number of predictions.

        Returns
        -------
        Self
            The validated ``AnalysisConfig`` instance.

        Raises
        ------
        ValueError
            If ``predictions`` and ``method_names`` have different lengths,
            or the number of predictions is outside the supported ``[1, 5]`` range.
        """
        if len(self.predictions) != len(self.method_names):
            raise ValueError(
                f"Number of predictions ({len(self.predictions)}) must match "
                f"method names ({len(self.method_names)})"
            )
        return self


class FRCAnalysisConfig(BaseModel):
    """Top-level configuration for the FRC CLI.

    Attributes
    ----------
    name : str
        Model name (used in the report header).
    dataset : str
        Dataset name.
    save_dir : pathlib.Path
        Directory where the JSON ``FRCMultiMethodReport`` is written.
    predictions : list of str
        Prediction file paths (one per method).
    ground_truths : list of str
        Ground-truth file paths matching ``predictions`` one-to-one.
    method_names : list of str
        Method names matching ``predictions`` one-to-one.
    tile_inner_sizes : list of int or None, optional
        Inner-tile size per method, used only to draw expected-harmonic
        verticals on the headline plot. ``None`` for methods without a
        fixed seam grid (e.g. SWiTi). When provided, must match
        ``predictions`` length one-to-one.
    frc : FRCConfig
        FRC metric parameters.
    """

    model_config = ConfigDict(protected_namespaces=())

    name: str
    dataset: str
    save_dir: Path
    predictions: list[str]
    ground_truths: list[str]
    method_names: list[str]
    tile_inner_sizes: list[Optional[int]] | None = None
    frc: FRCConfig

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        """Validate parallel-list lengths and the supported number of methods.

        Returns
        -------
        Self
            The validated ``FRCAnalysisConfig`` instance.

        Raises
        ------
        ValueError
            If ``predictions``, ``ground_truths``, and ``method_names``
            disagree on length, the count falls outside ``[1, 5]``, or
            ``tile_inner_sizes`` (when given) does not match the methods
            length.
        """
        n = len(self.predictions)
        if len(self.ground_truths) != n:
            raise ValueError(
                f"Number of ground_truths ({len(self.ground_truths)}) must "
                f"match predictions ({n})"
            )
        if len(self.method_names) != n:
            raise ValueError(
                f"Number of method_names ({len(self.method_names)}) must "
                f"match predictions ({n})"
            )
        if self.tile_inner_sizes is not None and len(self.tile_inner_sizes) != n:
            raise ValueError(
                f"tile_inner_sizes has {len(self.tile_inner_sizes)} entries; "
                f"expected {n} to match the number of methods"
            )
        return self


def load_gradient_test_config_from_args(args) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from parsed CLI arguments.

    Expects ``args.tile_size`` and ``args.overlap`` to be flat per-axis
    integer lists (the multi-method per-prediction broadcasting is handled
    later inside ``run_gradient_analysis_multi``).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments produced by ``analyze``'s ``parse_args``.

    Returns
    -------
    AnalysisConfig
        Validated configuration ready to drive the pipeline.
    """
    pred_files = [p.strip() for p in args.predictions.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]

    gradient_test = GradientTestConfig(
        tile_size=list(args.tile_size),
        overlap=list(args.overlap),
        statistic=args.statistic,
        strip_width=args.strip_width,
        block_size=args.block_size,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        num_bins_per_tile=args.num_bins_per_tile,
        random_seed=args.random_seed,
        normalize_per_axis=args.normalize_per_axis,
        balance_axis_counts=args.balance_axis_counts,
        channel=args.channel,
    )

    return AnalysisConfig(
        name=args.model_name,
        dataset=args.dataset,
        save_dir=Path(args.save_dir),
        predictions=pred_files,
        method_names=method_names,
        gradient_test=gradient_test,
    )


def _parse_tile_inner_sizes(value: str | None) -> list[Optional[int]] | None:
    """Parse ``--tile_inner_sizes`` (CSV of ints or ``"none"`` tokens).

    Parameters
    ----------
    value : str or None
        Raw CLI string, e.g. ``"32,none,64"``, or ``None`` if the flag was
        not provided.

    Returns
    -------
    list of (int or None), or None
        Parsed list; ``None`` when the flag was not supplied.

    Raises
    ------
    ValueError
        If a non-``"none"`` token cannot be parsed as a positive integer.
    """
    if value is None:
        return None
    out: list[Optional[int]] = []
    for tok in value.split(","):
        t = tok.strip().lower()
        if t == "none" or t == "":
            out.append(None)
        else:
            try:
                iv = int(t)
            except ValueError as e:
                raise ValueError(
                    f"invalid tile_inner_sizes token {tok!r}: expected an int "
                    "or 'none'"
                ) from e
            if iv <= 0:
                raise ValueError(
                    f"tile_inner_sizes entries must be positive, got {iv}"
                )
            out.append(iv)
    return out


def load_frc_config_from_args(args) -> FRCAnalysisConfig:
    """Build an :class:`FRCAnalysisConfig` from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments produced by ``frc_analyze``'s ``parse_args``.

    Returns
    -------
    FRCAnalysisConfig
        Validated configuration ready to drive the FRC pipeline.
    """
    pred_files = [p.strip() for p in args.predictions.split(",")]
    gt_files = [p.strip() for p in args.ground_truths.split(",")]
    method_names = [m.strip() for m in args.method_names.split(",")]

    frc = FRCConfig(
        apply_window=args.apply_window,
        channel=args.channel,
    )

    return FRCAnalysisConfig(
        name=args.model_name,
        dataset=args.dataset,
        save_dir=Path(args.save_dir),
        predictions=pred_files,
        ground_truths=gt_files,
        method_names=method_names,
        tile_inner_sizes=_parse_tile_inner_sizes(args.tile_inner_sizes),
        frc=frc,
    )


def load_gradient_test_single_config_from_args(args) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from single-method CLI arguments.

    The singular ``--prediction`` / ``--method_name`` flags are wrapped into
    1-element lists so the same :class:`AnalysisConfig` is used for both the
    single- and multi-method CLIs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments produced by the single-method ``analyze``
        CLI's ``parse_args``.

    Returns
    -------
    AnalysisConfig
        Validated configuration ready to drive the pipeline.
    """
    gradient_test = GradientTestConfig(
        tile_size=list(args.tile_size),
        overlap=list(args.overlap),
        statistic=args.statistic,
        strip_width=args.strip_width,
        block_size=args.block_size,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        num_bins_per_tile=args.num_bins_per_tile,
        random_seed=args.random_seed,
        normalize_per_axis=args.normalize_per_axis,
        balance_axis_counts=args.balance_axis_counts,
        channel=args.channel,
    )

    return AnalysisConfig(
        name=args.model_name,
        dataset=args.dataset,
        save_dir=Path(args.save_dir),
        predictions=[args.prediction.strip()],
        method_names=[args.method_name.strip()],
        gradient_test=gradient_test,
    )


def load_frc_single_config_from_args(args) -> FRCAnalysisConfig:
    """Build an :class:`FRCAnalysisConfig` from single-method CLI arguments.

    The singular ``--prediction`` / ``--ground_truth`` / ``--method_name``
    flags are wrapped into 1-element lists so the same
    :class:`FRCAnalysisConfig` is used for both the single- and multi-method
    CLIs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments produced by the single-method ``frc_analyze``
        CLI's ``parse_args``.

    Returns
    -------
    FRCAnalysisConfig
        Validated configuration ready to drive the FRC pipeline.
    """
    frc = FRCConfig(
        apply_window=args.apply_window,
        channel=args.channel,
    )

    return FRCAnalysisConfig(
        name=args.model_name,
        dataset=args.dataset,
        save_dir=Path(args.save_dir),
        predictions=[args.prediction.strip()],
        ground_truths=[args.ground_truth.strip()],
        method_names=[args.method_name.strip()],
        frc=frc,
    )
