"""Block-size calibration for the per-tile permutation test.

Given a known-artifact-free reference (ground truth or a non-tiled
prediction from the same modality), scan candidate ``block_size`` values,
measure ``frac_rejected`` on each, and pick the smallest ``B`` that
controls Type I error at the desired ``alpha`` (within a small tolerance).

Calibration is a once-per-dataset / per-tile-geometry step; the recommended
``B`` is then passed to ``analyze-experiment --block_size <B>`` on the test
set. No changes to the main pipeline — this module sits next to it and
reuses :func:`per_image_tile_scan` as the only computational primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .per_tile import per_image_tile_scan


@dataclass
class CandidateResult:
    """Aggregated ``frac_rejected`` for one candidate block size.

    Attributes
    ----------
    block_size : int
        Candidate block size ``B``.
    frac_rejected_mean : float
        Mean per-image ``frac_rejected`` across seeds and images.
    frac_rejected_sd : float
        Standard deviation across seeds; ``0.0`` when ``n_seeds == 1``.
    n_tiles_total : int
        Total number of valid tiles counted across seeds and images.
    """

    block_size: int
    frac_rejected_mean: float
    frac_rejected_sd: float
    n_tiles_total: int


@dataclass
class CalibrationReport:
    """Top-level output of :func:`calibrate_block_size`.

    Attributes
    ----------
    candidates : list of CandidateResult
        Per-candidate aggregated metrics, in input order.
    recommended_block_size : int, optional
        Smallest candidate ``B`` with ``frac_rejected_mean <= alpha + tolerance``,
        or ``None`` if no candidate passed.
    alpha : float
        Target Type I error rate.
    tolerance : float
        Slack on the selection rule.
    config_summary : dict
        Snapshot of the calibration configuration.
    """

    candidates: list[CandidateResult]
    recommended_block_size: Optional[int]
    alpha: float
    tolerance: float
    config_summary: dict = field(default_factory=dict)


def calibrate_block_size(
    reference_images: Sequence[np.ndarray],
    *,
    tile_size: Sequence[int],
    overlap: Sequence[int],
    strip_width: int = 4,
    statistic: str = "kl",
    n_permutations: int = 1000,
    num_bins_per_tile: int = 32,
    alpha: float = 0.05,
    tolerance: float = 0.01,
    candidate_block_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    n_seeds: int = 1,
    base_seed: int = 0,
    verbose: bool = True,
) -> CalibrationReport:
    """Calibrate ``block_size`` on a known-H₀ reference.

    Scans ``candidate_block_sizes`` and returns the smallest ``B`` with
    ``frac_rejected_mean <= alpha + tolerance``. ``tile_size``, ``overlap``,
    ``strip_width``, ``statistic``, ``num_bins_per_tile``, and
    ``n_permutations`` must match the test-time configuration exactly so
    calibration speaks to the same null distribution the test will use.

    With ``n_seeds == 1`` the RNG is ``default_rng(base_seed)``; with
    ``n_seeds > 1`` we average ``frac_rejected`` across runs seeded
    ``base_seed + i`` for ``i in range(n_seeds)``. Multi-seed averaging is
    only useful on small references where binomial noise dominates.

    Parameters
    ----------
    reference_images : Sequence[np.ndarray]
        Single-channel slices ``(H, W)`` or ``(D, H, W)`` — the caller is
        responsible for slicing batch and channel axes out of any loaded
        prediction.
    tile_size : Sequence[int]
        TiledPatching tile size per spatial axis.
    overlap : Sequence[int]
        TiledPatching overlap per spatial axis.
    strip_width : int, default=4
        Half-width ``N`` of the control strip around each seam.
    statistic : str, default="kl"
        Two-sample discrepancy statistic name.
    n_permutations : int, default=1000
        Permutations per tile.
    num_bins_per_tile : int, default=32
        Histogram bin count for binned statistics (KL, JS).
    alpha : float, default=0.05
        Target Type I error rate.
    tolerance : float, default=0.01
        Slack on the selection rule.
    candidate_block_sizes : Sequence[int], default=(1, 2, 4, 8, 16)
        Candidate ``B`` values to evaluate.
    n_seeds : int, default=1
        RNG seeds to average over.
    base_seed : int, default=0
        First RNG seed.
    verbose : bool, default=True
        If True, print a one-line summary per candidate as the scan runs.

    Returns
    -------
    CalibrationReport
        Per-candidate aggregated metrics plus the recommended ``B``.

    Raises
    ------
    ValueError
        If inputs are empty, ``n_seeds < 1``, or ``alpha + tolerance``
        falls outside ``(0, 1)``.
    """
    if not reference_images:
        raise ValueError("reference_images is empty")
    if not candidate_block_sizes:
        raise ValueError("candidate_block_sizes is empty")
    if not (0.0 < alpha + tolerance < 1.0):
        raise ValueError(
            f"alpha + tolerance = {alpha + tolerance} must lie in (0, 1)"
        )
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    threshold = alpha + tolerance
    candidates: list[CandidateResult] = []

    for block_size in candidate_block_sizes:
        fracs: list[float] = []
        n_tiles_total = 0
        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed_idx)
            for img in reference_images:
                ir = per_image_tile_scan(
                    img,
                    tile_size=tile_size,
                    overlap=overlap,
                    strip_width=strip_width,
                    block_size=block_size,
                    n_permutations=n_permutations,
                    statistic=statistic,
                    alpha=alpha,
                    num_bins_per_tile=num_bins_per_tile,
                    rng=rng,
                )
                if not np.isnan(ir.frac_rejected):
                    fracs.append(ir.frac_rejected)
                n_tiles_total += sum(1 for t in ir.tiles if not np.isnan(t.p))

        if not fracs:
            mean = float("nan")
            sd = float("nan")
        else:
            arr = np.array(fracs, dtype=np.float64)
            mean = float(arr.mean())
            sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        candidates.append(
            CandidateResult(
                block_size=int(block_size),
                frac_rejected_mean=mean,
                frac_rejected_sd=sd,
                n_tiles_total=n_tiles_total,
            )
        )

        if verbose:
            mark = "PASS" if mean <= threshold else "FAIL"
            print(
                f"  B={block_size:>3d}  frac_rejected={mean:.4f} (sd={sd:.4f}) "
                f"n_tiles={n_tiles_total}  {mark}"
            )

    passing_bs = [c.block_size for c in candidates if c.frac_rejected_mean <= threshold]
    recommended = min(passing_bs) if passing_bs else None

    return CalibrationReport(
        candidates=candidates,
        recommended_block_size=recommended,
        alpha=alpha,
        tolerance=tolerance,
        config_summary={
            "tile_size": list(tile_size),
            "overlap": list(overlap),
            "strip_width": strip_width,
            "statistic": statistic,
            "n_permutations": n_permutations,
            "num_bins_per_tile": num_bins_per_tile,
            "n_seeds": n_seeds,
            "base_seed": base_seed,
            "candidate_block_sizes": list(candidate_block_sizes),
        },
    )
