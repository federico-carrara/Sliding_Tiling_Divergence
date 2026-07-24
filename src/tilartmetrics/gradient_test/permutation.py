"""Block permutation engine for the per-tile two-sample test.

Each input slice (one seam-line slice, or one control strip) is partitioned
into contiguous blocks of size ``B`` along its (raveled) order. Trailing
partial blocks are kept to preserve all data — important at small tile sizes
where the parallel range is barely longer than a few blocks.

Block labels (seam vs. control) are then permuted ``R`` times while keeping
the total seam-block / control-block counts fixed; for each permutation the
two-sample statistic is recomputed and the p-value is the Phipson–Smyth
``(1 + #{T_null >= T_obs}) / (1 + R)`` to avoid a hard zero.

Vectorized fast paths:

- ``binned`` (KL, JS) — per-block histogram contributions are pre-computed
  on per-tile joint bin edges, then summed under each permutation.
- ``abs_ratio`` (mean-abs-ratio) — per-block absolute sum and length are
  pre-computed and summed.

Other statistics fall back to a Python loop calling the registered
``stat_spec.fn`` on the permuted samples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tilartmetrics.gradient_test.statistics import StatisticSpec


EPS = 1e-12

def _split_into_blocks(
    slices: list[np.ndarray], block_size: int
) -> list[np.ndarray]:
    """Split each input slice into contiguous blocks of length ``block_size``.

    Empty slices are skipped. The trailing partial block of each slice is
    kept to preserve all data; blocks never span slice boundaries.

    Parameters
    ----------
    slices : list of np.ndarray
        1-D arrays to split (e.g. per-seam or per-strip gradient samples).
    block_size : int
        Target block length ``B``.

    Returns
    -------
    list of np.ndarray
        Concatenated list of blocks across all input slices.
    """
    blocks: list[np.ndarray] = []
    for s in slices:
        if s.size == 0:
            continue
        for start in range(0, s.size, block_size):
            blocks.append(s[start : start + block_size])
    return blocks


def _build_permutations(
    n_blocks: int, n_permutations: int, rng: np.random.Generator
) -> np.ndarray:
    """Build a row-wise random permutation matrix.

    Parameters
    ----------
    n_blocks : int
        Number of block indices to permute per row.
    n_permutations : int
        Number of permutations ``R``.
    rng : numpy.random.Generator
        Random generator used for the permutation.

    Returns
    -------
    np.ndarray
        ``(R, n_blocks)`` integer matrix of permuted block indices.
    """
    base = np.tile(np.arange(n_blocks), (n_permutations, 1))
    return rng.permuted(base, axis=1)


def _binned_path(
    all_blocks: list[np.ndarray],
    n_seam_blocks: int,
    perms: np.ndarray,
    *,
    name: str,
    num_bins: int,
) -> tuple[float, np.ndarray]:
    """Vectorized binned-statistic path (KL or JS) on per-tile joint bin edges.

    Parameters
    ----------
    all_blocks : list of np.ndarray
        Concatenated seam blocks followed by control blocks.
    n_seam_blocks : int
        Number of leading seam blocks in ``all_blocks``.
    perms : np.ndarray
        ``(R, n_blocks)`` permutation index matrix.
    name : {"kl", "js"}
        Which binned statistic to compute.
    num_bins : int
        Histogram bin count for the joint per-tile edges.

    Returns
    -------
    T_obs : float
        Observed statistic value on the unpermuted split.
    T_null : np.ndarray
        ``(R,)`` array of permutation-null statistic values.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported binned statistics.
    """
    all_values = np.concatenate(all_blocks)
    bin_edges = np.histogram_bin_edges(all_values, bins=num_bins)

    n_blocks = len(all_blocks)
    n_bins = len(bin_edges) - 1
    block_hists = np.empty((n_blocks, n_bins), dtype=np.float64)
    block_lengths = np.empty(n_blocks, dtype=np.int64)
    for i, b in enumerate(all_blocks):
        block_hists[i] = np.histogram(b, bins=bin_edges)[0]
        block_lengths[i] = b.size

    def _from_blocks(seam_block_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate seam and control probability histograms for one mask.

        Accepts a 1-D index for the observed call and a 2-D ``(R, k)`` index
        for permutations.

        Parameters
        ----------
        seam_block_idx : np.ndarray
            1-D ``(k,)`` or 2-D ``(R, k)`` block indices assigned to "seam".

        Returns
        -------
        p_s : np.ndarray
            Seam probability histograms of shape ``(.., n_bins)``.
        p_c : np.ndarray
            Control probability histograms of shape ``(.., n_bins)``.
        """
        seam_hist = block_hists[seam_block_idx].sum(axis=-2)
        control_hist = block_hists.sum(axis=0) - seam_hist
        n_s = block_lengths[seam_block_idx].sum(axis=-1)
        n_c = block_lengths.sum() - n_s
        p_s = seam_hist / (n_s[..., None] + EPS)
        p_c = control_hist / (n_c[..., None] + EPS)
        return p_s, p_c

    seam_idx_obs = np.arange(n_seam_blocks)
    p_s_obs, p_c_obs = _from_blocks(seam_idx_obs)
    p_s_perm, p_c_perm = _from_blocks(perms[:, :n_seam_blocks])

    if name == "kl":
        T_obs = float(
            np.sum(p_s_obs * np.log((p_s_obs + EPS) / (p_c_obs + EPS)))
        )
        T_null = np.sum(
            p_s_perm * np.log((p_s_perm + EPS) / (p_c_perm + EPS)), axis=1
        )
    elif name == "js":
        m_obs = 0.5 * (p_s_obs + p_c_obs)
        T_obs = float(
            0.5 * np.sum(p_s_obs * np.log((p_s_obs + EPS) / (m_obs + EPS)))
            + 0.5 * np.sum(p_c_obs * np.log((p_c_obs + EPS) / (m_obs + EPS)))
        )
        m_perm = 0.5 * (p_s_perm + p_c_perm)
        T_null = 0.5 * np.sum(
            p_s_perm * np.log((p_s_perm + EPS) / (m_perm + EPS)), axis=1
        ) + 0.5 * np.sum(
            p_c_perm * np.log((p_c_perm + EPS) / (m_perm + EPS)), axis=1
        )
    else:
        raise ValueError(f"binned path got unexpected name {name!r}")
    return T_obs, T_null


def _abs_ratio_path(
    all_blocks: list[np.ndarray],
    n_seam_blocks: int,
    perms: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Vectorized fast path for the mean-abs-ratio statistic.

    Parameters
    ----------
    all_blocks : list of np.ndarray
        Concatenated seam blocks followed by control blocks.
    n_seam_blocks : int
        Number of leading seam blocks in ``all_blocks``.
    perms : np.ndarray
        ``(R, n_blocks)`` permutation index matrix.

    Returns
    -------
    T_obs : float
        Observed statistic value on the unpermuted split.
    T_null : np.ndarray
        ``(R,)`` array of permutation-null statistic values.
    """
    n_blocks = len(all_blocks)
    block_abs = np.array([float(np.abs(b).sum()) for b in all_blocks])
    block_len = np.array([b.size for b in all_blocks], dtype=np.int64)

    def _ratio(seam_block_idx: np.ndarray) -> np.ndarray:
        """Aggregate the mean-abs-ratio statistic for one block-label mask.

        Parameters
        ----------
        seam_block_idx : np.ndarray
            1-D ``(k,)`` or 2-D ``(R, k)`` block indices assigned to "seam".

        Returns
        -------
        np.ndarray
            Scalar (1-D input) or ``(R,)`` (2-D input) ratio
            ``mean(|seam|) / mean(|control|)``.
        """
        sum_s = block_abs[seam_block_idx].sum(axis=-1)
        sum_c = block_abs.sum() - sum_s
        n_s = block_len[seam_block_idx].sum(axis=-1)
        n_c = block_len.sum() - n_s
        return (sum_s / (n_s + EPS)) / ((sum_c / (n_c + EPS)) + EPS)

    T_obs = float(_ratio(np.arange(n_seam_blocks)))
    T_null = _ratio(perms[:, :n_seam_blocks])
    return T_obs, T_null


def _scalar_path(
    all_blocks: list[np.ndarray],
    n_seam_blocks: int,
    perms: np.ndarray,
    stat_spec: StatisticSpec,
    stat_kwargs: dict,
) -> tuple[float, np.ndarray]:
    """Fallback Python-loop path for statistics without a vectorized fast path.

    Parameters
    ----------
    all_blocks : list of np.ndarray
        Concatenated seam blocks followed by control blocks.
    n_seam_blocks : int
        Number of leading seam blocks in ``all_blocks``.
    perms : np.ndarray
        ``(R, n_blocks)`` permutation index matrix.
    stat_spec : StatisticSpec
        Statistic specification carrying the callable ``fn``.
    stat_kwargs : dict
        Extra keyword arguments forwarded to ``stat_spec.fn``.

    Returns
    -------
    T_obs : float
        Observed statistic value on the unpermuted split.
    T_null : np.ndarray
        ``(R,)`` array of permutation-null statistic values.
    """
    n_blocks = len(all_blocks)
    seam_obs = np.concatenate(all_blocks[:n_seam_blocks])
    control_obs = np.concatenate(all_blocks[n_seam_blocks:])
    T_obs = float(stat_spec.fn(seam_obs, control_obs, **stat_kwargs))

    n_permutations = perms.shape[0]
    T_null = np.empty(n_permutations, dtype=np.float64)
    for r in range(n_permutations):
        perm = perms[r]
        seam_p = np.concatenate([all_blocks[i] for i in perm[:n_seam_blocks]])
        control_p = np.concatenate([all_blocks[i] for i in perm[n_seam_blocks:]])
        T_null[r] = stat_spec.fn(seam_p, control_p, **stat_kwargs)
    return T_obs, T_null


def permutation_pvalue(
    seam_slices: list[np.ndarray],
    control_slices: list[np.ndarray],
    *,
    stat_spec: StatisticSpec,
    block_size: int,
    n_permutations: int,
    rng: np.random.Generator,
    stat_kwargs: Optional[dict] = None,
) -> tuple[float, float, np.ndarray]:
    """Run the block permutation test on a single tile.

    The p-value uses Phipson–Smyth
    ``p = (1 + #{T_null >= T_obs}) / (1 + R)`` to avoid a hard zero.

    If either side has zero non-empty blocks the test cannot run and the
    function returns ``(nan, nan, empty)`` — the orchestrator records these
    as skipped tiles.

    Parameters
    ----------
    seam_slices : list of np.ndarray
        Per-seam 1-D arrays of across-seam gradients.
    control_slices : list of np.ndarray
        Per-strip 1-D arrays of control gradients.
    stat_spec : StatisticSpec
        Statistic specification (name, callable, vectorization kind, defaults).
    block_size : int
        Contiguous-block size ``B`` for permutation.
    n_permutations : int
        Number of permutations ``R``.
    rng : numpy.random.Generator
        Random generator used to build the permutation matrix.
    stat_kwargs : dict, optional
        Extra keyword arguments forwarded to ``stat_spec.fn`` (merged with
        ``stat_spec.default_kwargs``).

    Returns
    -------
    T_obs : float
        Observed statistic value, or ``nan`` if the test was skipped.
    p : float
        Phipson–Smyth p-value, or ``nan`` if the test was skipped.
    T_null : np.ndarray
        ``(R,)`` permutation-null distribution, or an empty array if skipped.
    """
    stat_kwargs = dict(stat_kwargs or {})
    for k, v in stat_spec.default_kwargs.items():
        stat_kwargs.setdefault(k, v)

    seam_blocks = _split_into_blocks(seam_slices, block_size)
    control_blocks = _split_into_blocks(control_slices, block_size)
    if not seam_blocks or not control_blocks:
        return float("nan"), float("nan"), np.array([], dtype=np.float64)

    all_blocks = seam_blocks + control_blocks
    n_seam_blocks = len(seam_blocks)
    n_blocks = len(all_blocks)
    perms = _build_permutations(n_blocks, n_permutations, rng)

    if stat_spec.vec_kind == "binned":
        T_obs, T_null = _binned_path(
            all_blocks, n_seam_blocks, perms,
            name=stat_spec.name, num_bins=stat_kwargs["num_bins"],
        )
    elif stat_spec.vec_kind == "abs_ratio":
        T_obs, T_null = _abs_ratio_path(all_blocks, n_seam_blocks, perms)
    else:
        T_obs, T_null = _scalar_path(
            all_blocks, n_seam_blocks, perms, stat_spec, stat_kwargs
        )

    p = (1.0 + float(np.sum(T_null >= T_obs))) / (1.0 + n_permutations)
    return T_obs, p, T_null
