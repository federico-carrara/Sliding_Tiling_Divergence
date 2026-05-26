"""Metrics computation utilities."""

from typing import List
import numpy as np


# TODO: deprecated dead-code, kept only for keeping ideas around

def wiener_entropy(hist: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Wiener entropy (spectral-flatness-like) of a histogram.

    Parameters
    ----------
    hist : np.ndarray
        Input histogram array.
    eps : float, default=1e-12
        Small epsilon to avoid division by zero.

    Returns
    -------
    float
        Wiener entropy value.
    """
    hist = np.asarray(hist).astype(float)
    if hist.size == 0:
        return 0.0

    w = np.hanning(len(hist))
    X = np.fft.rfft(hist * w)
    P = np.abs(X) ** 2 + eps
    geom_mean = np.exp(np.mean(np.log(P)))
    arith_mean = np.mean(P)
    return 1.0 - float(geom_mean / (arith_mean + eps))


def get_peakiness_scores(
    histogram_edges: np.ndarray,
    histogram_middle: np.ndarray,
    eps: float = 1e-12,
) -> List[float]:
    """Compute Wiener entropy peakiness scores for paired histograms.

    Computes scores for the edges histogram, the middle histogram, and their
    difference (middle - edges).

    Parameters
    ----------
    histogram_edges : np.ndarray
        Histogram of edge gradients.
    histogram_middle : np.ndarray
        Histogram of middle gradients.
    eps : float, default=1e-12
        Small epsilon to avoid division by zero.

    Returns
    -------
    list of float
        Three Wiener-entropy peakiness scores ``[edges, middle, middle - edges]``.
    """
    scores = []
    for x in [histogram_edges, histogram_middle, histogram_middle - histogram_edges]:
        scores.append(wiener_entropy(x, eps=eps))
    return scores


def compute_peakiness(hist: np.ndarray, eps: float = 1e-12) -> float:
    """Compute a histogram 'peakiness' score (lower is smoother).

    Defined as the sum of the top 10% of bin masses after normalization.

    Parameters
    ----------
    hist : np.ndarray
        Input histogram.
    eps : float, default=1e-12
        Small epsilon for normalization.

    Returns
    -------
    float
        Peakiness score.
    """
    hist = normalize_histogram(hist, eps=eps)
    sorted_vals = np.sort(hist)[::-1]
    top_frac = int(0.1 * len(sorted_vals))
    return np.sum(sorted_vals[:top_frac])


def normalize_histogram(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a histogram so its entries sum to 1.

    Parameters
    ----------
    arr : np.ndarray
        Input histogram.
    eps : float, default=1e-12
        Small epsilon to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized histogram.
    """
    arr = np.asarray(arr, dtype=float)
    return arr / (arr.sum() + eps)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Kullback–Leibler divergence ``KL(p || q)``.

    Parameters
    ----------
    p : np.ndarray
        First probability distribution (histogram).
    q : np.ndarray
        Second probability distribution (histogram).
    eps : float, default=1e-12
        Small epsilon to avoid ``log(0)``.

    Returns
    -------
    float
        KL divergence value.
    """
    p = normalize_histogram(p, eps)
    q = normalize_histogram(q, eps)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def compute_kl_matrix(histograms: List[np.ndarray]) -> np.ndarray:
    """Compute the pairwise KL divergence matrix for a list of histograms.

    Parameters
    ----------
    histograms : list of np.ndarray
        Histograms to compare pairwise.

    Returns
    -------
    np.ndarray
        ``(N, N)`` matrix where element ``(i, j)`` is ``KL(hist_i || hist_j)``
        (diagonal is zero).
    """
    n = len(histograms)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_mat[i, j] = kl_divergence(histograms[i], histograms[j])
    return kl_mat
