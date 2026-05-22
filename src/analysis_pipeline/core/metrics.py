"""Metrics computation utilities."""

from typing import List
import numpy as np


# TODO: deprecated dead-code, kept only for keeping ideas around

def wiener_entropy(hist: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Wiener entropy (spectral flatness-like) for a histogram.

    Args:
        hist: Histogram array
        eps: Small epsilon to avoid division by zero

    Returns:
        Wiener entropy value
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
    """
    Compute Wiener entropy peakiness scores for histograms.

    Computes scores for:
      - edges histogram
      - middle histogram
      - difference (middle - edges)

    Args:
        histogram_edges: Histogram of edge gradients
        histogram_middle: Histogram of middle gradients
        eps: Small epsilon value

    Returns:
        List of three peakiness scores
    """
    scores = []
    for x in [histogram_edges, histogram_middle, histogram_middle - histogram_edges]:
        scores.append(wiener_entropy(x, eps=eps))
    return scores


def compute_peakiness(hist: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calculate histogram 'peakiness': sum of top 10% bin masses after normalization.

    Lower peakiness = better (smoother gradients).

    Args:
        hist: Input histogram
        eps: Small epsilon for normalization

    Returns:
        Peakiness score
    """
    hist = normalize_histogram(hist, eps=eps)
    sorted_vals = np.sort(hist)[::-1]
    top_frac = int(0.1 * len(sorted_vals))
    return np.sum(sorted_vals[:top_frac])


def normalize_histogram(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize histogram to sum to 1.

    Args:
        arr: Input histogram
        eps: Small epsilon to avoid division by zero

    Returns:
        Normalized histogram
    """
    arr = np.asarray(arr, dtype=float)
    return arr / (arr.sum() + eps)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Kullback-Leibler divergence KL(p || q).

    Args:
        p: First probability distribution (histogram)
        q: Second probability distribution (histogram)
        eps: Small epsilon to avoid log(0)

    Returns:
        KL divergence value
    """
    p = normalize_histogram(p, eps)
    q = normalize_histogram(q, eps)
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def compute_kl_matrix(histograms: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise KL divergence matrix for multiple histograms.

    Args:
        histograms: List of histograms

    Returns:
        NxN matrix where element (i,j) is KL(hist_i || hist_j)
    """
    n = len(histograms)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_mat[i, j] = kl_divergence(histograms[i], histograms[j])
    return kl_mat
