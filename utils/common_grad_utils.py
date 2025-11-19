import numpy as np

def wiener_entropy(hist: np.ndarray, eps=1e-12):
    """
    Compute Wiener entropy (spectral flatness-like) for a histogram.
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


def get_peakiness_scores(histogram_edges: np.ndarray, histogram_middle: np.ndarray, eps=1e-12):
    """
    Compute Wiener entropy peakiness scores for:
      - edges histogram
      - middle histogram
      - difference (middle - edges)
    Returns: list of three scores
    """
    scores = []
    for x in [histogram_edges, histogram_middle, histogram_middle - histogram_edges]:
        scores.append(wiener_entropy(x, eps=eps))
    return scores


def compute_histograms(gradients: np.ndarray, bin_edges: np.ndarray):
    """
    Compute histogram counts for `gradients` using `bin_edges`.
    """
    grads = np.asarray(gradients).flatten()
    hist = np.histogram(grads, bins=bin_edges)[0]
    return hist


def get_bin_edges(gradients_list: list, num_bins=200):
    """
    Make bin edges from a list of gradient arrays (raw).
    """
    flattened = np.concatenate([np.asarray(g).flatten() for g in gradients_list])
    _, bin_edges = np.histogram(flattened, bins=num_bins)
    return bin_edges
