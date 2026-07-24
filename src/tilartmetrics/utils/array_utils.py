"""Array manipulation utilities."""

import numpy as np


def ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Ensure array has 4 dimensions in channel-first layout ``(N, C, H, W)``.

    A 3-D input is interpreted as ``(N, H, W)`` (single-channel sample stack)
    and gets a singleton ``C`` dimension inserted at position 1.

    Parameters
    ----------
    arr : np.ndarray
        Input array with 3 or 4 dimensions.

    Returns
    -------
    np.ndarray
        Array with 4 dimensions in ``(N, C, H, W)`` layout.
    """
    if arr.ndim == 3:
        return arr[:, np.newaxis, :, :]
    return arr
