"""Array manipulation utilities."""

from typing import Union
import numpy as np


def ensure_4d(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array has 4 dimensions (N, H, W, C).

    Args:
        arr: Input array with 3 or 4 dimensions

    Returns:
        Array with 4 dimensions
    """
    if len(arr.shape) == 3:
        return arr[..., np.newaxis]
    return arr


def remove_padding(
    arr: np.ndarray,
    pad: Union[int, list[int]]
) -> np.ndarray:
    """
    Remove padding from array.

    Supports both 4D (N, H, W, C) and 5D (N, D, H, W, C) arrays.

    Args:
        arr: Input array
        pad: Padding size to remove. If int, same padding for all dimensions.
             If list, per-dimension padding values.

    Returns:
        Array with padding removed
    """
    if pad == 0:
        return arr

    if len(arr.shape) == 4:  # (N, H, W, C)
        return arr[:, pad:-pad, pad:-pad, :]
    elif len(arr.shape) == 5:  # (N, D, H, W, C)
        if isinstance(pad, int):
            return arr[:, :, pad:-pad, pad:-pad, :]
        else:
            # Support per-dimension padding for 5D
            pd, ph, pw = pad if isinstance(pad, (list, tuple)) else (0, pad, pad)
            if pd == 0:
                return arr[:, :, ph:-ph, pw:-pw, :]
            return arr[:, pd:-pd, ph:-ph, pw:-pw, :]
    else:
        return arr
