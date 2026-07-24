"""File I/O utilities."""

from pathlib import Path
from typing import Iterator, Optional, Union
import pickle

import numpy as np
import tifffile as tiff


def ensure_channel_first(arr: np.ndarray, n_spatial: int) -> np.ndarray:
    """Squeeze an array to channel-first ``(C, *spatial)`` for a given ndim.

    ``n_spatial`` is 2 for 2-D ``(C, H, W)`` or 3 for 3-D ``(C, D, H, W)``.
    A bare spatial array with no channel axis (``(H, W)`` / ``(D, H, W)``) is
    promoted to a single channel.

    Parameters
    ----------
    arr : np.ndarray
        Array to normalise; squeezed before the channel check.
    n_spatial : int
        Number of spatial axes (2 or 3).

    Returns
    -------
    np.ndarray
        Channel-first array with ``n_spatial + 1`` dimensions.

    Raises
    ------
    ValueError
        If the squeezed array is neither ``n_spatial`` nor ``n_spatial + 1``
        dimensional.
    """
    arr = np.asarray(arr).squeeze()
    if arr.ndim == n_spatial:
        arr = arr[np.newaxis, ...]
    if arr.ndim != n_spatial + 1:
        raise ValueError(
            f"expected {n_spatial + 1}-D channel-first array after squeeze "
            f"(n_spatial={n_spatial}), got shape {arr.shape}"
        )
    return arr


def read_image_names(
    npz_path: Union[str, Path], max_images: Optional[int] = None
) -> list[str]:
    """Return the image names (keys) in an ``.npz`` archive.

    Reads only the archive index, not the arrays, so this is cheap even for
    large archives. Optionally caps the list to the first ``max_images`` keys.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to a ``.npz`` archive whose keys are image names.
    max_images : int, optional
        If given, keep only the first ``max_images`` names (for quick trials).

    Returns
    -------
    list of str
        Image names in archive order.
    """
    names = list(np.load(npz_path, allow_pickle=True).files)
    return names if max_images is None else names[:max_images]


def iter_npz_images(
    npz_path: Union[str, Path], image_names: list[str], n_spatial: int
) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield ``(name, (C, *spatial))`` arrays from an ``.npz`` archive.

    ``.npz`` archives decompress each array only on access, so this keeps just
    one image in memory at a time. Each array is normalised to channel-first
    layout via :func:`ensure_channel_first`.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to a ``.npz`` archive whose keys are image names and whose arrays
        squeeze to channel-first ``(C, H, W)`` / ``(C, D, H, W)``.
    image_names : list of str
        Keys to yield, in the desired order.
    n_spatial : int
        Number of spatial axes (2 or 3), selecting the expected layout.

    Yields
    ------
    tuple of (str, np.ndarray)
        ``(name, channel_first_array)`` for each requested key.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        for name in image_names:
            yield name, ensure_channel_first(data[name], n_spatial)


def load_prediction(path: Union[str, Path]) -> np.ndarray:
    """Load a prediction array from file (.pkl, .dill, or .tiff).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to prediction file.

    Returns
    -------
    np.ndarray
        Loaded prediction array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not one of ``.pkl``, ``.dill``, ``.tif``, ``.tiff``.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix in [".pkl", ".dill"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix in [".tif", ".tiff"]:
        # TIFF stored in channel-first convention: (N, C, Z, Y, X)
        return tiff.imread(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Supported formats: .pkl, .dill, .tif, .tiff"
        )
