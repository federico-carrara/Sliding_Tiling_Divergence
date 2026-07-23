"""File I/O utilities."""

from pathlib import Path
from typing import Union
import pickle

import numpy as np
import tifffile as tiff


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
