"""File I/O utilities."""

from pathlib import Path
from typing import Union
import pickle

import numpy as np
import tifffile as tiff


def load_prediction(path: Union[str, Path]) -> np.ndarray:
    """
    Load prediction from file (.pkl, .dill, or .tiff).

    Args:
        path: Path to prediction file

    Returns:
        Loaded prediction array

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix in [".pkl", ".dill"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.suffix in [".tif", ".tiff"]:
        # TIFF expected shape: (N, C, H, W) → transpose to (N, H, W, C)
        return tiff.imread(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            "Supported formats: .pkl, .dill, .tif, .tiff"
        )
