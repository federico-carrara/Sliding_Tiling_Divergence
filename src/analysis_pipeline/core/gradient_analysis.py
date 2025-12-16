"""Gradient computation and analysis utilities."""

from typing import Optional, Union, List, Tuple
import numpy as np


class GradientUtils:
    """
    Base class for computing gradient-based tiling statistics, histograms,
    and peakiness scores.
    """

    def __init__(
        self,
        imgs: np.ndarray,
        tile_size: Union[int, List[int], Tuple[int, ...], np.ndarray],
        border_size: Optional[Union[int, List[int]]] = None,
        bin_edges: Optional[np.ndarray] = None,
        channel: Optional[int] = None,
    ):
        """
        Initialize gradient utilities.

        Args:
            imgs: Input images array
            tile_size: Size of tiles for gradient computation
            border_size: Border size to remove before processing
            bin_edges: Pre-computed bin edges for histograms
            channel: Specific channel to analyze (None for all channels)
        """
        self.imgs = imgs
        self.tile_size = (
            np.array(tile_size)
            if isinstance(tile_size, (tuple, list, np.ndarray))
            else tile_size
        )
        self.border_size = border_size
        self._bin_edges = bin_edges

        # Remove borders
        self.imgs_wo_borders = self.border_free(self.imgs, self.border_size)

        # Compute gradients along each axis
        self.gradients = self.compute_gradients(self.imgs_wo_borders, self.border_size)

        # Gradients along tile grid
        self.grad_edges = self.get_gradients_at(position="edge", channels=channel)
        self.grad_middle = self.get_gradients_at(position="middle", channels=channel)

        self.mean = self.grad_middle.mean()
        self.std = self.grad_middle.std()

        # Bin edges
        if self._bin_edges is None:
            self._bin_edges = self.get_bin_edges(
                list(self.gradients) + [self.grad_edges, self.grad_middle]
            )

    # ---------------- STATIC METHODS ----------------

    @staticmethod
    def get_bin_edges(gradients: list, num_bins: int = 200) -> np.ndarray:
        """
        Compute bin edges from a list of gradient arrays.

        Args:
            gradients: List of gradient arrays
            num_bins: Number of histogram bins

        Returns:
            Bin edges array
        """
        flattened = np.concatenate([g.flatten() for g in gradients])
        _, bin_edges = np.histogram(flattened, bins=num_bins)
        return bin_edges

    @staticmethod
    def compute_histograms(
        gradients: np.ndarray, bin_edges: np.ndarray
    ) -> np.ndarray:
        """
        Compute histogram from gradients using given bin edges.

        Args:
            gradients: Gradient array
            bin_edges: Histogram bin edges

        Returns:
            Histogram counts
        """
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def wiener_entropy(hist: np.ndarray, eps: float = 1e-12) -> float:
        """
        Compute Wiener entropy (spectral flatness) for a histogram.

        Args:
            hist: Histogram array
            eps: Small epsilon to avoid division by zero

        Returns:
            Wiener entropy value
        """
        w = np.hanning(len(hist))
        X = np.fft.rfft(hist * w)
        P = np.abs(X) ** 2 + eps
        geom_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        return 1.0 - float(geom_mean / (arith_mean + eps))

    # ----------------- PUBLIC METHODS -----------------

    def make_bin_edges(self, n_bins: int = 2000) -> np.ndarray:
        """
        Create bin edges for histograms.

        Args:
            n_bins: Number of bins

        Returns:
            Bin edges array
        """
        return self.get_bin_edges(
            list(self.gradients) + [self.grad_edges, self.grad_middle],
            num_bins=n_bins,
        )

    @staticmethod
    def get_peakiness_scores(
        histogram_edges: np.ndarray,
        histogram_middle: np.ndarray,
        eps: float = 1e-12,
    ) -> List[float]:
        """
        Compute peakiness scores using Wiener entropy.

        Args:
            histogram_edges: Histogram of edge gradients
            histogram_middle: Histogram of middle gradients
            eps: Small epsilon value

        Returns:
            List of three peakiness scores
        """
        scores = []
        for x in [
            histogram_edges,
            histogram_middle,
            histogram_middle - histogram_edges,
        ]:
            scores.append(GradientUtils.wiener_entropy(x, eps=eps))
        return scores

    def _normalize_gradients(
        self,
        gradients: np.ndarray,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize gradients using z-score normalization.

        Args:
            gradients: Array to normalize
            mu: Mean to use for normalization (defaults to self.mean)
            sigma: Std to use for normalization (defaults to self.std)

        Returns:
            Normalized array
        """
        if mu is None:
            mu = self.mean
        if sigma is None:
            sigma = self.std

        return (gradients - mu) / (sigma + 1e-8)

    # Abstract methods to be implemented by subclasses
    @staticmethod
    def border_free(imgs: np.ndarray, border_size: Optional[Union[int, List[int]]]):
        """Remove borders from images. Must be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def compute_gradients(
        imgs: np.ndarray, border_size: Optional[Union[int, List[int]]] = 0
    ):
        """Compute gradients. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_gradients_at(
        self, position: Union[str, int], channels: Optional[int] = None
    ):
        """Get gradients at specific position. Must be implemented by subclasses."""
        raise NotImplementedError


# ----------------- 2D IMPLEMENTATION -----------------


class GradientUtils2D(GradientUtils):
    """2D implementation of gradient utilities."""

    @staticmethod
    def border_free(
        imgs: np.ndarray, border_size: Optional[Union[int, List[int]]]
    ) -> np.ndarray:
        """
        Remove borders from 2D images.

        Args:
            imgs: Input images (N, H, W, C)
            border_size: Border size to remove

        Returns:
            Images with borders removed
        """
        if border_size is None or border_size == 0:
            return imgs
        return imgs[:, border_size:-border_size, border_size:-border_size, :]

    @staticmethod
    def compute_gradients(
        imgs: np.ndarray, border_size: Optional[Union[int, List[int]]] = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D gradients (horizontal and vertical).

        Args:
            imgs: Input images (N, H, W, C)
            border_size: Border size (unused in computation)

        Returns:
            Tuple of (grad_x, grad_y)
        """
        grad_x = imgs[:, :, 1:, :] - imgs[:, :, :-1, :]
        grad_y = imgs[:, 1:, :, :] - imgs[:, :-1, :, :]
        return grad_x, grad_y

    def _gradients_along_tile_grid(
        self, offset: Union[int, Tuple[int, int]], channels: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract gradients along tile grid at given offset.

        Args:
            offset: Offset within tile
            channels: Channel to extract (None for all)

        Returns:
            Flattened gradient array
        """
        oy, ox = (
            offset if isinstance(offset, (tuple, list, np.ndarray)) else (offset, offset)
        )
        tile_sz_y, tile_sz_x = (
            self.tile_size
            if isinstance(self.tile_size, (tuple, list, np.ndarray))
            else (self.tile_size, self.tile_size)
        )
        grad_x, grad_y = self.gradients

        if channels is None:
            grad_x_slice = grad_x[:, :, ox::tile_sz_x, :]
            grad_y_slice = grad_y[:, oy::tile_sz_y, :, :]
        elif isinstance(channels, int):
            grad_x_slice = grad_x[:, :, ox::tile_sz_x, channels]
            grad_y_slice = grad_y[:, oy::tile_sz_y, :, channels]
        else:
            raise ValueError("channels must be None or int")

        return np.concatenate([grad_x_slice.flatten(), grad_y_slice.flatten()])

    def get_gradients_at(
        self, position: Union[str, int] = "edge", channels: Optional[int] = None
    ) -> np.ndarray:
        """
        Get gradients at specific position (edge, middle, or custom offset).

        Args:
            position: Position identifier ('edge', 'middle', or integer offset)
            channels: Channel to extract

        Returns:
            Gradient array at specified position
        """
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge' or 'middle'")
        elif isinstance(position, int):
            offset = position
        else:
            raise TypeError("position must be string or int")

        return self._gradients_along_tile_grid(offset, channels)


# ----------------- 3D IMPLEMENTATION -----------------


class GradientUtils3D(GradientUtils):
    """3D implementation of gradient utilities."""

    @staticmethod
    def border_free(
        imgs: np.ndarray, border_size: Union[int, List[int]]
    ) -> np.ndarray:
        """
        Remove borders from 3D images.

        Args:
            imgs: Input images (N, D, H, W, C)
            border_size: Border size [z, y, x]

        Returns:
            Images with borders removed
        """
        if isinstance(border_size, int):
            border_size = [border_size, border_size, border_size]

        bz, by, bx = border_size
        z_slice = slice(bz, -bz if bz != 0 else None)
        y_slice = slice(by, -by if by != 0 else None)
        x_slice = slice(bx, -bx if bx != 0 else None)
        return imgs[:, z_slice, y_slice, x_slice, :]

    @staticmethod
    def compute_gradients(
        imgs: np.ndarray, border_size: Union[int, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 3D gradients (along z, y, x).

        Args:
            imgs: Input images (N, D, H, W, C)
            border_size: Border size

        Returns:
            Tuple of (grad_z, grad_y, grad_x)
        """
        wb = GradientUtils3D.border_free(imgs, border_size)
        grad_z = wb[:, 1:, :, :, :] - wb[:, :-1, :, :, :]
        grad_y = wb[:, :, 1:, :, :] - wb[:, :, :-1, :, :]
        grad_x = wb[:, :, :, 1:, :] - wb[:, :, :, :-1, :]
        return grad_z, grad_y, grad_x

    def _gradients_along_tile_grid(
        self,
        offset: Union[int, List[int], np.ndarray],
        channels: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract gradients along 3D tile grid at given offset.

        Args:
            offset: Offset [z, y, x] within tile
            channels: Channel to extract (None for all)

        Returns:
            Flattened gradient array
        """
        oz, oy, ox = offset

        if channels is None:
            channels = list(range(self.gradients[2].shape[-1]))
        elif isinstance(channels, int):
            channels = [channels]

        grad_z, grad_y, grad_x = self.gradients
        grad_x_slice = grad_x[:, :, :, ox :: self.tile_size[2], channels]
        grad_y_slice = grad_y[:, :, oy :: self.tile_size[1], :, channels]
        grad_z_slice = grad_z[:, oz :: self.tile_size[0], :, :, channels]

        return np.concatenate(
            [grad_x_slice.flatten(), grad_y_slice.flatten(), grad_z_slice.flatten()]
        )

    def get_gradients_at(
        self,
        position: Union[str, int, List[int]] = "edge",
        channels: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get gradients at specific position in 3D.

        Args:
            position: Position identifier ('edge', 'middle', or [z, y, x] offset)
            channels: Channel to extract

        Returns:
            Gradient array at specified position
        """
        if isinstance(position, str):
            position = position.lower()
            if position == "edge":
                offset = self.tile_size - 1
            elif position == "middle":
                offset = self.tile_size // 2 - 1
            else:
                raise ValueError("position must be 'edge' or 'middle'")
        elif isinstance(position, int):
            offset = np.array([position] * 3)
        else:
            offset = np.array(position)

        return self._gradients_along_tile_grid(offset, channels)
