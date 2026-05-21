"""Gradient computation and seam-aligned sampling utilities.

``GradientUtils2D`` / ``GradientUtils3D`` are thin orchestrators that combine
the finite-difference gradient of a stitched prediction with the seam
locator in :mod:`analysis_pipeline.core.seams`. The seam locator turns the
patching parameters ``(tile_size, overlap, axis_size)`` into the explicit
pixel positions of stitching seams; the gradient sampler fancy-indexes
those positions out of the per-axis gradient arrays.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .seams import (
    assert_shape_consistent,
    compute_middle_positions,
    compute_seam_positions,
    pixel_positions_to_grad_indices,
)


class GradientUtils:
    """Base class for seam-aligned gradient sampling and histograms."""

    # Subclasses describe their spatial layout here.
    _spatial_axes: tuple[str, ...] = ()
    _spatial_dims: tuple[int, ...] = ()  # axes of `imgs` that are spatial

    def __init__(
        self,
        imgs: np.ndarray,
        tile_size: Sequence[int],
        overlap: Sequence[int],
        bin_edges: Optional[np.ndarray] = None,
        channel: Optional[int] = None,
    ):
        if len(tile_size) != len(self._spatial_axes):
            raise ValueError(
                f"tile_size must have length {len(self._spatial_axes)} "
                f"(one per spatial axis {self._spatial_axes}); got {len(tile_size)}"
            )
        if len(overlap) != len(self._spatial_axes):
            raise ValueError(
                f"overlap must have length {len(self._spatial_axes)}; "
                f"got {len(overlap)}"
            )

        self.imgs = imgs
        self.tile_size = tuple(int(t) for t in tile_size)
        self.overlap = tuple(int(o) for o in overlap)
        self._bin_edges = bin_edges

        for axis_label, axis_dim, ts, ov in zip(
            self._spatial_axes,
            self._spatial_dims,
            self.tile_size,
            self.overlap,
            strict=True,
        ):
            assert_shape_consistent(imgs.shape[axis_dim], ts, ov, axis_label)

        seams_per_axis = [
            compute_seam_positions(imgs.shape[d], ts, ov)
            for d, ts, ov in zip(
                self._spatial_dims, self.tile_size, self.overlap, strict=True
            )
        ]
        middles_per_axis = [
            compute_middle_positions(imgs.shape[d], ts, ov)
            for d, ts, ov in zip(
                self._spatial_dims, self.tile_size, self.overlap, strict=True
            )
        ]

        if all(s.size == 0 for s in seams_per_axis):
            raise ValueError(
                "No seams found on any spatial axis — every axis fits in a "
                "single tile. The metric needs at least one seam to evaluate."
            )

        self.gradients = self.compute_gradients(self.imgs)
        self.grad_edges = self._sample_at(seams_per_axis, channel)
        self.grad_middle = self._sample_at(middles_per_axis, channel)

        self.mean = self.grad_middle.mean()
        self.std = self.grad_middle.std()

        if self._bin_edges is None:
            self._bin_edges = self.get_bin_edges(
                list(self.gradients) + [self.grad_edges, self.grad_middle]
            )

    @staticmethod
    def get_bin_edges(gradients: list, num_bins: int = 200) -> np.ndarray:
        flattened = np.concatenate([g.flatten() for g in gradients])
        _, bin_edges = np.histogram(flattened, bins=num_bins)
        return bin_edges

    @staticmethod
    def compute_histograms(
        gradients: np.ndarray, bin_edges: np.ndarray
    ) -> np.ndarray:
        return np.histogram(gradients, bins=bin_edges)[0]

    @staticmethod
    def wiener_entropy(hist: np.ndarray, eps: float = 1e-12) -> float:
        w = np.hanning(len(hist))
        X = np.fft.rfft(hist * w)
        P = np.abs(X) ** 2 + eps
        geom_mean = np.exp(np.mean(np.log(P)))
        arith_mean = np.mean(P)
        return 1.0 - float(geom_mean / (arith_mean + eps))

    @staticmethod
    def get_peakiness_scores(
        histogram_edges: np.ndarray,
        histogram_middle: np.ndarray,
        eps: float = 1e-12,
    ) -> list[float]:
        return [
            GradientUtils.wiener_entropy(x, eps=eps)
            for x in (
                histogram_edges,
                histogram_middle,
                histogram_middle - histogram_edges,
            )
        ]

    def make_bin_edges(self, n_bins: int = 2000) -> np.ndarray:
        return self.get_bin_edges(
            list(self.gradients) + [self.grad_edges, self.grad_middle],
            num_bins=n_bins,
        )

    def _normalize_gradients(
        self,
        gradients: np.ndarray,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        mu = self.mean if mu is None else mu
        sigma = self.std if sigma is None else sigma
        return (gradients - mu) / (sigma + 1e-8)

    @staticmethod
    def compute_gradients(imgs: np.ndarray):
        raise NotImplementedError

    def _sample_at(
        self,
        positions_per_axis: list[np.ndarray],
        channels: Optional[int],
    ) -> np.ndarray:
        raise NotImplementedError


# ----------------- 2D IMPLEMENTATION -----------------


class GradientUtils2D(GradientUtils):
    """2D implementation. Input shape ``(N, C, H, W)`` (channel-first)."""

    _spatial_axes = ("y", "x")
    _spatial_dims = (2, 3)

    @staticmethod
    def compute_gradients(imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad_y = imgs[:, :, 1:, :] - imgs[:, :, :-1, :]
        grad_x = imgs[:, :, :, 1:] - imgs[:, :, :, :-1]
        return grad_y, grad_x

    def _sample_at(
        self,
        positions_per_axis: list[np.ndarray],
        channels: Optional[int],
    ) -> np.ndarray:
        positions_y, positions_x = positions_per_axis
        grad_y, grad_x = self.gradients
        chan = slice(None) if channels is None else channels

        out: list[np.ndarray] = []
        if positions_y.size > 0:
            idx_y = pixel_positions_to_grad_indices(positions_y)
            out.append(grad_y[:, chan, idx_y, :].ravel())
        if positions_x.size > 0:
            idx_x = pixel_positions_to_grad_indices(positions_x)
            out.append(grad_x[:, chan, :, idx_x].ravel())

        return np.concatenate(out) if out else np.array([], dtype=grad_y.dtype)


# ----------------- 3D IMPLEMENTATION -----------------


class GradientUtils3D(GradientUtils):
    """3D implementation. Input shape ``(N, C, D, H, W)`` (channel-first)."""

    _spatial_axes = ("z", "y", "x")
    _spatial_dims = (2, 3, 4)

    @staticmethod
    def compute_gradients(
        imgs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad_z = imgs[:, :, 1:, :, :] - imgs[:, :, :-1, :, :]
        grad_y = imgs[:, :, :, 1:, :] - imgs[:, :, :, :-1, :]
        grad_x = imgs[:, :, :, :, 1:] - imgs[:, :, :, :, :-1]
        return grad_z, grad_y, grad_x

    def _sample_at(
        self,
        positions_per_axis: list[np.ndarray],
        channels: Optional[int],
    ) -> np.ndarray:
        positions_z, positions_y, positions_x = positions_per_axis
        grad_z, grad_y, grad_x = self.gradients
        chan = slice(None) if channels is None else channels

        out: list[np.ndarray] = []
        if positions_z.size > 0:
            idx_z = pixel_positions_to_grad_indices(positions_z)
            out.append(grad_z[:, chan, idx_z, :, :].ravel())
        if positions_y.size > 0:
            idx_y = pixel_positions_to_grad_indices(positions_y)
            out.append(grad_y[:, chan, :, idx_y, :].ravel())
        if positions_x.size > 0:
            idx_x = pixel_positions_to_grad_indices(positions_x)
            out.append(grad_x[:, chan, :, :, idx_x].ravel())

        return np.concatenate(out) if out else np.array([], dtype=grad_z.dtype)
