"""Configuration model for the FRC metric."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FRCConfig(BaseModel):
    """Parameters of the Fourier Ring Correlation metric.

    Attributes
    ----------
    apply_window : bool, default=True
        Apply a 2-D Hamming window before the FFT. Mandatory for real
        images; disable only for sanity tests.
    channel : int, default=0
        Channel index to analyse.
    """

    apply_window: bool = True
    channel: int = Field(default=0, ge=0)
