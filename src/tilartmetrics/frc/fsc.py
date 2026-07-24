"""Fourier Shell Correlation (3-D) — DEFERRED placeholder.

FSC is the 3-D analog of FRC: spherical shells in 3-D Fourier space
instead of rings in 2-D. MicroSplit ships some 3-D data, so FSC would
broaden the metric's applicability.

Why deferred: fluorescence microscopy has anisotropic resolution along Z
(PSF elongated along the optical axis). Plain isotropic FSC (i) under-
represents lateral resolution and (ii) dilutes any Z-direction seam signal
across all Fourier-space directions. The principled solution is the
direction-resolved SFSC of Koho et al. 2019 (Nat. Commun. 10:3103), with
angular wedges — non-trivial to implement correctly and out of scope for
the workshop submission.

This module is intentionally empty; the deferral note above is the entire
contents. See ``agents_artifacts/FRC_metric.md`` §5 for the design context.
"""
