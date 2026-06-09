"""Polar BEV grid: the target vocabulary for range and azimuth tokens.

The grid IS the output space of the detector — one range token per range
bin, one azimuth token per azimuth bin (tokenizer.py builds its
vocabulary directly from `n_r` and `n_theta`). Choosing the grid is
therefore choosing the localization resolution AND the vocabulary size
at once; record it with every dataset split (PLAN.md, M1).

CONVENTIONS (project-wide, do not change casually)
--------------------------------------------------
    theta   degrees in [-180, 180), bin 0 starts at -180,
            N_theta uniform bins. Azimuth is finer than range
            (default 5 deg vs 1.25 m) per the brief.
    r       meters in [0, r_max], N_r uniform bins.
    frame   ego at origin, x forward, y left, theta = atan2(y, x).

Dequantization returns bin centers, so the quantization error is bounded
by half a bin in each axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..utils import wrap_deg


# =====================================================================
# Grid
# =====================================================================
@dataclass(frozen=True)
class PolarGrid:
    """Frozen so a grid can be shared between dataset, tokenizer, and
    model without anyone mutating the vocabulary under everyone else."""

    n_r: int = 24
    n_theta: int = 72
    r_max: float = 30.0

    @property
    def dr(self) -> float:
        """Range bin width (m)."""
        return self.r_max / self.n_r

    @property
    def dtheta(self) -> float:
        """Azimuth bin width (deg)."""
        return 360.0 / self.n_theta

    # =================================================================
    # Quantize (continuous -> bin index)
    # =================================================================
    def r_to_bin(self, r) -> np.ndarray:
        """Range (m) -> bin index, clipped into [0, n_r - 1].

        Out-of-range values are clipped rather than rejected: a GT object
        a hair beyond r_max should land in the last bin, not crash a
        dataloader worker.
        """
        idx = np.floor(np.asarray(r, dtype=np.float64) / self.dr).astype(np.int64)
        return np.clip(idx, 0, self.n_r - 1)

    def theta_to_bin(self, theta_deg) -> np.ndarray:
        """Azimuth (deg, any wrap) -> bin index in [0, n_theta - 1].

        Wraps first, so theta_to_bin(181) == theta_to_bin(-179). The
        final clip only guards the floating-point edge case where
        wrap_deg returns a value within rounding of +180.
        """
        t = wrap_deg(np.asarray(theta_deg, dtype=np.float64))
        idx = np.floor((t + 180.0) / self.dtheta).astype(np.int64)
        return np.clip(idx, 0, self.n_theta - 1)

    # =================================================================
    # Dequantize (bin index -> bin center)
    # =================================================================
    def r_center(self, idx) -> np.ndarray:
        return (np.asarray(idx, dtype=np.float64) + 0.5) * self.dr

    def theta_center(self, idx) -> np.ndarray:
        return -180.0 + (np.asarray(idx, dtype=np.float64) + 0.5) * self.dtheta

    @property
    def r_centers(self) -> np.ndarray:
        """(n_r,) all range bin centers."""
        return self.r_center(np.arange(self.n_r))

    @property
    def theta_centers(self) -> np.ndarray:
        """(n_theta,) all azimuth bin centers."""
        return self.theta_center(np.arange(self.n_theta))


# =====================================================================
# Coordinate helpers
# =====================================================================
def xy_to_rtheta(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """BEV cartesian (x forward, y left) -> (r meters, theta degrees).

    theta comes back wrapped to [-180, 180), matching the grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.hypot(x, y)
    theta = wrap_deg(np.degrees(np.arctan2(y, x)))
    return r, theta


def rtheta_to_xy(r, theta_deg) -> Tuple[np.ndarray, np.ndarray]:
    """(r meters, theta degrees) -> BEV cartesian (x, y)."""
    th = np.radians(np.asarray(theta_deg, dtype=np.float64))
    r = np.asarray(r, dtype=np.float64)
    return r * np.cos(th), r * np.sin(th)


def polar_distance(r1, theta1_deg, r2, theta2_deg) -> np.ndarray:
    """Euclidean BEV distance between two polar points (law of cosines):

        d = sqrt( r1^2 + r2^2 - 2 r1 r2 cos(theta1 - theta2) )

    This is THE matching metric of the project (PLAN.md: Hungarian on
    BEV distance, valid if <= tau). It is wrap-safe by construction —
    cos doesn't care which side of the 0/360 seam the difference is on.

    Example:
        polar_distance(10, 179, 10, -179)  ->  ~0.35 m (same spot)
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    dth = np.radians(np.asarray(theta1_deg, dtype=np.float64) - theta2_deg)
    d2 = r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(dth)
    # Clamp tiny negative values from cancellation before the sqrt.
    return np.sqrt(np.maximum(d2, 0.0))
