"""Near-field array manifold, steering vectors, SRP maps, dominance scores.

PUBLIC SURFACE
--------------
    ArrayManifold
        Geometry + frequency axis; produces manifold/steering vectors at
        arbitrary BEV polar points, with a cached version for the full grid.

    srp_at_points
        Steered-response power at given (r, theta) points. Evaluated at
        labeled source positions this *is* the dominance score d_i of the
        brief (Eq. 3) that fixes the emission order.

    srp_polar_map
        SRP map(s) over the whole polar grid, split into frequency bands.
        Encoder input for the polar backbone and the map the SRP/heatmap
        baselines pick peaks from.

PROPAGATION CONVENTION
----------------------
One model is used by the simulator AND the beamformer, so the two can
never drift apart:

    a_m(p_s, f) = (1 / d_m) * exp(-j 2 pi f d_m / c),   d_m = ||p_s - p_m||

i.e. spherical spreading (1/d amplitude) plus the absolute propagation
delay as phase. The delay-and-sum steering vector is the unit-normalized
manifold

    w(p, f) = a(p, f) / ||a(p, f)||

and the steered-response power of a stacked-channel STFT X (M, F, T) is

    SRP(p) = sum over t, f of | w(p, f)^H X(:, f, t) |^2 .

With `phat=True` every channel-TF bin of X is magnitude-normalized first
(a per-channel PHAT-style whitening): robust localization cue, used for
the SRP feature maps. Dominance uses `phat=False` — raw power, because
dominance is about *energy*, not detectability.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .polar_grid import PolarGrid, rtheta_to_xy

# Speed of sound in air at ~20 C (m/s). Good enough at array scale;
# revisit only if the real data was recorded in extreme conditions.
SPEED_OF_SOUND = 343.0


# =====================================================================
# Array manifold
# =====================================================================
class ArrayManifold:
    """Mic geometry + frequency axis -> manifold / steering vectors.

    Sources are assumed to sit at a fixed height `src_z` above the BEV
    plane (vehicles radiate roughly from wheel/engine height). The grid
    steering tensor is cached per grid because it is needed for every
    SRP map and never changes.
    """

    def __init__(
        self,
        mic_pos: np.ndarray,
        freqs_hz: np.ndarray,
        c: float = SPEED_OF_SOUND,
        src_z: float = 1.0,
    ):
        """Args:
            mic_pos: (M, 3) microphone positions, meters, ego frame.
            freqs_hz: (F,) STFT bin center frequencies.
            c: Speed of sound (m/s).
            src_z: Assumed source height above ground (m).
        """
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        self.c = c
        self.src_z = src_z
        self._grid_cache: Dict[Tuple, np.ndarray] = {}

    @property
    def n_mics(self) -> int:
        return self.mic_pos.shape[0]

    @property
    def n_freqs(self) -> int:
        return self.freqs_hz.shape[0]

    def manifold(self, r, theta_deg) -> np.ndarray:
        """Array manifold a (amplitude 1/d, phase from delay) at polar points.

        Args:
            r: (P,) or scalar ranges, meters.
            theta_deg: (P,) or scalar azimuths, degrees.

        Returns:
            complex128 (P, F, M).
        """
        r = np.atleast_1d(np.asarray(r, dtype=np.float64))
        theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=np.float64))
        x, y = rtheta_to_xy(r, theta_deg)
        src = np.stack([x, y, np.full_like(x, self.src_z)], axis=1)  # (P, 3)
        d = np.linalg.norm(src[:, None, :] - self.mic_pos[None, :, :], axis=2)  # (P, M)
        # Floor the distance so a source placed exactly on a mic cannot
        # blow up the 1/d amplitude.
        d = np.maximum(d, 1e-3)
        phase = -2.0j * np.pi * self.freqs_hz[None, :, None] * d[:, None, :] / self.c
        return np.exp(phase) / d[:, None, :]  # (P, F, M)

    def steering(self, r, theta_deg) -> np.ndarray:
        """Unit-norm (over mics) delay-and-sum steering vectors.

        Returns:
            complex128 (P, F, M) with ||w[p, f, :]|| = 1.
        """
        a = self.manifold(r, theta_deg)
        return a / np.linalg.norm(a, axis=2, keepdims=True)

    def grid_steering(self, grid: PolarGrid) -> np.ndarray:
        """Steering vectors for every grid cell, cached.

        Returns:
            complex64 (N_theta, N_r, F, M). complex64 halves the cache
            (default grid: ~18 MB) at no observable accuracy cost.
        """
        key = (grid.n_theta, grid.n_r, grid.r_max)
        if key not in self._grid_cache:
            tt, rr = np.meshgrid(grid.theta_centers, grid.r_centers, indexing="ij")
            w = self.steering(rr.ravel(), tt.ravel())
            self._grid_cache[key] = w.reshape(
                grid.n_theta, grid.n_r, self.n_freqs, self.n_mics
            ).astype(np.complex64)
        return self._grid_cache[key]


# =====================================================================
# Steered-response power
# =====================================================================
def _phat_whiten(stft: np.ndarray) -> np.ndarray:
    """Magnitude-normalize every channel-TF bin (keep phase only)."""
    return stft / np.maximum(np.abs(stft), 1e-8)


def srp_at_points(
    stft: np.ndarray,
    manifold: ArrayManifold,
    r,
    theta_deg,
    phat: bool = False,
) -> np.ndarray:
    """Steered-response power at arbitrary polar points.

    This is the dominance score of the brief when evaluated at the GT
    positions:

        d_i = sum over t, f of | w(r_i, theta_i)^H X(t, f) |^2

    Args:
        stft: complex (M, F, T) mixture STFT.
        manifold: ArrayManifold matching the stft's geometry/freq axis.
        r, theta_deg: (P,) query points.
        phat: Whiten magnitudes first. Keep False for dominance —
            dominance must reflect raw energy, not just phase coherence.

    Returns:
        (P,) float steered power, descending order = emission order.
    """
    x = _phat_whiten(stft) if phat else stft
    w = manifold.steering(r, theta_deg)  # (P, F, M)
    y = np.einsum("pfm,mft->pft", np.conj(w), x)
    return np.sum(np.abs(y) ** 2, axis=(1, 2)).real


def srp_polar_map(
    stft: np.ndarray,
    manifold: ArrayManifold,
    grid: PolarGrid,
    n_bands: int = 1,
    phat: bool = True,
) -> np.ndarray:
    """SRP map(s) over the full polar grid.

    The frequency axis is split into `n_bands` contiguous bands and the
    power is aggregated per band, so the encoder sees coarse spectral
    structure instead of one fully collapsed map (a bus and an e-scooter
    light up different bands).

    Args:
        stft: complex (M, F, T) mixture STFT.
        manifold: ArrayManifold for the same geometry.
        grid: Target polar grid.
        n_bands: Number of contiguous frequency bands to keep separate.
        phat: PHAT-style whitening (default True — localization cue).

    Returns:
        float32 (n_bands, N_theta, N_r).
    """
    x = _phat_whiten(stft) if phat else stft
    w = manifold.grid_steering(grid)  # (N_theta, N_r, F, M)
    y = np.einsum("qrfm,mft->qrft", np.conj(w), x.astype(np.complex64))
    power = np.abs(y) ** 2  # (N_theta, N_r, F, T)
    f_edges = np.linspace(0, manifold.n_freqs, n_bands + 1).astype(int)
    bands = [
        power[:, :, f_edges[b] : f_edges[b + 1], :].sum(axis=(2, 3))
        for b in range(n_bands)
    ]
    return np.stack(bands, axis=0).astype(np.float32)
