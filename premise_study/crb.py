"""crb.py — best-possible (r, theta) accuracy for the actual array geometry.

Model: known broadband waveform, free-field delays, white Gaussian noise.
FIM over p = (r, theta) from the delay gradients:

    F(p) = (2 / sigma^2) * sum_f |S(f)|^2 (2 pi f)^2 * sum_m  g_m(p) g_m(p)^T
    g_m(p) = d tau_m / d p   (centered: common delay removed -> use deviations)

We express the broadband factor through array SNR and an effective bandwidth:
    F = 2 * SNR_per_mic * T_samples * (2 pi)^2 * E[f^2]_S * G(p)
For the decision rules only RELATIVE structure and order of magnitude matter;
we report sigma at stated SNRs and clearly label assumptions. The amplitude
(1/r) information term is omitted -> the range bound shown is CONSERVATIVE
(true achievable range accuracy is slightly better at close r).
"""
from __future__ import annotations
import numpy as np
from .geometry import C_SOUND


def delay_gradients(mics: np.ndarray, theta: float, r: float, src_z: float = 0.0):
    """d tau_m / d(r, theta) with the common (mean over mics) component removed."""
    s = np.array([r * np.cos(theta), r * np.sin(theta), src_z])
    d_vec = s[None, :] - mics                     # (M, 3)
    d = np.linalg.norm(d_vec, axis=1)             # (M,)
    u = d_vec / d[:, None]
    ds_dr = np.array([np.cos(theta), np.sin(theta), 0.0])
    ds_dth = np.array([-r * np.sin(theta), r * np.cos(theta), 0.0])
    g_r = (u @ ds_dr) / C_SOUND
    g_th = (u @ ds_dth) / C_SOUND
    g = np.stack([g_r - g_r.mean(), g_th - g_th.mean()], axis=1)   # (M, 2)
    return g


def crb_sigma(mics: np.ndarray, theta: float, r: float, snr_db: float,
              f_lo: float = 50.0, f_hi: float = 4000.0,
              t_obs_s: float = 1.0, fs: int = 16000):
    """Return (sigma_theta_deg, sigma_r_m) for one grid point.

    Flat signal spectrum on [f_lo, f_hi]; SNR is per-mic broadband SNR.
    """
    g = delay_gradients(mics, theta, r)            # (M, 2)
    G = g.T @ g                                     # (2, 2)
    Ef2 = (f_hi ** 3 - f_lo ** 3) / (3 * (f_hi - f_lo))   # E[f^2], flat spectrum
    snr = 10 ** (snr_db / 10)
    scale = 2.0 * snr * (t_obs_s * fs) * (2 * np.pi) ** 2 * Ef2
    F = scale * G
    try:
        C = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.inf, np.inf
    sig_r = float(np.sqrt(max(C[0, 0], 0)))
    sig_th = float(np.sqrt(max(C[1, 1], 0)))
    return np.degrees(sig_th), sig_r


def crb_maps(mics: np.ndarray, thetas: np.ndarray, rs: np.ndarray,
             snr_db: float, **kw):
    """sigma_theta_deg and sigma_r maps over the polar grid."""
    Sth = np.zeros((len(thetas), len(rs)))
    Sr = np.zeros_like(Sth)
    for i, th in enumerate(thetas):
        for j, r in enumerate(rs):
            Sth[i, j], Sr[i, j] = crb_sigma(mics, th, r, snr_db, **kw)
    return Sth, Sr


def r_max_for_relative_error(mics: np.ndarray, snr_db: float,
                             rel: float = 0.25, theta: float = 0.0,
                             r_grid=None, **kw) -> float:
    """Largest range where sigma_r / r <= rel (median over a few azimuths)."""
    rg = np.geomspace(2, 100, 60) if r_grid is None else r_grid
    ths = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    ok = np.zeros(len(rg), dtype=bool)
    for j, r in enumerate(rg):
        rels = [crb_sigma(mics, th, r, snr_db, **kw)[1] / r for th in ths]
        ok[j] = np.median(rels) <= rel
    return float(rg[ok][-1]) if ok.any() else 0.0
