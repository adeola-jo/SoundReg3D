"""render.py — free-field multichannel rendering of point sources.

Given a mono source clip and a position, produce the (T, M) array signal via
per-mic fractional delay (frequency-domain) and 1/r amplitude. Deliberately
v0-simple: no reflections, no car-body shadowing (see PROTOCOL.md §6).
"""
from __future__ import annotations
import numpy as np
from .geometry import distances_to, C_SOUND


def _frac_delay_fd(x: np.ndarray, delays_s: np.ndarray, fs: int) -> np.ndarray:
    """Delay mono signal x (T,) by each value in delays_s -> (T, M).

    Frequency-domain phase ramp (exact fractional delay, circular; we pad to
    avoid wrap-around for delays up to r_max/c).
    """
    T = len(x)
    pad = int(np.ceil(delays_s.max() * fs)) + 8
    n = int(2 ** np.ceil(np.log2(T + pad)))
    X = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    ph = np.exp(-2j * np.pi * f[None, :] * delays_s[:, None])     # (M, F)
    y = np.fft.irfft(X[None, :] * ph, n=n, axis=1)[:, :T]          # (M, T)
    return y.T.astype(np.float32)                                   # (T, M)


def render_source(clip: np.ndarray, fs: int, mics: np.ndarray,
                  xy: tuple[float, float], gain: float = 1.0,
                  ref_dist: float = 1.0) -> np.ndarray:
    """Render one source at xy -> (T, M) per-mic signals (pre-mix image x_i)."""
    d = distances_to(mics, np.asarray(xy, dtype=np.float64))        # (M,)
    delays = d / C_SOUND
    delays -= delays.min()            # remove common bulk delay (keeps clips aligned)
    amp = gain * (ref_dist / np.maximum(d, ref_dist))               # 1/r law
    y = _frac_delay_fd(clip.astype(np.float64), delays, fs)         # (T, M)
    return (y * amp[None, :]).astype(np.float32)


def mix_scene(source_images: list[np.ndarray], noise: np.ndarray | None) -> np.ndarray:
    """Sum per-source array images (+ optional (T, M) noise bed) -> mixture (T, M)."""
    T = min(s.shape[0] for s in source_images) if source_images else noise.shape[0]
    out = np.zeros((T, source_images[0].shape[1] if source_images else noise.shape[1]),
                   dtype=np.float32)
    for s in source_images:
        out += s[:T]
    if noise is not None:
        out += noise[:T, :out.shape[1]]
    return out


def stft_multi(x: np.ndarray, fs: int, n_fft: int = 512, hop: int = 256,
               fmin: float = 50.0, fmax: float = 4000.0):
    """Multichannel STFT -> (F, T_frames, M) complex64, plus kept freqs (Hz).

    Band-limited to [fmin, fmax]: below ~50 Hz is wind rumble; above ~4 kHz the
    32-mic spacing aliases spatially (free-field model unreliable there anyway).
    """
    from scipy.signal import stft as _stft
    f, t, Z = _stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, axis=0,
                    padded=False, boundary=None)
    # scipy returns (F, M, T) with axis=0 input (T, M) -> transpose to (F, T, M)
    Z = np.moveaxis(Z, 1, 2).astype(np.complex64)
    keep = (f >= fmin) & (f <= fmax)
    return Z[keep], f[keep]
