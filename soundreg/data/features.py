"""Encoder input features from the multichannel STFT.

FEATURE STACKS
--------------
Selected per-experiment via `data.features` (a list — a sample can carry
several stacks at once when comparing encoders on identical scenes):

    "logmag_ipd"   per-mic log magnitude + cos/sin inter-channel phase
                   difference w.r.t. mic 0.

                       output (C, T, F),  C = M + 2 (M - 1)

                   Consumed by SpectroCNNEncoder. Log-mag carries the
                   what (spectral content, level), IPD carries the where
                   (phase delays between mics encode direction). cos/sin
                   instead of the raw angle keeps the feature continuous
                   across the -pi/pi seam.

    "srp"          SRP-PHAT maps on the polar grid (steering.py):

                       output (n_bands, N_theta, N_r)

                   Consumed by PolarSRPEncoder and by the SRP-peaks and
                   heatmap baselines.

GCC-PHAT lag features are planned for M1; IPD already carries the same
phase information in a different parameterization, so they are a
robustness comparison rather than a gap.

STORAGE LAYOUT
--------------
Stored/real data ships STFTs as stacked real channels (the brief's
"32 mics; complex split to 64 real ch"): first M channels are the real
parts, last M the imaginary parts. `stacked_to_complex` /
`complex_to_stacked` convert between that and the complex (M, F, T)
used everywhere inside this package.
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# Layout conversion
# =====================================================================
def stacked_to_complex(stft_stacked: np.ndarray) -> np.ndarray:
    """(2M, F, T) stacked real+imag -> complex (M, F, T).

    Raises:
        ValueError: If the channel count is odd (cannot be a real/imag
            split of anything).
    """
    two_m = stft_stacked.shape[0]
    if two_m % 2 != 0:
        raise ValueError(
            f"Expected an even channel count (real halves + imag halves); "
            f"got {two_m}."
        )
    m = two_m // 2
    return stft_stacked[:m].astype(np.float32) + 1j * stft_stacked[m:].astype(
        np.float32
    )


def complex_to_stacked(stft: np.ndarray) -> np.ndarray:
    """complex (M, F, T) -> (2M, F, T) float32, real parts first."""
    return np.concatenate([stft.real, stft.imag], axis=0).astype(np.float32)


# =====================================================================
# Feature stacks
# =====================================================================
def logmag_ipd(stft: np.ndarray, ref_mic: int = 0) -> np.ndarray:
    """Log-magnitude + IPD feature image.

    Channel layout (C = M + 2(M-1) total):

        [ 0 .. M-1 ]            log1p|X_m|, standardized over the sample
        [ M .. M+M-2 ]          cos(phase_m - phase_ref), non-ref mics
        [ ...rest ]             sin(phase_m - phase_ref), non-ref mics

    Standardization is per-sample (one mean/std over all log-mag values):
    scenes vary hugely in absolute level and the absolute level is not a
    localization cue. cos/sin channels are already in [-1, 1] and are
    left untouched.

    Args:
        stft: complex (M, F, T) mixture STFT.
        ref_mic: Phase-reference microphone index.

    Returns:
        float32 (C, T, F) — note the (T, F) image orientation the conv
        encoder expects, transposed from the (F, T) storage order.
    """
    logmag = np.log1p(np.abs(stft))  # (M, F, T)
    mu, sigma = logmag.mean(), logmag.std() + 1e-6
    logmag = (logmag - mu) / sigma

    phase = np.angle(stft)
    ipd = phase - phase[ref_mic : ref_mic + 1]  # (M, F, T)
    others = [m for m in range(stft.shape[0]) if m != ref_mic]
    ipd = ipd[others]
    feats = np.concatenate([logmag, np.cos(ipd), np.sin(ipd)], axis=0)
    return feats.transpose(0, 2, 1).astype(np.float32)  # (C, T, F)


def logmag_ipd_channels(n_mics: int) -> int:
    """Channel count of the logmag_ipd stack — the factory needs it to
    size the encoder before any sample exists."""
    return n_mics + 2 * (n_mics - 1)


def normalize_srp(srp: np.ndarray) -> np.ndarray:
    """Standardize each band of an SRP map over its own (theta, r) plane.

    Per-sample, per-band: peak-picking thresholds and conv features then
    live on a comparable scale across scenes with wildly different
    absolute energies.

    Args:
        srp: (n_bands, N_theta, N_r) raw SRP power.

    Returns:
        float32, same shape, zero mean / unit std per band.
    """
    mu = srp.mean(axis=(1, 2), keepdims=True)
    sigma = srp.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((srp - mu) / sigma).astype(np.float32)
