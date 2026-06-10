"""geometry.py — microphone geometry and steering vectors.

Parses the array XML into mic positions and precomputes near-field steering
vectors on the polar (r, theta) grid. Backend: numpy, with optional torch
(GPU) for the big steering tensor used by the detector.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np

C_SOUND = 343.0  # m/s


def load_mic_positions(xml_path: str) -> np.ndarray:
    """Parse mic XML -> (M, 3) positions in meters.

    Tries common layouts: <microphone x= y= z=>, <mic><x>..</x>..</mic>,
    or any element with x/y/z attributes or children. If your XML differs,
    adapt ONLY this function — everything downstream consumes the (M, 3) array.
    """
    root = ET.parse(xml_path).getroot()
    pos = []
    for el in root.iter():
        attrs = {k.lower(): v for k, v in el.attrib.items()}
        if {"x", "y", "z"} <= set(attrs):
            pos.append([float(attrs["x"]), float(attrs["y"]), float(attrs["z"])])
            continue
        kids = {c.tag.lower(): c.text for c in el}
        if {"x", "y", "z"} <= set(kids):
            pos.append([float(kids["x"]), float(kids["y"]), float(kids["z"])])
    if not pos:
        raise ValueError(f"No mic positions found in {xml_path}; adapt load_mic_positions().")
    arr = np.asarray(pos, dtype=np.float64)
    # de-duplicate (nested elements can double-report)
    _, idx = np.unique(np.round(arr, 6), axis=0, return_index=True)
    return arr[np.sort(idx)]


def polar_grid(n_theta: int, n_r: int, r_min: float, r_max: float,
               log_range: bool = True):
    """Polar grid (theta in rad over full circle, r in meters).

    log_range=True spaces range bins logarithmically — constant *relative*
    range resolution, matching how the CRB for range degrades with distance.
    """
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    if log_range:
        rs = np.geomspace(r_min, r_max, n_r)
    else:
        rs = np.linspace(r_min, r_max, n_r)
    return thetas, rs


def grid_xy(thetas: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """(n_theta, n_r, 2) Cartesian coordinates of grid cells (z = source height 0)."""
    T, R = np.meshgrid(thetas, rs, indexing="ij")
    return np.stack([R * np.cos(T), R * np.sin(T)], axis=-1)


def delays_to(mics: np.ndarray, src_xy: np.ndarray, src_z: float = 0.0) -> np.ndarray:
    """Propagation delays (s). mics (M,3); src_xy (...,2) -> (..., M)."""
    src = np.concatenate([src_xy, np.full(src_xy.shape[:-1] + (1,), src_z)], axis=-1)
    d = np.linalg.norm(src[..., None, :] - mics[None, :, :]
                       if src.ndim == 1 else src[..., None, :] - mics, axis=-1)
    return d / C_SOUND


def distances_to(mics: np.ndarray, src_xy: np.ndarray, src_z: float = 0.0) -> np.ndarray:
    src = np.concatenate([src_xy, np.full(src_xy.shape[:-1] + (1,), src_z)], axis=-1)
    return np.linalg.norm(src[..., None, :] - mics, axis=-1)


def steering_matrix(mics: np.ndarray, freqs_hz: np.ndarray,
                    thetas: np.ndarray, rs: np.ndarray,
                    ref: str = "centroid") -> np.ndarray:
    """Near-field steering vectors W: (n_theta, n_r, F, M), complex64.

    w_m(g, f) = exp(-j 2 pi f (tau_m(g) - tau_ref(g)))  (unit-gain phase model;
    amplitude differences across the array are second-order for SRP).
    """
    xy = grid_xy(thetas, rs)                      # (T, R, 2)
    tau = delays_to(mics, xy.reshape(-1, 2))      # (T*R, M)
    tau -= tau.mean(axis=1, keepdims=True) if ref == "centroid" else tau.min(axis=1, keepdims=True)
    ph = -2j * np.pi * freqs_hz[None, :, None] * tau[:, None, :]   # (T*R, F, M)
    W = np.exp(ph).astype(np.complex64)
    return W.reshape(len(thetas), len(rs), len(freqs_hz), len(mics))


def synthetic_array(n_mics: int = 32, width: float = 1.6, depth: float = 0.4,
                    height: float = 0.5, seed: int = 0) -> np.ndarray:
    """Placeholder roof-rack-like 32-mic layout (4 x 8 grid) for smoke tests ONLY.

    Replaced by load_mic_positions(<real XML>) for all real results.
    """
    nx, ny = 8, max(1, n_mics // 8)
    xs = np.linspace(-width / 2, width / 2, nx)
    ys = np.linspace(-depth / 2, depth / 2, ny)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, height)], axis=1)
    return P[:n_mics]
