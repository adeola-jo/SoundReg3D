"""glimpse.py — how much of the TF plane does each source own?

G_j(C) = fraction of (t, f) cells where source j's energy exceeds the energy of
all interferers not in C, plus noise. Computed from per-source array images,
which the simulator stores before mixing — so this is exact, not estimated.
"""
from __future__ import annotations
import numpy as np
from .render import stft_multi


def tf_energy(img: np.ndarray, fs: int, **stft_kw) -> np.ndarray:
    """Per-source array image (T, M) -> TF energy map (F, T_frames), summed over mics."""
    Z, _ = stft_multi(img, fs, **stft_kw)
    return (np.abs(Z) ** 2).sum(axis=-1)


def glimpse_measures(energies: list[np.ndarray], noise_energy: np.ndarray,
                     beta: float = 1.0, floor_db: float = -60.0):
    """For each source j (energies sorted ANY order), return:

    G_empty[j]   = G_j(empty)        — vs everything (parallel detector's view)
    G_louder[j]  = G_j(louder set)   — vs only quieter sources + noise
    headroom[j]  = G_louder - G_empty

    'louder' is decided by total energy. beta=1.0 means 0 dB local dominance.
    Cells where source j is below floor_db relative to its own peak are
    excluded from j's denominator (a source can't claim cells where it is
    essentially absent).
    """
    E = [np.maximum(e, 0.0) for e in energies]
    tot = np.array([e.sum() for e in E])
    order = np.argsort(-tot)                       # loudest first
    rank = np.empty(len(E), dtype=int)
    rank[order] = np.arange(len(E))

    G_empty, G_louder, head = np.zeros(len(E)), np.zeros(len(E)), np.zeros(len(E))
    for j, Ej in enumerate(E):
        active = Ej > Ej.max() * 10 ** (floor_db / 10)
        if not active.any():
            continue
        others = [E[i] for i in range(len(E)) if i != j]
        interf_all = np.sum(others, axis=0) if others else 0.0
        quieter = [E[i] for i in range(len(E)) if i != j and rank[i] > rank[j]]
        interf_q = np.sum(quieter, axis=0) if quieter else 0.0

        dom_all = Ej > beta * (interf_all + noise_energy)
        dom_q = Ej > beta * (interf_q + noise_energy)
        G_empty[j] = (dom_all & active).sum() / active.sum()
        G_louder[j] = (dom_q & active).sum() / active.sum()
        head[j] = G_louder[j] - G_empty[j]
    return G_empty, G_louder, head, rank
