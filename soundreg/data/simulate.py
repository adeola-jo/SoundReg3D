"""STFT-domain acoustic scene simulator.

WHAT THIS IS FOR
----------------
Pipeline bring-up and *controlled* inductive-bias experiments. The
simulator gives three things real data cannot:

    - exact per-source mixture SNR (brief Eq. 2) for stratified recall,
    - oracle received energies (the `oracle_energy` ordering upper bound),
    - an NLOS toggle with known positions.

It is deliberately NOT an acoustically realistic street model — no
reverberation, no Doppler, no source movement. Headline claims belong to
real data (PLAN.md, milestone M2+).

SIGNAL MODEL
------------
Each source j has a stationary band-limited complex-Gaussian spectrum:

    s_j(f, t) = level_j * g_j(f) * CN(0, 1)

where g_j is a Gaussian band envelope and level_j is drawn from the
per-scene dominance spread (see below). Its array image is

    x_j(m, f, t) = a_m(p_j, f) * s_j(f, t)

with the SAME near-field manifold a the beamformer uses (steering.py),
so simulation and dominance scoring share one propagation convention by
construction. The mixture is

    X(t, f) = sum_j x_j(t, f) + n(t, f)        (brief Eq. 1)

with white complex-Gaussian noise n scaled to a per-scene SNR.

WHERE THE MASKING COMES FROM
----------------------------
Two knobs create the loud-hides-quiet hierarchy the project studies:

    dominance_spread_db   per-source levels drawn from U(-spread, 0) dB:
                          large spread = strong maskers over weak sources.
    bandwidth_hz          wide bands = more TF overlap between sources.

NLOS
----
With probability `nlos_prob` a source keeps its label but loses its
directional cues: the inter-mic phase coherence is destroyed (one random
phase per (m, f)) and the image is attenuated by `nlos_extra_atten_db`.
Crude, but it isolates exactly the question we care about — can the
model use scene context where the array geometry is silent?
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..types import SceneGT
from .polar_grid import polar_distance
from .steering import ArrayManifold


# =====================================================================
# Simulator
# =====================================================================
class SceneSimulator:
    """Draws one scene per call; fully determined by the passed RNG, so a
    (seed, index) pair is a reproducible dataset element with no storage."""

    def __init__(self, cfg: Dict, manifold: ArrayManifold, n_frames: int):
        """Args:
            cfg: The `data.sim` config section (see base.yaml for the
                full key list with comments).
            manifold: Shared propagation model (also used by dominance).
            n_frames: STFT frames per scene window.
        """
        self.cfg = cfg
        self.manifold = manifold
        self.n_frames = n_frames
        self.freqs = manifold.freqs_hz

    def sample(self, rng: np.random.Generator) -> Dict:
        """Generate one scene.

        Returns:
            dict with:
                stft    complex64 (M, F, T)   the mixture the model sees
                gt      SceneGT               positions + exact per-source SNR
                energy  (N,)                  oracle received energies
        """
        c = self.cfg
        m, f, t = self.manifold.n_mics, self.manifold.n_freqs, self.n_frames

        n_src = int(rng.integers(c["n_sources_min"], c["n_sources_max"] + 1))
        r, theta = self._sample_positions(rng, n_src, c)
        classes = rng.integers(0, c.get("n_classes", 1), size=n_src)
        nlos = rng.random(n_src) < c.get("nlos_prob", 0.0)

        # Per-source levels: the dominance spread creates the masking
        # hierarchy (0 dB = scene-loudest possible, -spread = weakest).
        level_db = rng.uniform(-c["dominance_spread_db"], 0.0, size=n_src)
        levels = 10.0 ** (level_db / 20.0)

        images = np.zeros((n_src, m, f, t), dtype=np.complex64)
        supports = np.zeros((n_src, f), dtype=bool)
        for j in range(n_src):
            env, support = self._band_envelope(rng, int(classes[j]))
            spec = (
                levels[j]
                * env[:, None]
                * (rng.standard_normal((f, t)) + 1j * rng.standard_normal((f, t)))
                / np.sqrt(2.0)
            )
            a = self.manifold.manifold(r[j], theta[j])[0]  # (F, M)
            img = a.T[:, :, None] * spec[None, :, :]  # (M, F, T)
            if nlos[j]:
                gamma = 10.0 ** (-c.get("nlos_extra_atten_db", 10.0) / 20.0)
                scramble = np.exp(2j * np.pi * rng.random((m, f)))
                img = img * gamma * scramble[:, :, None]
            images[j] = img
            supports[j] = support

        signal = images.sum(axis=0) if n_src > 0 else np.zeros((m, f, t), np.complex64)

        # Noise floor from the scene-level SNR over total received energy.
        scene_snr_db = rng.uniform(*c["scene_snr_db"])
        sig_energy = float(np.sum(np.abs(signal) ** 2)) if n_src > 0 else 1.0
        noise_energy = sig_energy * 10.0 ** (-scene_snr_db / 10.0)
        # Per-bin complex std: total noise energy spread over M*F*T bins,
        # split between real and imaginary parts.
        sigma = np.sqrt(noise_energy / (m * f * t) / 2.0)
        noise = sigma * (
            rng.standard_normal((m, f, t)) + 1j * rng.standard_normal((m, f, t))
        )
        stft = (signal + noise).astype(np.complex64)

        energy, snr_db = self._per_source_snr(images, supports, noise)

        gt = SceneGT(
            classes=classes.astype(np.int64),
            r=r,
            theta=theta,
            snr_db=snr_db,
            nlos=nlos,
        )
        return {"stft": stft, "gt": gt, "energy": energy}

    # =================================================================
    # Pieces
    # =================================================================
    def _sample_positions(
        self, rng, n_src: int, c: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform positions with a minimum pairwise BEV separation.

        Rejection sampling, 50 tries per source; after that the candidate
        is accepted regardless (a rare too-close pair beats an infinite
        loop in a dataloader worker). Sources are NOT separated in angle
        on purpose — angular collisions are part of the problem.
        """
        r = np.zeros(n_src)
        theta = np.zeros(n_src)
        min_sep = c.get("min_separation_m", 1.5)
        for j in range(n_src):
            for _ in range(50):
                rj = rng.uniform(c["r_min_m"], c["r_max_m"])
                tj = rng.uniform(-180.0, 180.0)
                if j == 0 or np.all(
                    polar_distance(rj, tj, r[:j], theta[:j]) >= min_sep
                ):
                    break
            r[j], theta[j] = rj, tj
        return r, theta

    def _band_envelope(self, rng, class_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian band envelope over frequency + its support mask.

        With n_classes > 1 each class owns a region of the spectrum, so
        "class" is acoustically meaningful and a classifier has something
        to learn. Single-class: random band center anywhere in
        [0.1, 0.9] * Nyquist.

        Returns:
            env      (F,) float32 amplitude envelope
            support  (F,) bool — bins where the source has meaningful
                     energy (env > support_threshold). Used to restrict
                     the per-source SNR to the source's own TF support,
                     exactly as the brief defines E^(j).
        """
        c = self.cfg
        f_nyq = float(self.freqs[-1])
        n_classes = c.get("n_classes", 1)
        if n_classes > 1:
            lo = f_nyq * (class_id + 0.25) / n_classes
            hi = f_nyq * (class_id + 0.75) / n_classes
            fc = rng.uniform(lo, hi)
        else:
            fc = rng.uniform(0.1 * f_nyq, 0.9 * f_nyq)
        bw = rng.uniform(*c["bandwidth_hz"])
        env = np.exp(-0.5 * ((self.freqs - fc) / bw) ** 2).astype(np.float32)
        support = env > c.get("support_threshold", 0.1)
        if not support.any():
            support[np.argmax(env)] = True
        return env, support

    def _per_source_snr(
        self, images: np.ndarray, supports: np.ndarray, noise: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact mixture SNR of each source on its own TF support.

        Implements brief Eq. 2 with C = empty set:

            SNR_j = E_j / ( sum_{i != j} E_i^(j) + E_n^(j) )

        where every energy is restricted to source j's frequency support
        (the stationary band model makes the time axis uninformative, so
        support is frequency-only here).

        Returns:
            energy  (N,) full received energy per source (oracle dominance)
            snr_db  (N,) per-source mixture SNR, dB (eval stratification)
        """
        n_src = images.shape[0]
        energy = np.array(
            [float(np.sum(np.abs(images[j]) ** 2)) for j in range(n_src)]
        )
        snr_db = np.zeros(n_src)
        for j in range(n_src):
            sup = supports[j]
            e_self = float(np.sum(np.abs(images[j][:, sup, :]) ** 2))
            e_interf = sum(
                float(np.sum(np.abs(images[i][:, sup, :]) ** 2))
                for i in range(n_src)
                if i != j
            )
            e_noise = float(np.sum(np.abs(noise[:, sup, :]) ** 2))
            snr_db[j] = 10.0 * np.log10(e_self / (e_interf + e_noise + 1e-12) + 1e-12)
        return energy, snr_db
