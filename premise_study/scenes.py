"""scenes.py — build Set A (naturalistic) and Set B (controlled) scenes.

Each scene dict:
  mixture   (T, M) float32
  images    list of per-source (T, M) arrays (pre-mix; enables all oracles)
  noise     (T, M)
  positions [(theta_rad, r_m)] per source
  ranks     loudness rank per source (0 = loudest, by image energy)
  meta      dict (set name, sweep cell, clip ids, seed)
"""
from __future__ import annotations
import numpy as np
from .render import render_source, mix_scene


def _rank(images):
    e = np.array([float((im ** 2).sum()) for im in images])
    r = np.empty(len(e), dtype=int)
    r[np.argsort(-e)] = np.arange(len(e))
    return r


def _scale_to_snr(img, noise, snr_db):
    """Scale a source image so its energy vs noise energy hits snr_db."""
    es, en = (img ** 2).sum(), (noise ** 2).sum() + 1e-12
    return img * np.sqrt(10 ** (snr_db / 10) * en / es)


def make_scene(clips, fs, mics, positions, snrs_db, noise, seed=0, meta=None):
    rng = np.random.default_rng(seed)
    T = noise.shape[0]
    images = []
    for clip, (th, r), snr in zip(clips, positions, snrs_db):
        if len(clip) < T:
            clip = np.tile(clip, int(np.ceil(T / len(clip))))
        off = rng.integers(0, max(1, len(clip) - T))
        img = render_source(clip[off:off + T], fs, mics,
                            (r * np.cos(th), r * np.sin(th)))
        images.append(_scale_to_snr(img, noise, snr))
    return {"mixture": mix_scene(images, noise), "images": images, "noise": noise,
            "positions": list(positions), "ranks": _rank(images),
            "meta": meta or {}}


def set_a(bank, n_scenes, fs, mics, card_probs, pos_sampler, snr_sampler,
          seed=0):
    """Naturalistic scenes. bank: dict(clips=[(id, mono)], noises=[(speed,(T,M))]).
    card_probs over N in {1..4}; pos_sampler(rng,N)->[(th,r)];
    snr_sampler(rng,N)->[dB] — wire all three to the REAL dataset statistics
    via data_adapter.py (this keeps Set A honest)."""
    rng = np.random.default_rng(seed)
    scenes = []
    for s in range(n_scenes):
        N = rng.choice(np.arange(1, len(card_probs) + 1), p=card_probs)
        ci = rng.choice(len(bank["clips"]), size=N, replace=True)
        _, noise = bank["noises"][rng.integers(len(bank["noises"]))]
        scenes.append(make_scene([bank["clips"][i][1] for i in ci], fs, mics,
                                 pos_sampler(rng, N), snr_sampler(rng, N), noise,
                                 seed=seed + s, meta={"set": "A", "n": int(N)}))
    return scenes


def set_b(bank, fs, mics, *, dthetas_deg=(2, 5, 10, 20, 45, 90),
          dsnrs_db=(0, 6, 12, 18, 24), per_cell=50, base_snr=12.0,
          r_pair=(12.0, 15.0), seed=0):
    """Controlled 2-source sweeps: angular separation x level difference.
    Spectral-overlap factor comes from clip pairing (same vs different clip id),
    alternated within each cell."""
    rng = np.random.default_rng(seed)
    scenes = []
    for dth in dthetas_deg:
        for dsnr in dsnrs_db:
            for k in range(per_cell):
                th0 = rng.uniform(-np.pi, np.pi)
                pos = [(th0, r_pair[0]),
                       (th0 + np.radians(dth), r_pair[1])]
                i0 = rng.integers(len(bank["clips"]))
                i1 = i0 if k % 2 == 0 else rng.integers(len(bank["clips"]))
                _, noise = bank["noises"][rng.integers(len(bank["noises"]))]
                scenes.append(make_scene(
                    [bank["clips"][i0][1], bank["clips"][i1][1]], fs, mics, pos,
                    [base_snr, base_snr - dsnr], noise, seed=seed + len(scenes),
                    meta={"set": "B", "dtheta": dth, "dsnr": dsnr,
                          "overlap": "same" if i1 == i0 else "diff"}))
    return scenes


def set_b3(bank, fs, mics, n_scenes=500, snrs=(12.0, 4.0, -4.0), seed=0):
    """3-source staggered-level scenes for the order test (P3)."""
    rng = np.random.default_rng(seed)
    scenes = []
    for s in range(n_scenes):
        pos = [(rng.uniform(-np.pi, np.pi), rng.uniform(6, 30)) for _ in range(3)]
        ci = rng.choice(len(bank["clips"]), size=3)
        _, noise = bank["noises"][rng.integers(len(bank["noises"]))]
        scenes.append(make_scene([bank["clips"][i][1] for i in ci], fs, mics,
                                 pos, list(snrs), noise, seed=seed + s,
                                 meta={"set": "B3"}))
    return scenes
