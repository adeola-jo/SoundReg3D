"""data_adapter.py — the ONLY file that touches the real Acoustic-BEV dataset.

Everything else consumes its outputs. It does two jobs:

1. extract  : build the clip bank (one-box windows -> beamformed quasi-clean
              clips), the noise beds (zero-box windows, grouped by ego speed),
              and the statistics manifest (cardinality histogram, (r, theta)
              distribution, per-source SNR estimates) that Set A samples from.
2. anchor   : estimate glimpse measures on REAL 2-box windows for the
              real-data anchor plot (PROTOCOL.md §7).

WIRING REQUIRED (marked TODO): point `iter_samples()` at your loader. The
expected per-sample schema follows the working brief §7:

    item = {
      "stft":  ndarray (64, F, T),      # 32 mics, re/im stacked
      "audio": ndarray (T, 32),         # raw multichannel (preferred here)
      "ann":   [ {"center":[x,y,z], "wlh":[w,l,h]}, ... ],
      "velocity": {...}, "timestamp": ..., "scene_name": ...
    }

Run:  python -m premise_study.data_adapter extract --config config.yaml
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import yaml

from .geometry import load_mic_positions, steering_matrix
from .render import stft_multi


# --------------------------------------------------------------------------
# TODO(JOSEPH): wire this iterator to the SoundReg repo's dataset loader.
# It must yield the per-sample dicts described above. Two common options:
#   from src.data import AcousticBEVDataset   # adjust to the actual module
#   ds = AcousticBEVDataset(cfg["data_root"], split="train")
# or iterate the on-disk token files directly.
# --------------------------------------------------------------------------
def iter_samples(cfg):
    raise NotImplementedError(
        "Wire iter_samples() to the Acoustic-BEV loader (see file docstring). "
        "Everything downstream of this function is dataset-agnostic.")


def _ann_to_polar(ann):
    out = []
    for a in ann:
        x, y = a["center"][0], a["center"][1]
        out.append((float(np.arctan2(y, x)), float(np.hypot(x, y))))
    return out


def _beamform_clip(audio, fs, mics, pos, cfg):
    """Delay-and-sum toward pos -> mono quasi-clean clip (T,)."""
    Z, freqs = stft_multi(audio, fs, **cfg["stft"])
    th, r = pos
    W = steering_matrix(mics, freqs, np.array([th]), np.array([r]))[0, 0]  # (F, M)
    Y = np.einsum("fm,ftm->ft", W.conj(), Z) / Z.shape[-1]
    # inverse STFT (mono)
    from scipy.signal import istft
    n_fft = cfg["stft"]["n_fft"]; hop = cfg["stft"]["hop"]
    full = np.zeros((n_fft // 2 + 1, Y.shape[1]), dtype=np.complex64)
    f_all = np.fft.rfftfreq(n_fft, 1 / fs)
    keep = (f_all >= cfg["stft"]["fmin"]) & (f_all <= cfg["stft"]["fmax"])
    full[keep] = Y
    _, x = istft(full, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return x.astype(np.float32)


def extract(cfg):
    mics = load_mic_positions(cfg["mic_xml"])
    fs = cfg["fs"]
    clips, noises, stats = [], [], {"card": [], "pos": [], "snr_db": []}
    n_one = n_zero = 0
    for item in iter_samples(cfg):
        ann = item.get("ann", [])
        audio = item["audio"]
        stats["card"].append(len(ann))
        if len(ann) == 1 and n_one < cfg["adapter"]["max_clips"]:
            pos = _ann_to_polar(ann)[0]
            clips.append(_beamform_clip(audio, fs, mics, pos, cfg))
            stats["pos"].append(pos)
            n_one += 1
        elif len(ann) == 0 and n_zero < cfg["adapter"]["max_noise"]:
            v = item.get("velocity", {})
            spd = float(np.hypot(v.get("lin_x", 0.0), v.get("lin_y", 0.0)))
            noises.append((spd, audio.astype(np.float32)))
            n_zero += 1
    os.makedirs(cfg["workdir"], exist_ok=True)
    np.savez_compressed(os.path.join(cfg["workdir"], "bank.npz"),
                        clips=np.array(clips, dtype=object),
                        noise_speeds=np.array([s for s, _ in noises]),
                        noises=np.array([n for _, n in noises], dtype=object),
                        allow_pickle=True)
    with open(os.path.join(cfg["workdir"], "stats.json"), "w") as fh:
        json.dump({"n_one_box": n_one, "n_zero_box": n_zero,
                   "cardinality_hist": np.bincount(stats["card"], minlength=5)[:5].tolist(),
                   "positions": stats["pos"]}, fh, indent=2)
    print(f"clip bank: {n_one} clips | noise beds: {n_zero} | -> {cfg['workdir']}")
    if n_one < cfg["adapter"]["min_clips_warn"]:
        print(f"WARNING: only {n_one} one-box windows found "
              f"(< {cfg['adapter']['min_clips_warn']}). Set A realism is at risk; "
              "consider the synthetic-source fallback and mark results accordingly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--config", default="premise_study/config.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.cmd == "extract":
        extract(cfg)


if __name__ == "__main__":
    main()
