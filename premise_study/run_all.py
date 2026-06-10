"""run_all.py — execute measurements M1–M6 and produce the three decision plots.

Usage:
    python -m premise_study.run_all --config premise_study/config.yaml [--smoke]

--smoke replaces the real clip bank and mic XML with synthetic stand-ins and
shrinks scene counts, so the whole pipeline can be verified end-to-end in
minutes before any real data is touched. Smoke results carry no scientific
meaning and are written to <workdir>_smoke.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import yaml

from .geometry import (load_mic_positions, synthetic_array, polar_grid,
                       steering_matrix)
from .render import stft_multi
from .scenes import set_a, set_b, set_b3
from .glimpse import tf_energy, glimpse_measures
from .detector import detect_scene, srp_map, pick_peaks, match
from .crb import crb_maps, r_max_for_relative_error


def _synthetic_bank(fs, mics, n_clips=24, n_noise=6, T=None, seed=0):
    """Smoke-test stand-in: band-limited noise 'vehicles' + diffuse noise beds."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    T = T or fs  # 1 s
    clips = []
    for i in range(n_clips):
        lo = rng.uniform(80, 400); hi = lo * rng.uniform(3, 10)
        b, a = butter(2, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], "band")
        clips.append((i, lfilter(b, a, rng.standard_normal(2 * T)).astype(np.float32)))
    noises = []
    for i in range(n_noise):
        n = 0.05 * rng.standard_normal((T, len(mics))).astype(np.float32)
        noises.append((float(rng.choice([0, 15, 30, 50])), n))
    return {"clips": clips, "noises": noises}


def _samplers(stats=None, r_lim=(4, 45)):
    def pos(rng, N):
        return [(rng.uniform(-np.pi, np.pi), rng.uniform(*r_lim)) for _ in range(N)]
    def snr(rng, N):
        base = rng.uniform(0, 18, size=N)
        return list(np.sort(base)[::-1])
    return pos, snr


def run(cfg, smoke=False):
    t0 = time.time()
    wd = cfg["workdir"] + ("_smoke" if smoke else "")
    os.makedirs(wd, exist_ok=True)
    fs = cfg["fs"]

    # --- geometry & steering ------------------------------------------------
    if smoke or not os.path.exists(cfg["mic_xml"]):
        mics = synthetic_array()
        if not smoke:
            raise FileNotFoundError(f"mic_xml not found: {cfg['mic_xml']}")
        print("[smoke] synthetic 32-mic array")
    else:
        mics = load_mic_positions(cfg["mic_xml"])
    g = cfg["grid"]
    thetas, rs = polar_grid(g["n_theta"], g["n_r"], g["r_min"], g["r_max"], g["log_range"])

    # probe STFT for frequency axis
    probe = np.zeros((fs, len(mics)), dtype=np.float32)
    _, freqs = stft_multi(probe, fs, **cfg["stft"])
    W = steering_matrix(mics, freqs, thetas, rs)
    steer_at = lambda p: steering_matrix(mics, freqs, np.array([p[0]]),
                                         np.array([p[1]]))[0, 0]

    # --- bank & scenes -------------------------------------------------------
    if smoke:
        bank = _synthetic_bank(fs, mics)
        nA, perB, nB3 = 40, 2, 12
    else:
        bz = np.load(os.path.join(cfg["workdir"], "bank.npz"), allow_pickle=True)
        bank = {"clips": [(i, c) for i, c in enumerate(bz["clips"])],
                "noises": list(zip(bz["noise_speeds"], bz["noises"]))}
        nA, perB, nB3 = cfg["sets"]["a_scenes"], cfg["sets"]["b_per_cell"], cfg["sets"]["b3_scenes"]
    pos_s, snr_s = _samplers()
    card = np.array([0.35, 0.35, 0.2, 0.1])
    A = set_a(bank, nA, fs, mics, card, pos_s, snr_s, seed=1)
    B = set_b(bank, fs, mics, per_cell=perB, seed=2)
    B3 = set_b3(bank, fs, mics, n_scenes=nB3, seed=3)
    print(f"scenes: A={len(A)} B={len(B)} B3={len(B3)}  ({time.time()-t0:.0f}s)")

    def _done(name):
        return os.path.exists(os.path.join(wd, name))

    # --- M1 + M5: glimpse ----------------------------------------------------
    rows = []
    if not _done("m1_glimpse.json"):
        for sc in A:
            Es = [tf_energy(im, fs, **cfg["stft"]) for im in sc["images"]]
            En = tf_energy(sc["noise"], fs, **cfg["stft"])
            Ge, Gl, H, rank = glimpse_measures(Es, En, beta=cfg["glimpse"]["beta"])
            for j in range(len(Es)):
                rows.append({"set": "A", "n": sc["meta"]["n"], "rank": int(rank[j]),
                             "G_empty": float(Ge[j]), "G_louder": float(Gl[j]),
                             "headroom": float(H[j])})
        json.dump(rows, open(os.path.join(wd, "m1_glimpse.json"), "w"))
    print(f"M1 done ({time.time()-t0:.0f}s)")

    # --- detector threshold: tune once on a split, freeze --------------------
    tune = A[: min(cfg["sets"]["tuning_split"], max(10, len(A) // 10))]
    eval_A = A[len(tune):]
    best, best_f1 = cfg["detector"]["thresh_rel"], -1
    for tr in (0.2, 0.3, 0.35, 0.45, 0.6):
        tp = fp = fn = 0
        for sc in tune:
            Z, _ = stft_multi(sc["mixture"], fs, **cfg["stft"])
            preds = pick_peaks(srp_map(Z, W), thetas, rs, tr)
            h, f = match(preds, sc["positions"], cfg["detector"]["gate_deg"],
                         cfg["detector"]["gate_m"])
            tp += h.sum(); fn += (~h).sum(); fp += f
        f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
        if f1 > best_f1:
            best, best_f1 = tr, f1
    cfg["detector"]["thresh_rel"] = best
    print(f"threshold frozen at {best} (tuning F1={best_f1:.3f}, {time.time()-t0:.0f}s)")

    # --- M2: plain / oracle-SIC / oracle-null on A(eval) + B ------------------
    def run_modes(scenes, tag):
        if _done(f"m2_{tag}.json"):
            return json.load(open(os.path.join(wd, f"m2_{tag}.json")))
        rec = []
        for sc in scenes:
            row = {"meta": sc["meta"], "ranks": sc["ranks"].tolist()}
            Es = [tf_energy(im, fs, **cfg["stft"]) for im in sc["images"]]
            En = tf_energy(sc["noise"], fs, **cfg["stft"])
            Ge, _, _, _ = glimpse_measures(Es, En)
            row["G_empty"] = Ge.tolist()
            for mode in ("plain", "oracle_sic", "oracle_null"):
                h, _ = detect_scene(sc, W, thetas, rs, mode, fs, cfg["stft"],
                                    cfg["detector"]["thresh_rel"], steer_at)
                row[mode] = h.tolist()
            rec.append(row)
        json.dump(rec, open(os.path.join(wd, f"m2_{tag}.json"), "w"))
        return rec
    m2a = run_modes(eval_A, "setA")
    m2b = run_modes(B, "setB")
    print(f"M2 done ({time.time()-t0:.0f}s)")

    # --- M3: order test on B3 -------------------------------------------------
    m3 = []
    if _done("m3_order.json"):
        m3 = json.load(open(os.path.join(wd, "m3_order.json")))
    for sc in (B3 if not m3 else []):
        row = {}
        for name, order_mul in (("dom", 1), ("rev", -1)):
            sc2 = dict(sc); sc2["ranks"] = sc["ranks"] * order_mul if order_mul == 1 \
                else (sc["ranks"].max() - sc["ranks"])
            h, _ = detect_scene(sc2, W, thetas, rs, "oracle_sic", fs, cfg["stft"],
                                cfg["detector"]["thresh_rel"], steer_at)
            row[name] = h.tolist()
        row["ranks"] = sc["ranks"].tolist()
        m3.append(row)
    json.dump(m3, open(os.path.join(wd, "m3_order.json"), "w"))
    print(f"M3 done ({time.time()-t0:.0f}s)")

    # --- M4: CRB ---------------------------------------------------------------
    m4 = {}
    if _done("m4_crb.json"):
        m4 = json.load(open(os.path.join(wd, "m4_crb.json")))
    for snr in ((0.0, cfg["decision"]["P4_snr_db"], 20.0) if not m4 else ()):
        Sth, Sr = crb_maps(mics, thetas[::max(1, len(thetas)//36)], rs, snr)
        m4[str(snr)] = {"sigma_theta_deg": Sth.tolist(), "sigma_r_m": Sr.tolist()}
    if "r_max_25" not in m4:
        m4["r_max_25"] = r_max_for_relative_error(
            mics, cfg["decision"]["P4_snr_db"], cfg["decision"]["P4_rel_err"])
        m4["rs"] = rs.tolist()
        json.dump(m4, open(os.path.join(wd, "m4_crb.json"), "w"))
    print(f"M4 done; r_max(25%)={m4['r_max_25']:.1f} m ({time.time()-t0:.0f}s)")

    # --- M6: few-mic sweep on B -------------------------------------------------
    m6 = {}
    if _done("m6_fewmic.json"):
        m6 = json.load(open(os.path.join(wd, "m6_fewmic.json")))
    B6 = B[:16] if smoke else B[:: max(1, len(B) // 40)]
    for M in (4, 8, 16, len(mics)):
        if str(M) in m6:
            continue
        sub = mics[np.linspace(0, len(mics) - 1, M).astype(int)]
        Wm = steering_matrix(sub, freqs, thetas, rs)
        sa = lambda p, _W=sub: steering_matrix(_W, freqs, np.array([p[0]]),
                                               np.array([p[1]]))[0, 0]
        rec = []
        for sc in B6:
            sc_sub = dict(sc)
            sc_sub["mixture"] = sc["mixture"][:, np.linspace(0, len(mics)-1, M).astype(int)]
            sc_sub["images"] = [im[:, np.linspace(0, len(mics)-1, M).astype(int)]
                                for im in sc["images"]]
            row = {}
            for mode in ("plain", "oracle_sic", "oracle_null"):
                h, _ = detect_scene(sc_sub, Wm, thetas, rs, mode, fs, cfg["stft"],
                                    cfg["detector"]["thresh_rel"], sa)
                row[mode] = h.tolist()
            row["ranks"] = sc["ranks"].tolist()
            rec.append(row)
        m6[str(M)] = rec
        json.dump(m6, open(os.path.join(wd, "m6_fewmic.json"), "w"))
        print(f"M6[{M} mics] done ({time.time()-t0:.0f}s)")
    print(f"M6 done ({time.time()-t0:.0f}s)")

    from .plots import make_all_plots
    make_all_plots(wd, cfg)
    print(f"plots written to {wd}/  ({time.time()-t0:.0f}s)")
    return wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="premise_study/config.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    wd = run(cfg, smoke=args.smoke)
    from .decide import decide
    decide(wd, cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
