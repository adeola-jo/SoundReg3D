"""plots.py — the three decision plots (PROTOCOL.md §5)."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(wd, name):
    return json.load(open(os.path.join(wd, name)))


def plot1_hiding(wd, cfg):
    rows = _load(wd, "m1_glimpse.json")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for rank, color in ((0, "#444"), (1, "#D9822B"), (2, "#A33E22")):
        g = sorted(r["G_empty"] for r in rows if r["rank"] == rank and r["n"] >= 2)
        if g:
            ax.plot(g, np.linspace(0, 1, len(g)),
                    label=f"rank {rank+1} ({'loudest' if rank==0 else 'quieter'})",
                    color=color)
    ax.axvline(cfg["glimpse"]["hidden_below"], ls="--", c="k", lw=1,
               label=f"hidden (G<{cfg['glimpse']['hidden_below']})")
    ax.set_xlabel("glimpse measure G(\u2205)"); ax.set_ylabel("CDF")
    ax.set_title("Plot 1 — how much hiding is there?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(wd, "plot1_hiding.png"), dpi=160); plt.close(fig)


def plot2_headroom(wd, cfg):
    rec = _load(wd, "m2_setA.json")
    bins = np.linspace(0, 1, 9)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for mode, color in (("plain", "#444"), ("oracle_null", "#2E8F8A"),
                        ("oracle_sic", "#A33E22")):
        x, y, lo, hi = [], [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            hits = [r[mode][j] for r in rec for j in range(len(r[mode]))
                    if b0 <= r["G_empty"][j] < b1]
            if len(hits) < 10:
                continue
            hits = np.array(hits, float)
            bs = [np.mean(np.random.choice(hits, len(hits))) for _ in range(1000)]
            x.append((b0 + b1) / 2); y.append(hits.mean())
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.plot(x, y, "-o", ms=3, color=color, label=mode)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("glimpse measure G(\u2205)  (low = hidden)")
    ax.set_ylabel("per-source recall")
    ax.set_title("Plot 2 — does explaining away help?")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(wd, "plot2_headroom.png"), dpi=160); plt.close(fig)


def plot3_crb(wd, cfg):
    m4 = _load(wd, "m4_crb.json")
    rs = np.array(m4["rs"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for snr, ls in (("0.0", ":"), (str(float(cfg["decision"]["P4_snr_db"])), "-"),
                    ("20.0", "--")):
        if snr not in m4:
            continue
        Sth = np.median(np.array(m4[snr]["sigma_theta_deg"]), axis=0)
        Sr = np.median(np.array(m4[snr]["sigma_r_m"]), axis=0)
        axes[0].plot(rs, Sth, ls, c="#25303B", label=f"{snr} dB")
        axes[1].plot(rs, Sr / rs, ls, c="#25303B", label=f"{snr} dB")
    axes[0].axhline(5.0, c="#A33E22", lw=1, label="5° bin")
    axes[0].axhline(1.5, c="#2E8F8A", lw=1, label="1.5° bin")
    axes[0].set_xlabel("range (m)"); axes[0].set_ylabel("CRB σθ (deg)")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    axes[1].axhline(cfg["decision"]["P4_rel_err"], c="#A33E22", lw=1,
                    label=f"{int(cfg['decision']['P4_rel_err']*100)}% rel. err")
    axes[1].axvline(m4["r_max_25"], c="#2E8F8A", lw=1,
                    label=f"r_max={m4['r_max_25']:.0f} m")
    axes[1].set_xlabel("range (m)"); axes[1].set_ylabel("CRB σr / r")
    axes[1].set_yscale("log"); axes[1].legend(fontsize=7)
    fig.suptitle("Plot 3 — what physics allows")
    fig.tight_layout()
    fig.savefig(os.path.join(wd, "plot3_crb.png"), dpi=160); plt.close(fig)


def make_all_plots(wd, cfg):
    plot1_hiding(wd, cfg)
    plot2_headroom(wd, cfg)
    plot3_crb(wd, cfg)
