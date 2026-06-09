"""Headline comparison across runs: recall stratified by per-source SNR and by
number of concurrent sources (the falsifier test of PLAN.md §1).

Usage:
    python scripts/compare_runs.py runs/soundreg_synth/seed0 runs/srp_peaks/seed0 ...
        [--snr-edges -30 -10 -5 0 5 10 40] [--plot out.png]

Reads each run's stratified.npz (per-GT match flags + SNR + scene size) so the
SNR bins can be re-cut here without re-running eval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> dict:
    z = np.load(run_dir / "stratified.npz")
    return {k: z[k] for k in z.files}


def recall_by_bins(matched, values, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values < hi)
        out.append((float(matched[sel].mean()) if sel.any() else np.nan, int(sel.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--snr-edges", nargs="+", type=float,
                    default=[-30, -10, -5, 0, 5, 10, 40])
    ap.add_argument("--plot", default=None, help="optional .png path (matplotlib)")
    args = ap.parse_args()

    edges = args.snr_edges
    runs = {}
    for r in args.runs:
        p = Path(r)
        runs[f"{p.parent.name}/{p.name}"] = load_run(p)

    # ---- recall vs SNR table -------------------------------------------------
    headers = [f"[{lo:g},{hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
    name_w = max(len(n) for n in runs) + 2
    print("\nRecall by per-source SNR (dB):")
    print(" " * name_w + "  ".join(f"{h:>10}" for h in headers))
    counts = None
    for name, d in runs.items():
        cells = recall_by_bins(d["gt_matched"], d["gt_snr_db"], edges)
        counts = [c for _, c in cells]
        print(f"{name:<{name_w}}" + "  ".join(f"{rec:>10.3f}" for rec, _ in cells))
    print(f"{'(n GT)':<{name_w}}" + "  ".join(f"{c:>10d}" for c in counts))

    # ---- recall vs concurrency ----------------------------------------------
    all_n = sorted(set(np.unique(next(iter(runs.values()))["gt_n_sources"]).astype(int)))
    print("\nRecall by number of concurrent sources:")
    print(" " * name_w + "  ".join(f"{n:>8d}" for n in all_n))
    for name, d in runs.items():
        row = []
        for n in all_n:
            sel = d["gt_n_sources"] == n
            row.append(float(d["gt_matched"][sel].mean()) if sel.any() else np.nan)
        print(f"{name:<{name_w}}" + "  ".join(f"{v:>8.3f}" for v in row))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = [(lo + hi) / 2 for lo, hi in zip(edges[:-1], edges[1:])]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, d in runs.items():
            cells = recall_by_bins(d["gt_matched"], d["gt_snr_db"], edges)
            axes[0].plot(centers, [r for r, _ in cells], marker="o", label=name)
            row = []
            for n in all_n:
                sel = d["gt_n_sources"] == n
                row.append(float(d["gt_matched"][sel].mean()) if sel.any() else np.nan)
            axes[1].plot(all_n, row, marker="s", label=name)
        axes[0].set_xlabel("per-source SNR (dB)")
        axes[0].set_ylabel("recall")
        axes[0].set_title("Recall vs SNR (headline test)")
        axes[1].set_xlabel("# concurrent sources")
        axes[1].set_title("Recall vs overlap")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


if __name__ == "__main__":
    main()
