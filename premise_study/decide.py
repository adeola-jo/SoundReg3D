"""decide.py — evaluate the pre-registered rules P1–P5 and print the branch.

The conclusion is computed, not narrated. Run after run_all.py:
    python -m premise_study.decide --workdir premise_out --config premise_study/config.yaml
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import yaml


def _recall(rec, mode, only_hidden=None, hidden_thr=0.2, quietest_only=False):
    hits = []
    for r in rec:
        ranks = r.get("ranks", list(range(len(r[mode]))))
        for j in range(len(r[mode])):
            if only_hidden is True and r["G_empty"][j] >= hidden_thr:
                continue
            if quietest_only and ranks[j] != max(ranks):
                continue
            hits.append(r[mode][j])
    return float(np.mean(hits)) if hits else float("nan"), len(hits)


def decide(wd, cfg, smoke=False):
    D = cfg["decision"]
    out = {}

    rows = json.load(open(os.path.join(wd, "m1_glimpse.json")))
    quiet = [r for r in rows if r["n"] >= 2 and r["rank"] >= 1]
    hidden_frac = (np.mean([r["G_empty"] < cfg["glimpse"]["hidden_below"]
                            for r in quiet]) if quiet else float("nan"))
    out["P1"] = {"hidden_fraction": hidden_frac,
                 "verdict": ("PASS" if hidden_frac >= D["P1_hidden_fraction_pass"]
                             else "FAIL" if hidden_frac < D["P1_hidden_fraction_fail"]
                             else "GRAY")}

    rec = json.load(open(os.path.join(wd, "m2_setA.json")))
    thr = cfg["glimpse"]["hidden_below"]
    r_plain, n_h = _recall(rec, "plain", only_hidden=True, hidden_thr=thr)
    r_sic, _ = _recall(rec, "oracle_sic", only_hidden=True, hidden_thr=thr)
    r_null, _ = _recall(rec, "oracle_null", only_hidden=True, hidden_thr=thr)
    gain = 100 * (r_sic - r_plain)
    out["P2a"] = {"plain": r_plain, "oracle_sic": r_sic, "gain_pts": gain,
                  "n_hidden_sources": n_h,
                  "verdict": ("PASS" if gain >= D["P2a_sic_gain_pts_pass"]
                              else "FAIL" if gain < D["P2a_sic_gain_pts_fail"]
                              else "GRAY")}
    null_frac = ((r_null - r_plain) / max(1e-9, r_sic - r_plain)
                 if r_sic > r_plain else float("nan"))
    out["P2b"] = {"oracle_null": r_null, "null_recovers_frac": null_frac,
                  "spatial_suffices": bool(null_frac == null_frac and
                                           null_frac >= D["P2b_null_recovers_frac"])}

    m3 = json.load(open(os.path.join(wd, "m3_order.json")))
    def quietest_recall(key):
        h = [r[key][int(np.argmax(r["ranks"]))] for r in m3]
        return float(np.mean(h)) if h else float("nan")
    g3 = 100 * (quietest_recall("dom") - quietest_recall("rev"))
    out["P3"] = {"dom": quietest_recall("dom"), "rev": quietest_recall("rev"),
                 "gain_pts": g3,
                 "verdict": ("PASS" if g3 >= D["P3_order_gain_pts_pass"]
                             else "FAIL" if g3 < D["P3_order_gain_pts_fail"]
                             else "GRAY")}

    m4 = json.load(open(os.path.join(wd, "m4_crb.json")))
    out["P4"] = {"r_max_25": m4["r_max_25"],
                 "verdict": ("PASS" if m4["r_max_25"] >= D["P4_rmax_m_pass"]
                             else "FAIL" if m4["r_max_25"] < D["P4_rmax_m_fail"]
                             else "GRAY")}

    # ---- branch (precedence: P2a fail > P2b > P4 > P1, per PROTOCOL §4) ----
    if out["P2a"]["verdict"] == "FAIL":
        branch = ("PREMISE DEAD for this data: oracle explaining-away does not "
                  "recover hidden sources. Investigate before any pivot.")
    elif out["P2b"]["spatial_suffices"]:
        branch = ("FEW-MIC BRANCH: spatial nulling alone recovers the gain at "
                  "M=32 — sequencing matters in the cheap-array regime (see M6).")
    elif out["P4"]["verdict"] == "FAIL":
        branch = "RESCOPE BRANCH: range beyond reach — azimuth + near/far classes."
    elif out["P1"]["verdict"] == "FAIL":
        branch = "EGO-NOISE BRANCH: inter-source hiding is rare; the masker is the ego."
    else:
        branch = "BUILD v2: premise holds. Proceed with SoundReg and the theory section."

    out["branch"] = branch
    tag = " (SMOKE — no scientific meaning)" if smoke else ""
    print("\n================ DECISION" + tag + " ================")
    for k in ("P1", "P2a", "P2b", "P3", "P4"):
        print(f"{k}: {json.dumps(out[k])}")
    print("BRANCH:", branch)
    print("====================================================")
    json.dump(out, open(os.path.join(wd, "decision.json"), "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="premise_out")
    ap.add_argument("--config", default="premise_study/config.yaml")
    args = ap.parse_args()
    decide(args.workdir, yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
