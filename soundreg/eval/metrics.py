"""Matching and metrics. Pure: tensors/arrays in, numbers out — no
logging, no model imports, no I/O.

PUBLIC SURFACE
--------------
    match_scene            Hungarian matching for one scene.
    evaluate               Aggregate metrics + flat per-GT arrays.
    stratified_recall      Recall by SNR bin and by scene cardinality
                           (the headline plot data).
    apply_score_threshold  Thin a score-based prediction.
    sweep_score_threshold  Best-F1 operating point (chosen on val ONLY).
    eos_calibration        P(EOS) vs should-have-stopped records.

MATCHING RULE (PLAN.md, fixed project-wide)
-------------------------------------------
A prediction matches a GT object iff

    polar_distance(pred, gt) <= dist_thresh   AND   classes agree,

assignment by Hungarian on the distance matrix (invalid pairs get a
prohibitive cost and are dropped after assignment). From the matches:

    precision / recall / F1     micro over all scenes
    cardinality_mae             mean |N_hat - N|
    range_mae                   meters, matched pairs only
    azimuth_mae_deg             circular, matched pairs only

WHY PER-GT ARRAYS COME BACK TOO
-------------------------------
The headline test is recall STRATIFIED by per-source SNR and by number
of concurrent sources. Stratification needs one row per GT object
(matched?, snr_db, scene size), not aggregates — `evaluate` returns
those flat arrays and `stratified_recall` cuts them into bins. They are
also dumped to stratified.npz so bins can be re-cut without re-running
inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..data.polar_grid import polar_distance
from ..types import SceneGT, ScenePrediction
from ..utils import circular_diff_deg

# Cost assigned to forbidden (pred, gt) pairs. Anything matched at this
# cost is discarded after the assignment, so its exact value only needs
# to dominate every real distance.
_INVALID = 1e9


# =====================================================================
# Matching
# =====================================================================
def match_scene(
    pred: ScenePrediction,
    gt: SceneGT,
    dist_thresh: float,
    require_class: bool = True,
) -> Dict:
    """Hungarian matching of one scene.

    Args:
        pred: K predicted objects.
        gt: N ground-truth objects.
        dist_thresh: Max BEV distance (m) for a valid match.
        require_class: Class agreement gate (moot for single-class data).

    Returns:
        dict with:
            tp, fp, fn      ints
            gt_matched      (N,) bool — per-GT recall flags
            r_err           (tp,) |range error| of matched pairs, m
            theta_err       (tp,) |circular azimuth error|, deg
    """
    k, n = pred.n, gt.n
    if n == 0 or k == 0:
        return {
            "tp": 0,
            "fp": k,
            "fn": n,
            "gt_matched": np.zeros(n, dtype=bool),
            "r_err": np.zeros(0),
            "theta_err": np.zeros(0),
        }
    dist = polar_distance(
        pred.r[:, None], pred.theta[:, None], gt.r[None, :], gt.theta[None, :]
    )
    valid = dist <= dist_thresh
    if require_class:
        valid &= pred.classes[:, None] == gt.classes[None, :]
    cost = np.where(valid, dist, _INVALID)
    rows, cols = linear_sum_assignment(cost)
    ok = cost[rows, cols] < _INVALID
    rows, cols = rows[ok], cols[ok]

    gt_matched = np.zeros(n, dtype=bool)
    gt_matched[cols] = True
    return {
        "tp": len(rows),
        "fp": k - len(rows),
        "fn": n - len(rows),
        "gt_matched": gt_matched,
        "r_err": np.abs(pred.r[rows] - gt.r[cols]),
        "theta_err": np.abs(circular_diff_deg(pred.theta[rows], gt.theta[cols])),
    }


# =====================================================================
# Aggregation
# =====================================================================
def evaluate(
    preds: List[ScenePrediction],
    gts: List[SceneGT],
    dist_thresh: float,
    require_class: bool = True,
) -> Dict:
    """Aggregate metrics over a list of scenes.

    Returns the scalar metrics plus flat per-GT arrays (gt_matched,
    gt_snr_db, gt_n_sources, gt_nlos) for stratification. GT objects
    without an SNR annotation (real data before M1 beamforming) carry
    NaN and simply fall outside every SNR bin.
    """
    tp = fp = fn = 0
    card_errs, r_errs, t_errs = [], [], []
    gt_matched, gt_snr, gt_nsrc, gt_nlos = [], [], [], []

    for pred, gt in zip(preds, gts):
        m = match_scene(pred, gt, dist_thresh, require_class)
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
        card_errs.append(abs(pred.n - gt.n))
        r_errs.append(m["r_err"])
        t_errs.append(m["theta_err"])
        gt_matched.append(m["gt_matched"])
        gt_snr.append(gt.snr_db if gt.snr_db is not None else np.full(gt.n, np.nan))
        gt_nsrc.append(np.full(gt.n, gt.n))
        gt_nlos.append(gt.nlos if gt.nlos is not None else np.zeros(gt.n, dtype=bool))

    r_errs = np.concatenate(r_errs) if r_errs else np.zeros(0)
    t_errs = np.concatenate(t_errs) if t_errs else np.zeros(0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cardinality_mae": float(np.mean(card_errs)) if card_errs else 0.0,
        "range_mae": float(r_errs.mean()) if len(r_errs) else float("nan"),
        "azimuth_mae_deg": float(t_errs.mean()) if len(t_errs) else float("nan"),
        "n_scenes": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        # Flat per-GT arrays for stratified recall.
        "gt_matched": np.concatenate(gt_matched) if gt_matched else np.zeros(0, bool),
        "gt_snr_db": np.concatenate(gt_snr) if gt_snr else np.zeros(0),
        "gt_n_sources": np.concatenate(gt_nsrc) if gt_nsrc else np.zeros(0),
        "gt_nlos": np.concatenate(gt_nlos) if gt_nlos else np.zeros(0, bool),
    }


def stratified_recall(result: Dict, snr_edges: List[float]) -> Dict:
    """Cut the per-GT arrays into recall-per-bin — the headline numbers.

    Args:
        result: Output of `evaluate`.
        snr_edges: Bin edges in dB, e.g. [-30, -10, -5, 0, 5, 10, 40].

    Returns:
        {"by_snr": {label: {recall, n}}, "by_n_sources": {N: {recall, n}}}
        Bins with no GT report recall = NaN rather than a misleading 0.
    """
    matched = result["gt_matched"]
    snr = result["gt_snr_db"]
    nsrc = result["gt_n_sources"]

    by_snr = {}
    for lo, hi in zip(snr_edges[:-1], snr_edges[1:]):
        sel = (snr >= lo) & (snr < hi)
        by_snr[f"[{lo},{hi})"] = {
            "recall": float(matched[sel].mean()) if sel.any() else float("nan"),
            "n": int(sel.sum()),
        }
    by_n = {}
    for n in np.unique(nsrc).astype(int):
        sel = nsrc == n
        by_n[int(n)] = {
            "recall": float(matched[sel].mean()),
            "n": int(sel.sum()),
        }
    return {"by_snr": by_snr, "by_n_sources": by_n}


# =====================================================================
# Operating point for score-based detectors
# =====================================================================
def apply_score_threshold(pred: ScenePrediction, thr: float) -> ScenePrediction:
    """Keep candidates with score >= thr. No-op for score-less models."""
    if pred.scores is None:
        return pred
    keep = pred.scores >= thr
    return ScenePrediction(
        classes=pred.classes[keep],
        r=pred.r[keep],
        theta=pred.theta[keep],
        scores=pred.scores[keep],
        extras=pred.extras,
    )


def sweep_score_threshold(
    preds: List[ScenePrediction],
    gts: List[SceneGT],
    dist_thresh: float,
    n_grid: int = 40,
) -> Tuple[float, float]:
    """Best-F1 score threshold over a quantile grid of observed scores.

    The fairness rule: call this on VAL predictions only, then apply the
    returned threshold to test (mirrors the AutoReg3D paper's treatment
    of thresholded baselines). The grid tops out at the 0.99 quantile so
    at least some predictions always survive.

    Returns:
        (threshold, f1_at_threshold). (-inf, 0.0) when there are no
        scored predictions to sweep.
    """
    all_scores = np.concatenate(
        [p.scores for p in preds if p.scores is not None and p.n > 0]
    )
    if len(all_scores) == 0:
        return -np.inf, 0.0
    grid = np.quantile(all_scores, np.linspace(0.0, 0.99, n_grid))
    best_thr, best_f1 = -np.inf, -1.0
    for thr in np.unique(grid):
        thinned = [apply_score_threshold(p, thr) for p in preds]
        f1 = evaluate(thinned, gts, dist_thresh)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, best_f1


# =====================================================================
# EOS calibration
# =====================================================================
def eos_calibration(preds: List[ScenePrediction], gts: List[SceneGT]) -> Dict:
    """Collect (P(EOS), should_stop) at every object boundary visited.

    should_stop is exact under the ordered-emission semantics: after k
    objects are out, stopping is correct iff k >= N_gt.

    Watch this from epoch 1. The known failure mode of the EOS mechanism
    is firing early and silently dropping the quiet masked sources this
    project is about — it shows up here (high P(EOS) where should_stop
    is 0) long before it is legible in recall curves.

    Returns:
        dict with n_records, brier, the two conditional means, and the
        raw (n_records, 2) [p_eos, should_stop] array for plotting.
    """
    p_eos, should_stop = [], []
    for pred, gt in zip(preds, gts):
        probs = pred.extras.get("eos_probs")
        if probs is None:
            continue
        for k, p in enumerate(np.asarray(probs)):
            p_eos.append(float(p))
            should_stop.append(float(k >= gt.n))
    p = np.asarray(p_eos)
    y = np.asarray(should_stop)
    out: Dict = {"n_records": len(p)}
    if len(p):
        out["brier"] = float(np.mean((p - y) ** 2))
        out["mean_p_eos_when_should_continue"] = (
            float(p[y == 0].mean()) if (y == 0).any() else float("nan")
        )
        out["mean_p_eos_when_should_stop"] = (
            float(p[y == 1].mean()) if (y == 1).any() else float("nan")
        )
        out["records"] = np.stack([p, y], axis=1)
    return out
