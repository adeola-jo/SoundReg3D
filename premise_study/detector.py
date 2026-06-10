"""detector.py — classical SRP-PHAT detector on the polar grid, three modes.

plain       : detect on the full mixture (what a parallel model faces)
oracle_sic  : before detecting source k+1, subtract ground-truth array images
              of the k already-handled (louder) sources       (ceiling for AR)
oracle_null : keep the mixture; orthogonally project out the known steering
              direction of louder sources, per frequency      (spatial-only)

Torch (GPU) is used for the grid scan when available; numpy otherwise.
The SRP map doubles as the dominance score d_i of the brief (Eq. 3).
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    _TORCH = torch.cuda.is_available()
except Exception:  # torch absent
    torch = None
    _TORCH = False


def _phat(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return Z / (np.abs(Z) + eps)


def srp_map(Z: np.ndarray, W: np.ndarray, phat: bool = True) -> np.ndarray:
    """Z: (F, T, M) mixture STFT; W: (nT, nR, F, M) steering -> SRP (nT, nR).

    P(g) = sum_{f,t} | w(g,f)^H z(f,t) |^2
    """
    X = _phat(Z) if phat else Z
    if _TORCH:
        dev = "cuda"
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
        Wt = torch.from_numpy(np.ascontiguousarray(W)).to(dev)
        # (nT,nR,F,M) x (F,T,M) -> (nT,nR,F,T)
        resp = torch.einsum("abfm,ftm->abft", Wt.conj(), Xt)
        P = (resp.real ** 2 + resp.imag ** 2).sum(dim=(2, 3))
        return P.float().cpu().numpy()
    resp = np.einsum("abfm,ftm->abft", W.conj(), X, optimize=True)
    return (np.abs(resp) ** 2).sum(axis=(2, 3)).astype(np.float32)


def null_project(Z: np.ndarray, steer_cols: list[np.ndarray]) -> np.ndarray:
    """Project the mixture STFT onto the orthogonal complement of the louder
    sources' steering vectors, per frequency.

    steer_cols: list of (F, M) steering vectors (one per louder source).
    """
    if not steer_cols:
        return Z
    F, T, M = Z.shape
    Zo = Z.copy()
    A = np.stack(steer_cols, axis=-1)                 # (F, M, K)
    for f in range(F):
        Af = A[f]                                      # (M, K)
        # P_perp = I - Af (Af^H Af)^-1 Af^H   (regularized)
        G = Af.conj().T @ Af + 1e-6 * np.eye(Af.shape[1])
        P = Af @ np.linalg.solve(G, Af.conj().T)
        Zo[f] = Zo[f] @ (np.eye(M) - P).T
    return Zo


def pick_peaks(P: np.ndarray, thetas: np.ndarray, rs: np.ndarray,
               thresh_rel: float, max_peaks: int = 6,
               gate_deg: float = 10.0, gate_m: float = 8.0):
    """Greedy local-max picking with polar NMS. Returns [(theta, r, score), ...].

    thresh_rel is relative to the map max (frozen once on the tuning split).
    """
    out = []
    Pw = P.copy()
    thr = thresh_rel * P.max() if P.max() > 0 else np.inf
    for _ in range(max_peaks):
        idx = np.unravel_index(np.argmax(Pw), Pw.shape)
        score = Pw[idx]
        if score < thr:
            break
        th, r = thetas[idx[0]], rs[idx[1]]
        out.append((float(th), float(r), float(score)))
        dth = np.angle(np.exp(1j * (thetas[:, None] - th)))
        sup = (np.abs(np.degrees(dth)) < gate_deg) & (np.abs(rs[None, :] - r) < gate_m)
        Pw[sup] = -np.inf
    return out


def match(preds, gts, gate_deg: float = 5.0, gate_m: float = 5.0):
    """Greedy matching by angular distance within gates -> per-GT hit flags."""
    hits = np.zeros(len(gts), dtype=bool)
    used = np.zeros(len(preds), dtype=bool)
    for gi, (gth, gr) in enumerate(gts):
        best, bj = np.inf, -1
        for pj, (pth, pr, _) in enumerate(preds):
            if used[pj]:
                continue
            dth = abs(np.degrees(np.angle(np.exp(1j * (pth - gth)))))
            if dth <= gate_deg and abs(pr - gr) <= gate_m and dth < best:
                best, bj = dth, pj
        if bj >= 0:
            hits[gi] = True
            used[bj] = True
    n_fp = int((~used).sum())
    return hits, n_fp


def detect_scene(scene, W, thetas, rs, mode: str, fs: int, stft_kw: dict,
                 thresh_rel: float, steer_at=None):
    """Run one mode on one scene dict (see scenes.py for the schema).

    Returns per-source hit flags aligned with scene['positions'] order, plus
    false-positive count. For sequential modes the loudness order comes from
    the ground-truth ranks (oracle), since this study measures *information*,
    not a particular estimator of order.
    """
    from .render import stft_multi
    gts = scene["positions"]                      # [(theta, r)] per source
    ranks = scene["ranks"]                        # 0 = loudest
    order = np.argsort(ranks)

    if mode == "plain":
        Z, _ = stft_multi(scene["mixture"], fs, **stft_kw)
        preds = pick_peaks(srp_map(Z, W), thetas, rs, thresh_rel)
        return match(preds, gts)

    hits = np.zeros(len(gts), dtype=bool)
    n_fp = 0
    residual = scene["mixture"].copy()
    handled_steer: list[np.ndarray] = []
    for k in order:
        if mode == "oracle_sic":
            Z, _ = stft_multi(residual, fs, **stft_kw)
        elif mode == "oracle_null":
            Zfull, _ = stft_multi(scene["mixture"], fs, **stft_kw)
            Z = null_project(Zfull, handled_steer)
        else:
            raise ValueError(mode)
        preds = pick_peaks(srp_map(Z, W), thetas, rs, thresh_rel, max_peaks=1)
        h, fp = match(preds, [gts[k]])
        hits[k] = h[0]
        n_fp += fp if not h[0] else 0
        # account for source k before moving to the next
        if mode == "oracle_sic":
            residual = residual - scene["images"][k][: residual.shape[0]]
        else:
            handled_steer.append(steer_at(gts[k]))
    return hits, n_fp
