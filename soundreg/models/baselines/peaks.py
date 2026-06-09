"""Shared peak utilities for the map-based baselines.

Both SRPPeakDetector and HeatmapDetector are "parallel" detectors in the
brief's sense: every grid cell scores independently and objects are read
off as local maxima. These helpers keep the two baselines numerically
identical in everything except where their score map comes from.

Axis convention: theta is the FIRST map axis and wraps; r is the second
and does not.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def local_peaks(
    score_map: torch.Tensor, max_peaks: int
) -> List[Tuple[int, int, float]]:
    """3x3 local maxima of a (N_theta, N_r) score map, circular in theta.

    A cell is a peak iff it equals the max over its 3x3 neighbourhood,
    where the neighbourhood wraps across the theta seam (a peak at bin 0
    competes with bin N_theta - 1) and is zero-padded in r.

    Args:
        score_map: (N_theta, N_r) scores, any scale.
        max_peaks: Keep at most this many, by descending score. This is a
            capacity cap, not an operating point — the score threshold is
            applied later by the eval harness.

    Returns:
        [(theta_bin, r_bin, score), ...] sorted by descending score.
    """
    x = score_map[None, None]  # (1, 1, N_theta, N_r)
    xp = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)  # wrap theta
    pooled = F.max_pool2d(xp, kernel_size=3, stride=1, padding=(0, 1))
    is_peak = x == pooled
    idx = is_peak[0, 0].nonzero(as_tuple=False)
    if idx.numel() == 0:
        return []
    scores = score_map[idx[:, 0], idx[:, 1]]
    order = torch.argsort(scores, descending=True)[:max_peaks]
    return [(int(idx[i, 0]), int(idx[i, 1]), float(scores[i])) for i in order]


def gaussian_splat_targets(
    n_theta: int,
    n_r: int,
    theta_bins: np.ndarray,
    r_bins: np.ndarray,
    sigma_bins: float = 1.0,
) -> torch.Tensor:
    """CenterNet-style heatmap target: max of Gaussians at the GT bins.

        target(t, r) = max over objects of
            exp( -(d_theta^2 + d_r^2) / (2 sigma^2) )

    with d_theta the CIRCULAR bin distance (the splat of a source near
    -180 deg leaks across the seam, matching what the circular convs in
    the heatmap trunk can see). Max (not sum) so overlapping splats keep
    a peak value of exactly 1 at each GT cell — the focal loss treats
    "== 1" cells as positives.

    Returns:
        (N_theta, N_r) float32 in [0, 1].
    """
    tt = np.arange(n_theta)[:, None]
    rr = np.arange(n_r)[None, :]
    target = np.zeros((n_theta, n_r), dtype=np.float32)
    for tb, rb in zip(theta_bins, r_bins):
        dt = (tt - tb + n_theta / 2) % n_theta - n_theta / 2
        dr = rr - rb
        g = np.exp(-(dt**2 + dr**2) / (2.0 * sigma_bins**2))
        target = np.maximum(target, g.astype(np.float32))
    return torch.from_numpy(target)
