"""Parallel baseline: learned polar heatmap + peak extraction.

Baseline 3 of the brief's experiment section, and the closest learned
parallel counterpart to SoundReg's polar variant: SAME SRP input, SAME
grid — only the output side differs (independent per-cell scores +
peak picking vs. sequential conditioned token emission). When SoundReg
beats this model on low-SNR recall, the difference is attributable to
the formulation, not the features.

FLOW
----
    srp (B, n_bands, N_theta, N_r)
        -> CircularConv trunk, stride 1 (keep full grid resolution)
        -> 1x1 conv -> logits (B, N_theta, N_r)

    train:    penalty-reduced focal loss vs. Gaussian splat targets
              (the CenterPoint/CenterNet recipe, 2D-polar edition)
    predict:  sigmoid -> local peaks -> candidates with scores
              (operating threshold swept on val by the eval harness)

Single-class for now; a per-class map head is the M3 extension if
classes become relevant.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from ...data.polar_grid import PolarGrid
from ...registry import MODELS
from ...types import ScenePrediction
from ..base import DetectorBase
from ..encoders.polar import CircularConv
from .peaks import gaussian_splat_targets, local_peaks


@MODELS.register("heatmap")
class HeatmapDetector(DetectorBase):
    input_key = "srp"

    def __init__(self, model_cfg: Dict, grid: PolarGrid, in_bands: int):
        """Args:
            model_cfg: The `model` config section: channels, max_peaks,
                target_sigma_bins, focal_alpha, focal_beta.
            grid: Output grid (and splat-target geometry).
            in_bands: Number of SRP input bands (from data.srp_bands).
        """
        super().__init__()
        self.grid = grid
        self.max_peaks = int(model_cfg.get("max_peaks", 12))
        # Splat width in bins; alpha/beta are the standard CenterNet
        # focal exponents (alpha sharpens hard examples, beta down-
        # weights negatives near a splat).
        self.sigma_bins = float(model_cfg.get("target_sigma_bins", 1.0))
        self.alpha = float(model_cfg.get("focal_alpha", 2.0))
        self.beta = float(model_cfg.get("focal_beta", 4.0))

        channels = list(model_cfg.get("channels", [32, 64, 64]))
        blocks, prev = [], in_bands
        for c in channels:
            blocks.append(CircularConv(prev, c, 1, 1))  # stride 1: full res
            prev = c
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Conv2d(prev, 1, 1)
        # Bias init so sigmoid(logit) ~ 0.1 at start: the standard focal
        # trick that stops the all-background gradient from dominating
        # the first epochs.
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, srp: torch.Tensor) -> torch.Tensor:
        """(B, n_bands, N_theta, N_r) -> (B, N_theta, N_r) logits."""
        return self.head(self.trunk(srp))[:, 0]

    # =================================================================
    # Training
    # =================================================================
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Penalty-reduced focal loss (CenterNet Eq. 1):

            cell is a positive (target == 1):
                - (1 - p)^alpha * log(p)
            cell is background:
                - (1 - target)^beta * p^alpha * log(1 - p)

        normalized by the number of positives.
        """
        logits = self.forward(batch["srp"])
        device = logits.device
        targets = torch.stack(
            [
                gaussian_splat_targets(
                    self.grid.n_theta,
                    self.grid.n_r,
                    self.grid.theta_to_bin(gt.theta),
                    self.grid.r_to_bin(gt.r),
                    self.sigma_bins,
                )
                for gt in batch["gt"]
            ]
        ).to(device)

        p = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
        pos = targets >= 1.0 - 1e-6
        pos_loss = -((1 - p) ** self.alpha) * torch.log(p)
        neg_loss = -((1 - targets) ** self.beta) * (p**self.alpha) * torch.log(1 - p)
        n_pos = pos.float().sum().clamp(min=1)
        loss = (pos_loss[pos].sum() + neg_loss[~pos].sum()) / n_pos
        return loss, {"loss": float(loss)}

    # =================================================================
    # Inference
    # =================================================================
    @torch.no_grad()
    def predict(self, batch: Dict, **kw) -> List[ScenePrediction]:
        scores = torch.sigmoid(self.forward(batch["srp"]))
        preds = []
        for b in range(scores.shape[0]):
            peaks = local_peaks(scores[b].cpu(), self.max_peaks)
            if not peaks:
                preds.append(ScenePrediction.empty())
                continue
            tb, rb, sc = (np.asarray(v) for v in zip(*peaks))
            preds.append(
                ScenePrediction(
                    classes=np.zeros(len(sc), dtype=np.int64),
                    r=self.grid.r_center(rb),
                    theta=self.grid.theta_center(tb),
                    scores=sc.astype(np.float64),
                )
            )
        return preds
