"""Training-free baseline: SRP-PHAT map + peak picking.

Baseline 1 of the brief's experiment section — the classical signal-
processing floor every learned model must clear. There are no learnable
parameters: the score map is the band-summed SRP-PHAT input feature
itself, and detections are its local maxima.

It emits up to `max_peaks` candidates WITH scores; the eval harness owns
the operating point (threshold swept on val, frozen for test — the
fairness rule of PLAN.md). Single-class only: a steered-power map says
where energy comes from, not what kind of object emitted it.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ...data.polar_grid import PolarGrid
from ...registry import MODELS
from ...types import ScenePrediction
from ..base import DetectorBase
from .peaks import local_peaks


@MODELS.register("srp_peaks")
class SRPPeakDetector(DetectorBase):
    requires_training = False
    input_key = "srp"

    def __init__(self, model_cfg: Dict, grid: PolarGrid):
        """Args:
            model_cfg: The `model` config section; only `max_peaks` (the
                candidate capacity, NOT the operating point) is read.
            grid: Maps peak bins back to (r, theta) bin centers.
        """
        super().__init__()
        self.grid = grid
        self.max_peaks = int(model_cfg.get("max_peaks", 12))

    def compute_loss(self, batch):
        raise RuntimeError(
            "SRPPeakDetector is training-free; the train script should "
            "have skipped the Trainer (requires_training = False)."
        )

    def predict(self, batch: Dict, **kw) -> List[ScenePrediction]:
        srp = batch["srp"]  # (B, bands, N_theta, N_r), standardized per band
        # Bands are z-scored individually, so a plain sum weights them
        # equally regardless of absolute energy.
        score_maps = srp.sum(dim=1)
        preds = []
        for b in range(score_maps.shape[0]):
            peaks = local_peaks(score_maps[b].cpu(), self.max_peaks)
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
