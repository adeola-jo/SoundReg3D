"""The one interface every detector implements.

SoundReg and every baseline are interchangeable behind two methods, so
the SAME trainer trains them and the SAME metrics judge them — baseline
fairness is structural, not a convention someone has to remember.

DATA CONTRACT
-------------
    compute_loss(batch) -> (scalar loss, dict of float logs)
        Trainable detectors only. The logs dict is averaged over the
        epoch and written to log.jsonl with a train_ prefix.

    predict(batch, **decode_kwargs) -> list[ScenePrediction]
        One prediction per scene, always. Score-thresholded baselines
        return ALL candidates with scores attached; the eval harness
        owns the operating point (swept on val, frozen for test).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from ..types import ScenePrediction


class DetectorBase(nn.Module):
    #: Training-free baselines (e.g. SRP peak picking) set this False;
    #: the train script then skips the Trainer and goes straight to eval.
    requires_training: bool = True

    #: Which batch key the model consumes: "features" (logmag+IPD image)
    #: or "srp" (polar SRP maps). The factory validates that the dataset
    #: actually computes it.
    input_key: str = "features"

    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, batch: Dict, **decode_kwargs) -> List[ScenePrediction]:
        raise NotImplementedError
