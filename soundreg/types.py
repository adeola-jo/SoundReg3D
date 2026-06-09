"""Shared containers passed between datasets, models, and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SceneGT:
    """Ground truth for one scene. Angles in degrees, ranges in meters."""

    classes: np.ndarray  # (N,) int
    r: np.ndarray  # (N,) float
    theta: np.ndarray  # (N,) float, degrees in [-180, 180)
    snr_db: np.ndarray | None = None  # (N,) per-source mixture SNR (eval only)
    nlos: np.ndarray | None = None  # (N,) bool
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.r)


@dataclass
class ScenePrediction:
    """One detector's output for one scene.

    ``scores`` is optional: threshold-based baselines must provide it (the eval
    sweeps the operating point); sequence models may provide per-object
    confidence (mean token probability) for diagnostics.
    """

    classes: np.ndarray  # (K,) int
    r: np.ndarray  # (K,) float
    theta: np.ndarray  # (K,) float degrees
    scores: np.ndarray | None = None  # (K,) higher = more confident
    extras: dict = field(default_factory=dict)  # e.g. eos_probs, tokens, logprob

    @property
    def n(self) -> int:
        return len(self.r)

    @staticmethod
    def empty() -> "ScenePrediction":
        z = np.zeros(0)
        return ScenePrediction(z.astype(int), z.copy(), z.copy(), z.copy())
