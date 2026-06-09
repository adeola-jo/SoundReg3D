from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def wrap_deg(angle):
    """Wrap angle(s) to [-180, 180). Works on floats and numpy arrays."""
    return (angle + 180.0) % 360.0 - 180.0


def circular_diff_deg(a, b):
    """Signed circular difference a - b in degrees, in [-180, 180)."""
    return wrap_deg(a - b)
