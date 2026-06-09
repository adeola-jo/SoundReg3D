"""FiLM conditioning on ego velocity (Acoustic-BEV's SpeedFiLM, simplified).

Not wired into the synthetic pipeline — the simulator has no ego motion,
so there is nothing for it to learn there. It exists now so that M1 real
data (where ego speed shifts Doppler and raises the ego-noise floor) can
condition any encoder with two lines:

    self.film = SpeedFiLM(n_channels)        # in __init__
    x = self.film(x, batch["velocity"])      # after an early conv block

The last layer is zero-initialized, so an untrained FiLM is exactly the
identity: adding it can never make a model worse at init.
"""

from __future__ import annotations

import torch
from torch import nn


class SpeedFiLM(nn.Module):
    def __init__(self, n_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * n_channels)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, H, W) feature map.
            speed: (B, 1) ego speed (m/s).

        Returns:
            x * (1 + gamma) + beta, with gamma/beta per (sample, channel).
        """
        gamma, beta = self.net(speed).chunk(2, dim=-1)
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
