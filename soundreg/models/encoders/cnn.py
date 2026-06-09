"""SpectroCNNEncoder: STFT feature image -> memory tokens.

FLOW
----
    features (B, C, T, F)
        -> strided ConvBlocks (k=3, s=2, GroupNorm, SiLU)
        -> 1x1 projection to d_model
        -> flatten (T', F') cells to a token sequence
        -> + factorized learned PE (time emb + frequency emb)
        -> memory (B, T' * F', d_model)

This is the simple v0 backbone: nothing acoustic-specific beyond the
input features. The interesting alternatives (polar SRP encoder today,
conformer-style later) swap in via `model.encoder.name` — the decoder
only ever sees (B, S, D).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ...registry import ENCODERS
from ..pos_enc import FactorizedPE2D


def _conv_out(n: int, n_layers: int) -> int:
    """Spatial size after n_layers of conv(k=3, s=2, p=1):
    n -> floor((n - 1) / 2) + 1 per layer. The PE tables must be sized
    at init, before any input exists, hence the closed form."""
    for _ in range(n_layers):
        n = (n - 1) // 2 + 1
    return n


class ConvBlock(nn.Module):
    """Conv(k=3) + GroupNorm + SiLU. GroupNorm because audio-feature
    batches are small and statistics per-sample beat per-batch here."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, c_out), c_out),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# Encoder
# =====================================================================
@ENCODERS.register("spectro_cnn")
class SpectroCNNEncoder(nn.Module):
    input_key = "features"

    def __init__(self, cfg: Dict, in_shape: Tuple[int, int, int]):
        """Args:
            cfg: The `model.encoder` config section: d_model, channels
                (one ConvBlock per entry, each stride 2).
            in_shape: (C, T, F) of the feature image; supplied by the
                factory from mic count and STFT geometry.
        """
        super().__init__()
        c_in, t_in, f_in = in_shape
        channels = list(cfg.get("channels", [64, 128, 256]))
        self.d_model = int(cfg["d_model"])

        blocks, prev = [], c_in
        for c in channels:
            blocks.append(ConvBlock(prev, c, stride=2))
            prev = c
        self.trunk = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(prev, self.d_model, 1)

        t_out = _conv_out(t_in, len(channels))
        f_out = _conv_out(f_in, len(channels))
        self.pos = FactorizedPE2D(t_out, f_out, self.d_model)
        self.out_shape = (t_out, f_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, F) -> (B, T' * F', d_model) memory tokens."""
        h = self.proj(self.trunk(x))  # (B, D, T', F')
        b, d, t, f = h.shape
        tokens = h.flatten(2).transpose(1, 2)  # (B, T'*F', D)
        return tokens + self.pos()[None]
