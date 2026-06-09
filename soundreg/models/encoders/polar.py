"""PolarSRPEncoder: SRP maps on the polar grid -> memory tokens.

The brief's canonical encoder. Three properties distinguish it from the
plain spectrogram CNN:

    1. Features already live on the TARGET grid: an activation at cell
       (theta_i, r_j) is evidence for the token pair the decoder will
       emit for a source there. No learned coordinate transform needed.
    2. Convolutions are circular along the azimuth axis — theta wraps,
       and a source at 179 deg is adjacent to one at -179 deg.
    3. Tokens carry the polar positional encoding of the brief:
       separate azimuth and range embeddings, summed.

FLOW
----
    srp (B, n_bands, N_theta, N_r)
        -> CircularConv blocks (circular pad in theta, zero pad in r)
        -> 1x1 projection to d_model
        -> flatten (N_theta', N_r') cells
        -> + polar PE (theta emb + r emb)
        -> memory (B, N_theta' * N_r', d_model)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ...registry import ENCODERS
from ..pos_enc import FactorizedPE2D


class CircularConv(nn.Module):
    """3x3 conv, circular padding on the theta (H) axis, zero on r (W).

    Padding is done manually (F.pad) because Conv2d's padding_mode is
    all-or-nothing per layer and we need circular on exactly one axis.
    """

    def __init__(self, c_in: int, c_out: int, stride_theta: int, stride_r: int):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, 3, stride=(stride_theta, stride_r), padding=0
        )
        self.norm = nn.GroupNorm(min(8, c_out), c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 1, 1), mode="circular")  # theta wraps
        x = F.pad(x, (1, 1, 0, 0))  # r does not
        return F.silu(self.norm(self.conv(x)))


def _strided_out(n: int, stride: int) -> int:
    """Output size of the padded 3x3 conv: ceil(n / stride)."""
    return (n - 1) // stride + 1


# =====================================================================
# Encoder
# =====================================================================
@ENCODERS.register("polar_srp")
class PolarSRPEncoder(nn.Module):
    input_key = "srp"

    def __init__(self, cfg: Dict, in_shape: Tuple[int, int, int]):
        """Args:
            cfg: The `model.encoder` config section: d_model, channels,
                strides (per block, applied to both axes; default [2, 2]
                turns the 72 x 24 grid into 18 x 6 = 108 tokens).
            in_shape: (n_bands, N_theta, N_r) from the factory.
        """
        super().__init__()
        bands, n_theta, n_r = in_shape
        channels = list(cfg.get("channels", [64, 128]))
        strides = list(cfg.get("strides", [2, 2]))
        self.d_model = int(cfg["d_model"])

        blocks, prev = [], bands
        t_out, r_out = n_theta, n_r
        for c, s in zip(channels, strides):
            blocks.append(CircularConv(prev, c, s, s))
            prev = c
            t_out = _strided_out(t_out, s)
            r_out = _strided_out(r_out, s)
        self.trunk = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(prev, self.d_model, 1)
        self.pos = FactorizedPE2D(t_out, r_out, self.d_model)
        self.out_shape = (t_out, r_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_bands, N_theta, N_r) -> (B, N_theta' * N_r', d_model)."""
        h = self.proj(self.trunk(x))
        tokens = h.flatten(2).transpose(1, 2)
        return tokens + self.pos()[None]
