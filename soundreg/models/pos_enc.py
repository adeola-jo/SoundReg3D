"""Learned factorized 2D positional encoding.

The AutoReg3D scheme: one learned embedding per axis, summed —

    PE(a, b) = emb_a[a] + emb_b[b]

which keeps the parameter count linear (n_a + n_b vectors instead of
n_a * n_b) while still giving every grid cell a unique code. Used for
both axis pairs in this project:

    spectro encoder   (time, frequency)
    polar encoder     (azimuth, range)   <- the brief's "polar positional
                                            encoding with separate range
                                            and azimuth embeddings"
"""

from __future__ import annotations

import torch
from torch import nn


class FactorizedPE2D(nn.Module):
    def __init__(self, n_a: int, n_b: int, d_model: int):
        """Args:
            n_a: Size of the first (row-major outer) axis.
            n_b: Size of the second axis.
            d_model: Embedding width; must match the token width it is
                added to.
        """
        super().__init__()
        self.n_a, self.n_b = n_a, n_b
        self.emb_a = nn.Embedding(n_a, d_model)
        self.emb_b = nn.Embedding(n_b, d_model)
        nn.init.normal_(self.emb_a.weight, std=0.02)
        nn.init.normal_(self.emb_b.weight, std=0.02)

    def forward(self) -> torch.Tensor:
        """The full PE table, (n_a * n_b, d_model), row-major over (a, b)
        — the same order `flatten(2)` produces on a (B, D, n_a, n_b) map."""
        a = self.emb_a.weight[:, None, :]  # (n_a, 1, d)
        b = self.emb_b.weight[None, :, :]  # (1, n_b, d)
        return (a + b).reshape(self.n_a * self.n_b, -1)
