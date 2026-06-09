"""Scene <-> token-sequence conversion for the autoregressive detector.

PUBLIC SURFACE
--------------
    PAD, BOS, EOS
        Special token ids (0, 1, 2). PAD also doubles as the ignore
        index in the cross-entropy loss.

    TYPE_SPECIAL, TYPE_CLASS, TYPE_RANGE, TYPE_AZIMUTH
        Token-type ids consumed by the decoder's type embedding.

    SceneTokenizer
        encode / decode, the token-type schedule, the legal-next-token
        mask for type-masked decoding, and Gaussian soft targets.

VOCABULARY LAYOUT
-----------------
One shared embedding space, contiguous per-type blocks:

    [ PAD=0, BOS=1, EOS=2 | class x n_classes (optional)
                          | range x N_r | azimuth x N_theta ]

Each block has its own sub-vocabulary (the AutoReg3D design: separate
vocab per box parameter, not one shared quantizer), but all blocks live
in one embedding table so the decoder needs a single output head.

OBJECT AND SCENE LAYOUT
-----------------------
One object is the token triple

    (class, t_r, t_theta)

emitted in that order: class first (the AutoReg3D token-ordering ablation
found class-first works best — knowing *what* the object is helps predict
*where* it is). With a single class the class token is constant and drops
out entirely (brief, section 7), so an object is just (t_r, t_theta).

A scene in emission order pi is

    [BOS, (c, t_r, t_theta)_pi(1), ..., (c, t_r, t_theta)_pi(N), EOS]

EOS is the cardinality mechanism: no fixed query count, no score
threshold. EOS is only *legal* at an object boundary (where a new object
would start), never in the middle of a triple — `valid_next_mask`
enforces this during decoding.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from ..types import SceneGT, ScenePrediction
from .polar_grid import PolarGrid

# Special token ids. PAD is 0 so padded positions are also the CE
# ignore_index without any remapping.
PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3

# Token-type ids for the decoder's type embedding. SPECIAL covers
# PAD/BOS/EOS; the others tag which sub-vocabulary a position holds.
TYPE_SPECIAL, TYPE_CLASS, TYPE_RANGE, TYPE_AZIMUTH = 0, 1, 2, 3


# =====================================================================
# Tokenizer
# =====================================================================
class SceneTokenizer:
    """Bidirectional map between object lists and token sequences.

    The polar grid *is* the target vocabulary: one range token per range
    bin, one azimuth token per azimuth bin. Decoding returns bin centers,
    so quantization error is bounded by half a bin in each axis.
    """

    def __init__(
        self,
        grid: PolarGrid,
        # Number of object classes. With n_classes == 1 the class token
        # carries no information and is dropped from the object layout.
        n_classes: int = 1,
        # Force class tokens on/off; None = automatic (on iff n_classes > 1).
        include_class: Optional[bool] = None,
    ):
        self.grid = grid
        self.n_classes = n_classes
        self.include_class = n_classes > 1 if include_class is None else include_class

        # Lay out the contiguous vocabulary blocks after the specials.
        offset = N_SPECIAL
        self.class_offset = offset if self.include_class else None
        if self.include_class:
            offset += n_classes
        self.range_offset = offset
        offset += grid.n_r
        self.azimuth_offset = offset
        offset += grid.n_theta
        self.vocab_size = offset

        # Per-object slot schedule, in emission order within one object.
        self.slot_types = ([TYPE_CLASS] if self.include_class else []) + [
            TYPE_RANGE,
            TYPE_AZIMUTH,
        ]
        self.tokens_per_object = len(self.slot_types)

    # =================================================================
    # Encode (GT objects -> tokens)
    # =================================================================
    def encode(self, gt: SceneGT, order: np.ndarray) -> np.ndarray:
        """Tokenize a scene's ground truth in the given emission order.

        Args:
            gt: Scene ground truth (r in meters, theta in degrees).
            order: Permutation of range(gt.n) — the emission order chosen
                by the ordering policy (dominance, near-to-far, ...).

        Returns:
            int64 array [BOS, obj tokens ..., EOS] of length
            2 + gt.n * tokens_per_object.
        """
        r_bins = self.grid.r_to_bin(gt.r)
        t_bins = self.grid.theta_to_bin(gt.theta)
        seq = [BOS]
        for i in order:
            if self.include_class:
                seq.append(self.class_offset + int(gt.classes[i]))
            seq.append(self.range_offset + int(r_bins[i]))
            seq.append(self.azimuth_offset + int(t_bins[i]))
        seq.append(EOS)
        return np.asarray(seq, dtype=np.int64)

    def max_seq_len(self, max_objects: int) -> int:
        """Worst-case sequence length: BOS + max_objects triples + EOS."""
        return 2 + max_objects * self.tokens_per_object

    # =================================================================
    # Decode (tokens -> predicted objects)
    # =================================================================
    def decode(self, tokens) -> ScenePrediction:
        """Parse a generated sequence back into objects.

        Parsing is deliberately lenient:
            - stops at the first EOS or PAD,
            - drops an incomplete trailing group,
            - drops any group whose tokens fall outside their block.

        Type-masked decoding can never produce a malformed group, so the
        leniency only matters for hand-built or corrupted sequences —
        but a detector should not crash on its own output, ever.
        """
        toks = [int(t) for t in tokens]
        if toks and toks[0] == BOS:
            toks = toks[1:]
        classes: List[int] = []
        rs: List[float] = []
        thetas: List[float] = []
        group: List[int] = []
        for t in toks:
            if t in (EOS, PAD):
                break
            group.append(t)
            if len(group) == self.tokens_per_object:
                obj = self._parse_group(group)
                if obj is not None:
                    classes.append(obj[0])
                    rs.append(obj[1])
                    thetas.append(obj[2])
                group = []
        return ScenePrediction(
            classes=np.asarray(classes, dtype=np.int64),
            r=np.asarray(rs, dtype=np.float64),
            theta=np.asarray(thetas, dtype=np.float64),
        )

    def _parse_group(self, group: List[int]) -> Optional[Tuple[int, float, float]]:
        """One token group -> (class_id, r_m, theta_deg), or None if any
        token is outside its designated vocabulary block."""
        idx = 0
        cls = 0
        if self.include_class:
            cls = group[idx] - self.class_offset
            if not (0 <= cls < self.n_classes):
                return None
            idx += 1
        r_bin = group[idx] - self.range_offset
        th_bin = group[idx + 1] - self.azimuth_offset
        if not (0 <= r_bin < self.grid.n_r and 0 <= th_bin < self.grid.n_theta):
            return None
        return cls, float(self.grid.r_center(r_bin)), float(self.grid.theta_center(th_bin))

    # =================================================================
    # Token-type schedule and decoding mask
    # =================================================================
    def type_of_step(self, step: int) -> int:
        """Type of the token *emitted* at generation step `step`.

        Step 0 is the first token after BOS; steps cycle through
        slot_types, e.g. for the single-class layout:

            step:  0      1        2      3        4    ...
            type:  RANGE  AZIMUTH  RANGE  AZIMUTH  RANGE ...
        """
        return self.slot_types[step % self.tokens_per_object]

    def type_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token-type id for every position of an input batch.

        Position p > 0 of the input holds the token that was emitted at
        step p - 1, so its type follows the slot schedule. Position 0 is
        BOS; PAD/BOS/EOS keep TYPE_SPECIAL regardless of position.

        Args:
            tokens: (B, L) input ids, BOS at position 0.

        Returns:
            (B, L) int64 type ids for the decoder's type embedding.
        """
        b, length = tokens.shape
        steps = torch.arange(length, device=tokens.device) - 1
        slot = torch.where(steps >= 0, steps % self.tokens_per_object, 0)
        types = torch.tensor(self.slot_types, device=tokens.device)[slot]
        types = torch.where(steps >= 0, types, TYPE_SPECIAL)
        types = types.unsqueeze(0).expand(b, length).clone()
        types[tokens <= EOS] = TYPE_SPECIAL  # PAD / BOS / EOS
        return types

    def valid_next_mask(self, step: int, device=None) -> torch.Tensor:
        """Legal tokens at generation step `step` (token-type masking).

        Class steps may only emit class tokens, range steps range tokens,
        azimuth steps azimuth tokens. EOS is additionally legal at slot 0
        (an object boundary): the model may either start another object
        or stop. Mid-object EOS is illegal — a half-emitted object is
        not a thing.

        Returns:
            (vocab_size,) bool mask; True = token is legal.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        ttype = self.type_of_step(step)
        if step % self.tokens_per_object == 0:
            mask[EOS] = True
        if ttype == TYPE_CLASS:
            mask[self.class_offset : self.class_offset + self.n_classes] = True
        elif ttype == TYPE_RANGE:
            mask[self.range_offset : self.range_offset + self.grid.n_r] = True
        else:
            mask[self.azimuth_offset : self.azimuth_offset + self.grid.n_theta] = True
        return mask

    # =================================================================
    # Soft targets (the "one-bin Gaussian vs hard label" ablation)
    # =================================================================
    def soft_target_distribution(
        self, targets: torch.Tensor, sigma_bins: float
    ) -> torch.Tensor:
        """Expand hard next-token targets into per-position distributions.

        Location tokens get a Gaussian over *bin distance* inside their
        own vocabulary block:

            w(b) ~ exp( -0.5 * ((b - b_target) / sigma_bins)^2 )

        normalized to sum to 1 within the block. Azimuth uses circular
        bin distance, so bin 0 and bin N_theta - 1 are neighbours (the
        0/360 seam leaks mass both ways, as it should). Class and EOS
        targets stay one-hot. PAD rows come back all-zero; the loss must
        mask them out (SoundRegDetector does).

        Args:
            targets: (B, L) hard next-token ids.
            sigma_bins: Gaussian width in *bins* (not meters/degrees).

        Returns:
            (B, L, vocab_size) float32 target distributions.
        """
        b, length = targets.shape
        v = self.vocab_size
        out = torch.zeros(b, length, v, device=targets.device, dtype=torch.float32)
        flat = targets.reshape(-1)
        rows = out.reshape(-1, v)

        nonpad = flat != PAD
        # Targets that remain one-hot after the loop below (class, EOS, BOS).
        hard = nonpad.clone()

        for offset, size, circular in (
            (self.range_offset, self.grid.n_r, False),
            (self.azimuth_offset, self.grid.n_theta, True),
        ):
            sel = (flat >= offset) & (flat < offset + size)
            if not sel.any():
                continue
            hard &= ~sel
            centers = (flat[sel] - offset).float()  # (S,)
            bins = torch.arange(size, device=targets.device, dtype=torch.float32)
            diff = bins.unsqueeze(0) - centers.unsqueeze(1)  # (S, size)
            if circular:
                # Shortest signed bin distance on the circle.
                diff = torch.remainder(diff + size / 2, size) - size / 2
            w = torch.exp(-0.5 * (diff / max(sigma_bins, 1e-6)) ** 2)
            w = w / w.sum(dim=1, keepdim=True)
            rows[sel.nonzero(as_tuple=True)[0], offset : offset + size] = w

        idx = hard.nonzero(as_tuple=True)[0]
        rows[idx, flat[idx]] = 1.0
        return out
