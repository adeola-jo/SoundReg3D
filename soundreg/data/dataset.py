"""Datasets and batching.

PUBLIC SURFACE
--------------
    SyntheticSceneDataset
        On-the-fly simulated scenes with deterministic per-(split, index)
        seeds: a reproducible dataset with zero storage.

    RealArrayDataset
        M1 skeleton for the recorded array+LiDAR data (brief, section 7).
        Untested until the real data is wired in.

    build_sample
        The shared sample-assembly path both datasets funnel through:
        dominance -> ordering -> tokens -> encoder features. One code
        path means synthetic and real samples can never drift apart in
        format.

    collate
        Pads token sequences, stacks feature tensors, keeps per-scene GT
        as Python lists (N varies per scene — that is the whole point).

SAMPLE CONTRACT
---------------
Every sample is a dict:

    tokens     LongTensor (L,)        [BOS, obj tokens ..., EOS]
    gt         SceneGT                positions, classes, SNR, NLOS flags
    dominance  (N,) float             SRP at GT positions (diagnostics)
    features   FloatTensor (C, T, F)  if "logmag_ipd" in data.features
    srp        FloatTensor (B_b, N_theta, N_r)  if "srp" in data.features

After collate the batch adds a leading B dim to the tensors, tokens are
right-padded with PAD to the batch max length, and gt / dominance become
lists of length B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..types import SceneGT
from .features import logmag_ipd, normalize_srp, stacked_to_complex
from .ordering import order_objects
from .polar_grid import PolarGrid, xy_to_rtheta
from .simulate import SceneSimulator
from .steering import ArrayManifold, srp_at_points, srp_polar_map
from .tokenizer import SceneTokenizer

# Split seed offsets: large odd primes so train/val/test index ranges can
# never collide even for very large synthetic datasets.
_SPLIT_SEEDS = {"train": 0, "val": 1_000_003, "test": 2_000_003}


# =====================================================================
# Synthetic dataset
# =====================================================================
class SyntheticSceneDataset(Dataset):
    """Simulated scenes, generated lazily in __getitem__.

    Index i of split s is always the same scene (rng seeded with
    base_seed + i), so "the synthetic val set" is a stable object across
    runs and machines without a single file on disk. The dataset *size*
    is just the configured count (`data.n_train` etc.).
    """

    def __init__(
        self,
        data_cfg: Dict,
        grid: PolarGrid,
        tokenizer: SceneTokenizer,
        manifold: ArrayManifold,
        split: str,
    ):
        self.cfg = data_cfg
        self.grid = grid
        self.tokenizer = tokenizer
        self.manifold = manifold
        self.split = split
        self.length = int(data_cfg["n_" + split])
        # data.seed shifts ALL splits together: a different seed is a
        # genuinely different dataset draw (used for seed-variance runs).
        self.base_seed = int(data_cfg.get("seed", 0)) * 10_000_019 + _SPLIT_SEEDS[split]
        self.features = list(data_cfg["features"])
        self.ordering = data_cfg["ordering"]
        self.srp_bands = int(data_cfg.get("srp_bands", 4))
        self.sim = SceneSimulator(
            data_cfg["sim"], manifold, int(data_cfg["sim"]["n_frames"])
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        rng = np.random.default_rng(self.base_seed + idx)
        scene = self.sim.sample(rng)
        return build_sample(
            stft=scene["stft"],
            gt=scene["gt"],
            tokenizer=self.tokenizer,
            grid=self.grid,
            manifold=self.manifold,
            features=self.features,
            ordering=self.ordering,
            srp_bands=self.srp_bands,
            rng=rng,
            ctx_extra={"energy": scene["energy"]},  # oracle orderings
        )


# =====================================================================
# Real dataset (M1 skeleton)
# =====================================================================
class RealArrayDataset(Dataset):
    """Recorded array+LiDAR data (brief, section 7). M1 skeleton.

    Expected layout: `root/<split>/*.npz`, one scene window per file,
    with at least:

        stft         (2M, F, T) float   real parts then imag parts
        ann_centers  (N, 3) float       sounding objects only, ego frame

    Optional: ann_classes (N,), velocity (6,), timestamp, scene_name.

    The annotated boxes are the sounding objects; silent objects are
    simply not annotated, so there is nothing to filter here — the label
    set IS the detection target set.
    """

    def __init__(self, data_cfg, grid, tokenizer, manifold, split):
        self.cfg = data_cfg
        self.grid = grid
        self.tokenizer = tokenizer
        self.manifold = manifold
        self.files = sorted(Path(data_cfg["root"], split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files under {data_cfg['root']}/{split}"
            )
        self.features = list(data_cfg["features"])
        self.ordering = data_cfg["ordering"]
        self.srp_bands = int(data_cfg.get("srp_bands", 4))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        item = np.load(self.files[idx], allow_pickle=True)
        stft = stacked_to_complex(item["stft"])
        centers = item["ann_centers"].reshape(-1, 3)
        r, theta = xy_to_rtheta(centers[:, 0], centers[:, 1])
        classes = (
            item["ann_classes"].astype(np.int64)
            if "ann_classes" in item
            else np.zeros(len(r), dtype=np.int64)
        )
        gt = SceneGT(classes=classes, r=r, theta=theta)
        # The rng only feeds the 'random' ordering; idx-seeded keeps even
        # that reproducible.
        rng = np.random.default_rng(idx)
        return build_sample(
            stft, gt, self.tokenizer, self.grid, self.manifold,
            self.features, self.ordering, self.srp_bands, rng, ctx_extra={},
        )


# =====================================================================
# Shared assembly + collate
# =====================================================================
def build_sample(
    stft, gt, tokenizer, grid, manifold, features, ordering, srp_bands, rng, ctx_extra
) -> Dict:
    """One scene -> one training sample.

    Steps (mirrors the brief's preparation list, section 7):
        1. dominance: SRP at every GT position (raw power, no PHAT),
        2. ordering:  permutation from the configured policy,
        3. tokens:    [BOS, ordered object tokens, EOS],
        4. features:  every stack named in `data.features`.
    """
    if gt.n > 0:
        dom = srp_at_points(stft, manifold, gt.r, gt.theta, phat=False)
    else:
        dom = np.zeros(0)
    ctx = {"dominance": dom, **ctx_extra}
    order = order_objects(ordering, gt, ctx, rng) if gt.n > 0 else np.zeros(0, int)
    tokens = tokenizer.encode(gt, order)

    sample = {
        "tokens": torch.from_numpy(tokens),
        "gt": gt,
        "dominance": dom,
    }
    if "logmag_ipd" in features:
        sample["features"] = torch.from_numpy(logmag_ipd(stft))
    if "srp" in features:
        srp = srp_polar_map(stft, manifold, grid, n_bands=srp_bands, phat=True)
        sample["srp"] = torch.from_numpy(normalize_srp(srp))
    return sample


def collate(batch: List[Dict]) -> Dict:
    """Pad tokens to the batch max length; stack tensors; pass GT through
    as lists (scene cardinality is variable by design)."""
    from .tokenizer import PAD

    max_len = max(s["tokens"].shape[0] for s in batch)
    tokens = torch.full((len(batch), max_len), PAD, dtype=torch.long)
    for i, s in enumerate(batch):
        tokens[i, : s["tokens"].shape[0]] = s["tokens"]
    out = {
        "tokens": tokens,
        "gt": [s["gt"] for s in batch],
        "dominance": [s["dominance"] for s in batch],
    }
    for key in ("features", "srp"):
        if key in batch[0]:
            out[key] = torch.stack([s[key] for s in batch])
    return out
