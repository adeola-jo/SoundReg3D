"""Wires config -> grid -> tokenizer -> array manifold -> datasets -> model.

This is the only place that knows how the pieces fit together; everything else
depends on interfaces. Add a new encoder/model/ordering by registering it and
naming it in a config.
"""

from __future__ import annotations

import numpy as np

from . import models  # noqa: F401  (registration side effects)
from .data.dataset import RealArrayDataset, SyntheticSceneDataset
from .data.features import logmag_ipd_channels
from .data.geometry import circular_array, load_mic_xml
from .data.polar_grid import PolarGrid
from .data.steering import ArrayManifold
from .data.tokenizer import SceneTokenizer
from .models.base import DetectorBase
from .registry import ENCODERS, MODELS

_INPUT_KEY_TO_FEATURE = {"features": "logmag_ipd", "srp": "srp"}


def build_grid(cfg) -> PolarGrid:
    g = cfg.grid
    return PolarGrid(n_r=int(g.n_r), n_theta=int(g.n_theta), r_max=float(g.r_max))


def build_manifold(cfg) -> ArrayManifold:
    arr = cfg.data.array
    if arr.type == "circular":
        mic_pos = circular_array(int(arr.n_mics), float(arr.radius_m), float(arr.z_m))
    elif arr.type == "xml":
        mic_pos = load_mic_xml(arr.path)
    else:
        raise ValueError(f"Unknown array type '{arr.type}'")
    audio = cfg.data.audio
    freqs = np.fft.rfftfreq(int(audio.n_fft), 1.0 / float(audio.fs))
    return ArrayManifold(mic_pos, freqs, src_z=float(audio.get("src_z", 1.0)))


def build_tokenizer(cfg, grid: PolarGrid) -> SceneTokenizer:
    n_classes = int(cfg.data.sim.get("n_classes", 1)) if "sim" in cfg.data else int(
        cfg.data.get("n_classes", 1)
    )
    return SceneTokenizer(grid, n_classes=n_classes)


def build_dataset(cfg, grid, tokenizer, manifold, split: str):
    name = cfg.data.get("dataset", "synthetic")
    if name == "synthetic":
        return SyntheticSceneDataset(cfg.data, grid, tokenizer, manifold, split)
    if name == "real":
        return RealArrayDataset(cfg.data, grid, tokenizer, manifold, split)
    raise ValueError(f"Unknown dataset '{name}'")


def build_model(cfg, grid, tokenizer, manifold) -> DetectorBase:
    mc = cfg.model
    name = mc.name
    if name == "soundreg":
        enc_cfg = mc.encoder
        enc_cls = ENCODERS.get(enc_cfg.name)
        if enc_cls.input_key == "features":
            in_shape = (
                logmag_ipd_channels(manifold.n_mics),
                int(cfg.data.sim.n_frames),
                manifold.n_freqs,
            )
        else:  # "srp"
            in_shape = (int(cfg.data.get("srp_bands", 4)), grid.n_theta, grid.n_r)
        encoder = enc_cls(enc_cfg, in_shape)
        model = MODELS.get("soundreg")(mc, encoder, tokenizer)
    elif name == "heatmap":
        model = MODELS.get("heatmap")(mc, grid, in_bands=int(cfg.data.get("srp_bands", 4)))
    elif name == "srp_peaks":
        model = MODELS.get("srp_peaks")(mc, grid)
    else:
        raise ValueError(f"Unknown model '{name}'. Available: {MODELS.names()}")

    needed = _INPUT_KEY_TO_FEATURE[model.input_key]
    if needed not in list(cfg.data.features):
        raise ValueError(
            f"Model '{name}' consumes '{needed}' but data.features={list(cfg.data.features)}"
        )
    return model


def build_all(cfg) -> dict:
    grid = build_grid(cfg)
    tokenizer = build_tokenizer(cfg, grid)
    manifold = build_manifold(cfg)
    model = build_model(cfg, grid, tokenizer, manifold)
    return {
        "grid": grid,
        "tokenizer": tokenizer,
        "manifold": manifold,
        "model": model,
        "datasets": {
            split: build_dataset(cfg, grid, tokenizer, manifold, split)
            for split in ("train", "val", "test")
        },
    }
