"""Fast end-to-end checks: physics sanity (SRP finds a lone source, dominance
tracks energy), tokenizer round-trips, decoder masking, metrics, and a tiny
overfit run proving the training loop learns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soundreg.config import Cfg
from soundreg.data.dataset import SyntheticSceneDataset, collate
from soundreg.data.geometry import circular_array
from soundreg.data.polar_grid import PolarGrid, polar_distance
from soundreg.data.simulate import SceneSimulator
from soundreg.data.steering import ArrayManifold, srp_at_points, srp_polar_map
from soundreg.data.tokenizer import BOS, EOS, SceneTokenizer
from soundreg.eval.metrics import evaluate, match_scene, sweep_score_threshold
from soundreg.factory import build_all
from soundreg.types import SceneGT, ScenePrediction
from soundreg.utils import set_seed


# ----------------------------------------------------------------- fixtures
def small_cfg(**overrides) -> Cfg:
    cfg = Cfg.from_dict(
        {
            "exp_name": "test",
            "data": {
                "dataset": "synthetic",
                "features": ["logmag_ipd"],
                "ordering": "dominance",
                "srp_bands": 2,
                "seed": 0,
                "n_train": 16,
                "n_val": 8,
                "n_test": 8,
                "array": {"type": "circular", "n_mics": 6, "radius_m": 0.5, "z_m": 0.0},
                "audio": {"fs": 8000, "n_fft": 128, "src_z": 1.0},
                "sim": {
                    "n_frames": 12,
                    "n_sources_min": 1,
                    "n_sources_max": 3,
                    "r_min_m": 2.0,
                    "r_max_m": 18.0,
                    "min_separation_m": 2.0,
                    "dominance_spread_db": 20.0,
                    "scene_snr_db": [10.0, 20.0],
                    "bandwidth_hz": [200.0, 800.0],
                    "n_classes": 1,
                    "nlos_prob": 0.0,
                },
            },
            "grid": {"n_r": 12, "n_theta": 36, "r_max": 20.0},
            "model": {
                "name": "soundreg",
                "max_objects": 5,
                "encoder": {"name": "spectro_cnn", "d_model": 64, "channels": [32, 64]},
                "decoder": {
                    "d_model": 64,
                    "n_layers": 2,
                    "n_heads": 4,
                    "d_ff": 128,
                    "dropout": 0.0,
                    "context_mode": "full",
                },
                "loss": {"soft_sigma_bins": 0.0},
            },
            "training": {
                "seed": 0,
                "epochs": 1,
                "batch_size": 8,
                "lr": 1.0e-3,
                "num_workers": 0,
                "device": "cpu",
            },
            "eval": {
                "dist_thresh_m": 3.0,
                "decode": {"mode": "greedy"},
                "snr_edges": [-30, 0, 40],
            },
        }
    )
    for dotted, val in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return cfg


def make_manifold(cfg) -> ArrayManifold:
    mics = circular_array(cfg.data.array.n_mics, cfg.data.array.radius_m)
    freqs = np.fft.rfftfreq(cfg.data.audio.n_fft, 1.0 / cfg.data.audio.fs)
    return ArrayManifold(mics, freqs, src_z=1.0)


# ------------------------------------------------------------- grid & tokens
def test_grid_roundtrip_and_wrap():
    grid = PolarGrid(n_r=24, n_theta=72, r_max=30.0)
    r = np.array([0.1, 12.3, 29.9])
    th = np.array([-179.9, 0.0, 179.9])
    rb, tb = grid.r_to_bin(r), grid.theta_to_bin(th)
    assert np.all(np.abs(grid.r_center(rb) - r) <= grid.dr)
    assert np.all(np.abs(grid.theta_center(tb) - th) <= grid.dtheta)
    # wrap: 181 deg == -179 deg
    assert grid.theta_to_bin(181.0) == grid.theta_to_bin(-179.0)
    # polar distance: same point, wrapped angle
    assert polar_distance(10.0, 179.0, 10.0, -179.0) < 0.5


def test_tokenizer_roundtrip():
    grid = PolarGrid(n_r=12, n_theta=36, r_max=20.0)
    tok = SceneTokenizer(grid, n_classes=1)
    assert tok.tokens_per_object == 2  # class drops out when single-class
    gt = SceneGT(
        classes=np.zeros(3, dtype=int),
        r=np.array([3.0, 10.0, 17.0]),
        theta=np.array([-120.0, 5.0, 170.0]),
    )
    seq = tok.encode(gt, np.array([2, 0, 1]))
    assert seq[0] == BOS and seq[-1] == EOS and len(seq) == 2 + 3 * 2
    dec = tok.decode(seq)
    assert dec.n == 3
    # order preserved: first decoded object is gt index 2
    assert abs(dec.r[0] - 17.0) <= grid.dr
    assert abs(dec.theta[1] - (-120.0)) <= grid.dtheta


def test_tokenizer_multiclass_and_type_mask():
    grid = PolarGrid(n_r=12, n_theta=36, r_max=20.0)
    tok = SceneTokenizer(grid, n_classes=3)
    assert tok.tokens_per_object == 3
    m0 = tok.valid_next_mask(0)
    assert m0[EOS] and m0[tok.class_offset : tok.class_offset + 3].all()
    assert not m0[tok.range_offset : tok.range_offset + grid.n_r].any()
    m1 = tok.valid_next_mask(1)  # range step: no EOS mid-object
    assert not m1[EOS] and m1[tok.range_offset : tok.range_offset + grid.n_r].all()


def test_soft_targets_sum_to_one_and_wrap():
    grid = PolarGrid(n_r=12, n_theta=36, r_max=20.0)
    tok = SceneTokenizer(grid, n_classes=1)
    targets = torch.tensor([[tok.azimuth_offset, tok.range_offset + 5, EOS, 0]])
    soft = tok.soft_target_distribution(targets, sigma_bins=1.5)
    assert torch.allclose(soft[0, :3].sum(-1), torch.ones(3), atol=1e-5)
    assert soft[0, 3].sum() == 0  # PAD row empty
    # circular smoothing: azimuth bin 0 leaks mass to the last bin
    assert soft[0, 0, tok.azimuth_offset + grid.n_theta - 1] > 1e-3


# ------------------------------------------------------------ physics sanity
def test_srp_localizes_single_source_and_dominance_orders_energy():
    cfg = small_cfg()
    manifold = make_manifold(cfg)
    grid = PolarGrid(n_r=12, n_theta=36, r_max=20.0)
    sim_cfg = dict(cfg.data.sim.to_dict())
    sim_cfg["n_sources_min"] = sim_cfg["n_sources_max"] = 1
    sim = SceneSimulator(sim_cfg, manifold, sim_cfg["n_frames"])
    scene = sim.sample(np.random.default_rng(3))
    gt = scene["gt"]

    srp = srp_polar_map(scene["stft"], manifold, grid, n_bands=1, phat=True)[0]
    tb, rb = np.unravel_index(np.argmax(srp), srp.shape)
    err = polar_distance(
        grid.r_center(rb), grid.theta_center(tb), gt.r[0], gt.theta[0]
    )
    assert err < 4.0, f"SRP peak {err:.1f} m from the true source"

    # dominance score tracks received energy across many scenes
    sim_cfg["n_sources_min"], sim_cfg["n_sources_max"] = 3, 3
    sim = SceneSimulator(sim_cfg, manifold, sim_cfg["n_frames"])
    agree = 0
    for s in range(12):
        scene = sim.sample(np.random.default_rng(100 + s))
        dom = srp_at_points(
            scene["stft"], manifold, scene["gt"].r, scene["gt"].theta
        )
        agree += int(np.argmax(dom) == np.argmax(scene["energy"]))
    assert agree >= 8, f"dominance argmax matched energy argmax {agree}/12"


# ------------------------------------------------------------------- metrics
def test_matching_and_metrics():
    gt = SceneGT(
        classes=np.zeros(2, dtype=int),
        r=np.array([5.0, 10.0]),
        theta=np.array([0.0, 90.0]),
    )
    perfect = ScenePrediction(
        classes=np.zeros(2, dtype=int),
        r=np.array([5.2, 9.8]),
        theta=np.array([1.0, 89.0]),
    )
    m = match_scene(perfect, gt, dist_thresh=2.0)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    res = evaluate([perfect], [gt], 2.0)
    assert res["f1"] == pytest.approx(1.0)

    extra = ScenePrediction(
        classes=np.zeros(3, dtype=int),
        r=np.array([5.2, 9.8, 20.0]),
        theta=np.array([1.0, 89.0, -90.0]),
        scores=np.array([0.9, 0.8, 0.1]),
    )
    res = evaluate([extra], [gt], 2.0)
    assert res["tp"] == 2 and res["fp"] == 1
    thr, f1 = sweep_score_threshold([extra], [gt], 2.0)
    assert f1 == pytest.approx(1.0) and 0.1 < thr <= 0.8


# ----------------------------------------------------- model + training loop
def test_forward_generate_and_context_masking():
    set_seed(0)
    cfg = small_cfg()
    built = build_all(cfg)
    model, ds = built["model"], built["datasets"]["train"]
    batch = collate([ds[i] for i in range(4)])
    loss, logs = model.compute_loss(batch)
    assert torch.isfinite(loss)
    preds = model.predict(batch)
    assert len(preds) == 4
    for p in preds:
        assert p.n <= cfg.model.max_objects
        if p.n:
            assert np.all(p.r >= 0) and np.all(p.r <= cfg.grid.r_max)
            assert len(p.extras["eos_probs"]) >= 1

    # object_only context must change the logits beyond the first object
    cfg2 = small_cfg(**{"model.decoder.context_mode": "object_only"})
    built2 = build_all(cfg2)
    model2 = built2["model"]
    model2.load_state_dict(model.state_dict())
    model.eval(), model2.eval()
    tokens = batch["tokens"]
    with torch.no_grad():
        mem = model.encoder(batch["features"])
        full = model.decoder(tokens[:, :-1], mem)
        masked = model2.decoder(tokens[:, :-1], mem)
    assert torch.allclose(full[:, :2], masked[:, :2], atol=1e-5)  # BOS + 1st token
    assert not torch.allclose(full[:, 3:], masked[:, 3:], atol=1e-4)


def test_sample_and_beam_decode_run():
    set_seed(0)
    cfg = small_cfg()
    built = build_all(cfg)
    model, ds = built["model"], built["datasets"]["val"]
    batch = collate([ds[i] for i in range(2)])
    for kw in ({"mode": "sample", "top_p": 0.9}, {"mode": "beam", "beam_size": 3},
               {"mode": "greedy", "min_objects": 2}):
        preds = model.predict(batch, **kw)
        assert len(preds) == 2
        if kw.get("min_objects"):
            assert all(p.n >= 2 for p in preds)


def test_tiny_overfit_learns():
    """Teacher-forced loss must drop sharply when overfitting a fixed batch."""
    set_seed(0)
    cfg = small_cfg()
    built = build_all(cfg)
    model, ds = built["model"], built["datasets"]["train"]
    batch = collate([ds[i] for i in range(8)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    first = None
    for step in range(60):
        opt.zero_grad()
        loss, _ = model.compute_loss(batch)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < 0.5 * first, f"loss {first:.3f} -> {float(loss):.3f}"


def test_baselines_predict():
    set_seed(0)
    cfg = small_cfg(**{"data.features": ["srp"], "model.name": "srp_peaks"})
    built = build_all(cfg)
    ds = built["datasets"]["val"]
    batch = collate([ds[i] for i in range(2)])
    preds = built["model"].predict(batch)
    assert len(preds) == 2 and all(p.scores is not None for p in preds)

    cfg = small_cfg(**{"data.features": ["srp"], "model.name": "heatmap"})
    built = build_all(cfg)
    model, ds = built["model"], built["datasets"]["train"]
    batch = collate([ds[i] for i in range(2)])
    loss, _ = model.compute_loss(batch)
    assert torch.isfinite(loss)
    preds = model.predict(batch)
    assert len(preds) == 2
