"""Train (or, for training-free baselines, just evaluate) a detector.

Usage:
    python scripts/train.py --config experiments/configs/soundreg_synth.yaml
    python scripts/train.py --config ... --seed 1 --set data.ordering=random

Run artifacts land in runs/<exp_name>/seed<k>/:
    config.yaml  log.jsonl  best.pt  last.pt  metrics.json  stratified.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soundreg.config import load_config, save_config
from soundreg.eval.metrics import stratified_recall
from soundreg.factory import build_all
from soundreg.training.trainer import Trainer, evaluate_model, make_loader, scalar_metrics
from soundreg.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg.training.seed = args.seed
    seed = int(cfg.training.seed)
    set_seed(seed)

    run_dir = Path("runs") / cfg.exp_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    built = build_all(cfg)
    model, datasets = built["model"], built["datasets"]
    device = get_device(cfg.training.get("device", "auto"))

    score_threshold = None
    if model.requires_training:
        trainer = Trainer(cfg, model, datasets, run_dir)
        trainer.train()
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        score_threshold = ckpt.get("score_threshold")
    else:
        model.to(device)
        # operating point for training-free baselines: swept on val
        val = evaluate_model(
            model,
            make_loader(datasets["val"], int(cfg.training.batch_size), False),
            device,
            float(cfg.eval.dist_thresh_m),
            sweep_threshold=True,
        )
        score_threshold = val.get("score_threshold")

    test = evaluate_model(
        model,
        make_loader(datasets["test"], int(cfg.training.batch_size), False),
        device,
        float(cfg.eval.dist_thresh_m),
        dict(cfg.eval.get("decode", {})),
        score_threshold=score_threshold,
    )
    strat = stratified_recall(test, list(cfg.eval.snr_edges))

    metrics = {"test": scalar_metrics(test), "stratified": strat, "seed": seed}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(
        run_dir / "stratified.npz",
        gt_matched=test["gt_matched"],
        gt_snr_db=test["gt_snr_db"],
        gt_n_sources=test["gt_n_sources"],
        gt_nlos=test["gt_nlos"],
        eos_records=test["eos"].get("records", np.zeros((0, 2))),
    )
    print(json.dumps(metrics["test"], indent=2))
    print("stratified recall by SNR:", json.dumps(strat["by_snr"], indent=2))


if __name__ == "__main__":
    main()
