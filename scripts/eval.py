"""Evaluate a trained checkpoint with arbitrary decoding settings.

Usage:
    python scripts/eval.py --run runs/soundreg_synth/seed0 \
        [--mode beam|sample|greedy] [--min-objects 2] [--split test] [--set k=v]

Reads config.yaml + best.pt from the run dir; writes metrics_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from soundreg.config import load_config
from soundreg.eval.metrics import stratified_recall
from soundreg.factory import build_all
from soundreg.training.trainer import evaluate_model, make_loader, scalar_metrics
from soundreg.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", default=None, help="greedy | sample | beam")
    ap.add_argument("--min-objects", type=int, default=0)
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml", args.set)
    set_seed(int(cfg.training.seed))
    device = get_device(cfg.training.get("device", "auto"))

    built = build_all(cfg)
    model = built["model"].to(device)
    score_threshold = None
    if model.requires_training:
        ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        score_threshold = ckpt.get("score_threshold")

    decode = dict(cfg.eval.get("decode", {}))
    if args.mode:
        decode["mode"] = args.mode
    decode["min_objects"] = args.min_objects
    decode["beam_size"] = args.beam_size

    result = evaluate_model(
        model,
        make_loader(built["datasets"][args.split], int(cfg.training.batch_size), False),
        device,
        float(cfg.eval.dist_thresh_m),
        decode,
        score_threshold=score_threshold,
    )
    strat = stratified_recall(result, list(cfg.eval.snr_edges))
    tag = f"{args.split}_{decode.get('mode', 'greedy')}_min{args.min_objects}"
    out = {"metrics": scalar_metrics(result), "stratified": strat, "decode": decode}
    with open(run_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["metrics"], indent=2))
    print("stratified recall by SNR:", json.dumps(strat["by_snr"], indent=2))


if __name__ == "__main__":
    main()
