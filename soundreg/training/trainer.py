"""Minimal training loop shared by every trainable detector.

Deliberately small and framework-free — research code you can read top
to bottom. What it does provide:

    - AdamW + linear warmup -> cosine decay (per step),
    - optional AMP and gradient clipping,
    - per-epoch validation through the SAME evaluate_model used for the
      final test numbers (no train-time/eval-time metric drift),
    - JSONL logging (one record per epoch, machine-readable),
    - best-F1 + last checkpointing; the best checkpoint also stores the
      val-swept score threshold so test never re-tunes it.

PUBLIC SURFACE
--------------
    Trainer            the loop.
    evaluate_model     predict over a loader -> metrics dict.
    make_loader        DataLoader with the project collate.
    batch_to_device    tensors to device, GT lists untouched.
    scalar_metrics     the JSON-safe subset of an eval result.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data.dataset import collate
from ..eval.metrics import (
    apply_score_threshold,
    eos_calibration,
    evaluate,
    sweep_score_threshold,
)
from ..models.base import DetectorBase
from ..utils import count_params, get_device


# =====================================================================
# Loaders and helpers
# =====================================================================
def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0):
    """DataLoader with the project collate. num_workers defaults to 0:
    on Windows, worker spawn + the on-the-fly simulator is usually slower
    than just generating in-process; raise it on Linux boxes."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def batch_to_device(batch: Dict, device) -> Dict:
    """Move tensors; leave GT/dominance lists where they are (numpy)."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# =====================================================================
# Evaluation entry point
# =====================================================================
@torch.no_grad()
def evaluate_model(
    model: DetectorBase,
    loader,
    device,
    dist_thresh: float,
    decode_kwargs: Optional[Dict] = None,
    score_threshold: Optional[float] = None,
    sweep_threshold: bool = False,
) -> Dict:
    """Predict over a loader and compute the full metrics dict.

    Operating-point policy (the fairness rule): score-based detectors
    either receive a FIXED `score_threshold` (the one swept on val and
    stored in the checkpoint) or, with `sweep_threshold=True`, get the
    best-F1 threshold found on THIS loader — only ever do that on val.
    Sequence models (SoundReg) are exempt: their cardinality comes from
    EOS and their scores are diagnostics, not an operating point.
    """
    model.eval()
    preds, gts = [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        preds.extend(model.predict(batch, **(decode_kwargs or {})))
        gts.extend(batch["gt"])

    uses_threshold = preds and preds[0].scores is not None and not isinstance(
        model, _seq_types()
    )
    if uses_threshold and sweep_threshold:
        score_threshold, _ = sweep_score_threshold(preds, gts, dist_thresh)
    if uses_threshold and score_threshold is not None:
        preds = [apply_score_threshold(p, score_threshold) for p in preds]

    result = evaluate(preds, gts, dist_thresh)
    result["score_threshold"] = score_threshold
    result["eos"] = eos_calibration(preds, gts)
    # Raw predictions ride along (underscore = not for JSON) so callers
    # can run extra analyses without a second inference pass.
    result["_preds"] = preds
    result["_gts"] = gts
    return result


def _seq_types():
    # Imported lazily: trainer must not import model modules at module
    # level or the registries would have circular-import problems.
    from ..models.soundreg import SoundRegDetector

    return (SoundRegDetector,)


def scalar_metrics(result: Dict) -> Dict:
    """The JSON-safe scalar subset of an evaluate_model result."""
    keep = (
        "precision", "recall", "f1", "cardinality_mae",
        "range_mae", "azimuth_mae_deg", "score_threshold",
    )
    out = {k: result[k] for k in keep if result.get(k) is not None}
    if result.get("eos", {}).get("n_records"):
        out["eos_brier"] = result["eos"]["brier"]
    return out


# =====================================================================
# Trainer
# =====================================================================
class Trainer:
    def __init__(self, cfg, model: DetectorBase, datasets: Dict, run_dir):
        self.cfg = cfg
        self.model = model
        self.device = get_device(cfg.training.get("device", "auto"))
        self.model.to(self.device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        tc = cfg.training
        self.epochs = int(tc.epochs)
        self.dist_thresh = float(cfg.eval.dist_thresh_m)
        self.decode_kwargs = dict(cfg.eval.get("decode", {}))
        self.grad_clip = float(tc.get("grad_clip", 1.0))
        self.amp = bool(tc.get("amp", False)) and self.device.type == "cuda"

        nw = int(tc.get("num_workers", 0))
        self.train_loader = make_loader(datasets["train"], int(tc.batch_size), True, nw)
        self.val_loader = make_loader(datasets["val"], int(tc.batch_size), False, nw)

        self.opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tc.lr),
            weight_decay=float(tc.get("weight_decay", 0.01)),
        )
        # Per-step schedule: linear warmup over warmup_frac of all steps,
        # then cosine to zero.
        total = self.epochs * max(len(self.train_loader), 1)
        warmup = max(int(float(tc.get("warmup_frac", 0.05)) * total), 1)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.log_path = self.run_dir / "log.jsonl"
        self.best_f1 = -1.0

    def log(self, record: Dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def train(self) -> Dict:
        print(f"model params: {count_params(self.model):,} | device: {self.device}")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_logs = self._train_epoch()
            # Threshold swept on val every epoch (cheap) so the best
            # checkpoint always carries a matching operating point.
            val = evaluate_model(
                self.model, self.val_loader, self.device,
                self.dist_thresh, self.decode_kwargs, sweep_threshold=True,
            )
            record = {
                "epoch": epoch,
                "lr": self.sched.get_last_lr()[0],
                "time_s": round(time.time() - t0, 1),
                **{f"train_{k}": v for k, v in train_logs.items()},
                **{f"val_{k}": v for k, v in scalar_metrics(val).items()},
            }
            self.log(record)
            print(
                f"epoch {epoch:3d} | loss {train_logs['loss']:.4f} | "
                f"val F1 {val['f1']:.3f} (P {val['precision']:.3f} "
                f"R {val['recall']:.3f}) | "
                f"card {val['cardinality_mae']:.2f} | {record['time_s']}s"
            )
            ckpt = {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "val_f1": val["f1"],
                "score_threshold": val.get("score_threshold"),
            }
            torch.save(ckpt, self.run_dir / "last.pt")
            if val["f1"] > self.best_f1:
                self.best_f1 = val["f1"]
                torch.save(ckpt, self.run_dir / "best.pt")
        return {"best_val_f1": self.best_f1}

    def _train_epoch(self) -> Dict:
        self.model.train()
        totals: Dict[str, float] = {}
        n = 0
        for batch in self.train_loader:
            batch = batch_to_device(batch, self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                loss, logs = self.model.compute_loss(batch)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            for k, v in logs.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}
