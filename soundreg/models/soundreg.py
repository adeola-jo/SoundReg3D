"""SoundRegDetector: audio encoder + causal decoder behind one loss.

FLOW
----
    batch[input_key]                    audio features (key set by encoder)
        -> encoder                      memory (B, S, D)
    batch["tokens"]                     [BOS, ordered objects, EOS]
        -> decoder (teacher forcing)    logits (B, L-1, vocab)
        -> single cross-entropy         brief Eq. 6 / AutoReg3D Eq. 2

One CE over all token types — no per-attribute losses, no matching, no
weighting knobs. That single-loss property is half the appeal of the
autoregressive formulation; keep it that way.

INFERENCE
---------
predict() runs decoder.generate and detokenizes. Cardinality comes from
EOS — there is NO score threshold anywhere in this detector. The
per-object `scores` it still attaches are

    exp( mean log-prob of the object's tokens )

for diagnostics (confidence vs. error analysis), never for filtering.
predict() also forwards the per-boundary P(EOS) record that feeds the
EOS-calibration diagnostic.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..data.tokenizer import PAD, SceneTokenizer
from ..registry import MODELS
from ..types import ScenePrediction
from .base import DetectorBase
from .decoder import CausalTransformerDecoder


# =====================================================================
# Detector
# =====================================================================
@MODELS.register("soundreg")
class SoundRegDetector(DetectorBase):
    """Encoder + CausalTransformerDecoder, trained with teacher forcing."""

    def __init__(self, model_cfg: Dict, encoder: nn.Module, tokenizer: SceneTokenizer):
        """Args:
            model_cfg: The `model` config section (max_objects, decoder.*,
                loss.soft_sigma_bins).
            encoder: Any module mapping batch[input_key] -> (B, S, D)
                with attributes `d_model` and `input_key`. Built by the
                factory from the ENCODERS registry.
            tokenizer: Shared with the dataset — the decoder must speak
                exactly the vocabulary the dataset wrote.
        """
        super().__init__()
        self.encoder = encoder
        self.input_key = encoder.input_key
        self.tokenizer = tokenizer
        self.max_objects = int(model_cfg.get("max_objects", 12))
        # soft_sigma_bins > 0 switches the hard CE to Gaussian soft
        # targets over location bins (the soft-target ablation).
        self.soft_sigma_bins = float(
            model_cfg.get("loss", {}).get("soft_sigma_bins", 0.0)
        )
        dec_cfg = dict(model_cfg["decoder"])
        if int(dec_cfg["d_model"]) != encoder.d_model:
            raise ValueError(
                f"decoder d_model {dec_cfg['d_model']} != encoder d_model "
                f"{encoder.d_model}; cross-attention needs matching widths."
            )
        self.decoder = CausalTransformerDecoder(dec_cfg, tokenizer, self.max_objects)

    # =================================================================
    # Training
    # =================================================================
    def compute_loss(self, batch: Dict):
        """Teacher-forced next-token loss.

        Input is tokens[:, :-1] (BOS..last object token), target is
        tokens[:, 1:] (first object token..EOS). PAD positions are
        excluded — they exist only because scenes in a batch have
        different N.

        Returns:
            (scalar loss, {"loss", "token_acc"}). token_acc is the
            teacher-forced next-token accuracy: cheap, and a far earlier
            signal than F1 during the first epochs.
        """
        tokens = batch["tokens"]
        memory = self.encoder(batch[self.input_key])
        logits = self.decoder(tokens[:, :-1], memory)
        targets = tokens[:, 1:]

        if self.soft_sigma_bins > 0:
            soft = self.tokenizer.soft_target_distribution(
                targets, self.soft_sigma_bins
            )
            logp = F.log_softmax(logits, dim=-1)
            nll = -(soft * logp).sum(-1)
            mask = (targets != PAD).float()
            loss = (nll * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=PAD,
            )

        with torch.no_grad():
            mask = targets != PAD
            acc = (logits.argmax(-1)[mask] == targets[mask]).float().mean()
        return loss, {"loss": float(loss), "token_acc": float(acc)}

    # =================================================================
    # Inference
    # =================================================================
    @torch.no_grad()
    def predict(self, batch: Dict, **kw) -> List[ScenePrediction]:
        """Decode object lists.

        Keyword args are forwarded to decoder.generate: mode ("greedy" |
        "sample" | "beam"), max_objects, min_objects (length-controlled
        variant), temperature, top_p, beam_size.
        """
        memory = self.encoder(batch[self.input_key])
        results = self.decoder.generate(
            memory,
            mode=kw.get("mode", "greedy"),
            max_objects=kw.get("max_objects", self.max_objects),
            min_objects=kw.get("min_objects", 0),
            temperature=kw.get("temperature", 1.0),
            top_p=kw.get("top_p"),
            beam_size=kw.get("beam_size", 4),
        )
        preds = []
        tpo = self.tokenizer.tokens_per_object
        for res in results:
            pred = self.tokenizer.decode(res["tokens"])
            lps = np.asarray(res["token_logprobs"])
            scores = []
            for k in range(pred.n):
                obj_lps = lps[k * tpo : (k + 1) * tpo]
                scores.append(float(np.exp(obj_lps.mean())) if len(obj_lps) else 0.0)
            pred.scores = np.asarray(scores)
            pred.extras = {
                "eos_probs": np.asarray(res["eos_probs"]),
                "tokens": res["tokens"],
            }
            preds.append(pred)
        return preds
