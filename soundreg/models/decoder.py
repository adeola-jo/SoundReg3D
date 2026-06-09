"""Causal Transformer decoder, written from scratch (no HF dependency).

PUBLIC SURFACE
--------------
    CausalTransformerDecoder
        forward   teacher-forced logits over the whole sequence
        generate  greedy / sample / beam decoding with token-type masking

FLOW
----
    tokens (B, L)
        -> token emb + token-type emb + position emb
        -> N x [ masked self-attention -> cross-attention to memory -> MLP ]
        -> LayerNorm -> tied output head
        -> logits (B, L, vocab)

The encoder memory (B, S, D) enters through cross-attention only, exactly
as in AutoReg3D: the decoder stays conditioned on the scene features at
every step.

RESEARCH KNOBS THAT LIVE HERE
-----------------------------
    context_mode = "full" | "object_only"
        "full" is the normal causal decoder. "object_only" lets each
        token attend ONLY to BOS and to tokens of its own object — the
        conditioning ablation of the brief: predict each object without
        seeing previously emitted objects, with tokenization held fixed.
        If "full" beats "object_only", conditioning (not tokenization)
        does the work.

    token-type logit masking (generate only)
        Class steps emit class tokens, location steps location tokens;
        EOS is legal only at object boundaries. The model never wastes
        probability mass on impossible tokens at decode time.

    min_objects (generate only)
        Disallows EOS until at least min_objects are emitted — the
        length-controlled decoding variant we report next to greedy,
        because early-EOS is the known failure mode that silently drops
        quiet masked sources.

Why no KV cache: sequences are at most 2 + tokens_per_object * max_objects
tokens (default 18). Re-running the full forward per step costs nothing
at this scale; revisit only if scenes grow far beyond N ~ 8.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..data.tokenizer import BOS, EOS, PAD, SceneTokenizer


# =====================================================================
# Attention + block
# =====================================================================
class MultiHeadAttention(nn.Module):
    """Standard multi-head attention on top of F.scaled_dot_product_attention.

    Hand-rolled instead of nn.MultiheadAttention because the self-attention
    mask here is a *combination* of causal x context-mode x key-padding,
    which is awkward to express through the stock module and trivial to
    pass as one boolean mask to SDPA.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            q_in: (B, Lq, D) queries.
            kv_in: (B, Lk, D) keys/values (same tensor for self-attention,
                encoder memory for cross-attention).
            attn_mask: bool, broadcastable to (B, h, Lq, Lk);
                True = may attend. None = unrestricted (cross-attention).
        """
        b, lq, d = q_in.shape
        lk = kv_in.shape[1]
        h = self.n_heads
        q = self.q(q_in).view(b, lq, h, -1).transpose(1, 2)
        k = self.k(kv_in).view(b, lk, h, -1).transpose(1, 2)
        v = self.v(kv_in).view(b, lk, h, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, lq, d))


class DecoderBlock(nn.Module):
    """Pre-LN decoder block: masked self-attn -> cross-attn -> MLP."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, self_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), self.ln1(x), attn_mask=self_mask)
        x = x + self.cross_attn(self.ln2(x), memory)
        x = x + self.mlp(self.ln3(x))
        return x


# =====================================================================
# Decoder
# =====================================================================
class CausalTransformerDecoder(nn.Module):
    # Token-type embedding size: special / class / range / azimuth.
    N_TYPES = 4

    def __init__(self, cfg: Dict, tokenizer: SceneTokenizer, max_objects: int):
        """Args:
            cfg: The `model.decoder` config section: d_model, n_layers,
                n_heads, d_ff, dropout, context_mode, tie_embeddings.
            tokenizer: Supplies vocab size, type schedule, legal-token masks.
            max_objects: Hard cap on emitted objects; fixes the position
                embedding table size via tokenizer.max_seq_len.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.context_mode = cfg.get("context_mode", "full")
        d_model = int(cfg["d_model"])
        max_len = tokenizer.max_seq_len(max_objects)

        self.tok_emb = nn.Embedding(tokenizer.vocab_size, d_model)
        self.type_emb = nn.Embedding(self.N_TYPES, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(cfg.get("dropout", 0.1))
        self.blocks = nn.ModuleList(
            DecoderBlock(
                d_model,
                int(cfg["n_heads"]),
                int(cfg.get("d_ff", 4 * d_model)),
                cfg.get("dropout", 0.1),
            )
            for _ in range(int(cfg["n_layers"]))
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tokenizer.vocab_size, bias=False)
        # Weight tying: with a ~100-token vocabulary the head is tiny
        # anyway, but tying is standard practice and costs nothing.
        if cfg.get("tie_embeddings", True):
            self.head.weight = self.tok_emb.weight

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    # =================================================================
    # Self-attention mask
    # =================================================================
    def _self_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Build the combined self-attention mask.

        Three constraints AND-ed together:
            1. causal: position p attends to positions <= p,
            2. context_mode == "object_only": additionally restricted to
               BOS + the token's own object triple,
            3. PAD positions are never attended as keys.

        The diagonal is forced open at the end: an all-False attention
        row makes SDPA emit NaNs, and padded query rows (whose loss is
        masked anyway) would otherwise poison the backward pass.

        Returns:
            bool (B, 1, L, L); True = may attend.
        """
        b, length = tokens.shape
        dev = tokens.device
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=dev))
        if self.context_mode == "object_only":
            tpo = self.tokenizer.tokens_per_object
            # Object index per position: BOS = -1, then 0, 0, 1, 1, ...
            # (for the 2-token single-class layout).
            obj = torch.full((length,), -1, dtype=torch.long, device=dev)
            obj[1:] = torch.arange(length - 1, device=dev) // tpo
            same = obj[:, None] == obj[None, :]
            bos_col = torch.zeros(length, dtype=torch.bool, device=dev)
            bos_col[0] = True
            mask = mask & (same | bos_col[None, :])
        mask = mask[None, None].expand(b, 1, length, length).clone()
        notpad = (tokens != PAD)[:, None, None, :]
        mask = mask & notpad
        eye = torch.eye(length, dtype=torch.bool, device=dev)[None, None]
        return mask | eye

    # =================================================================
    # Forward (teacher forcing)
    # =================================================================
    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: (B, L) input ids, BOS at position 0 (i.e. the target
                sequence shifted right by one).
            memory: (B, S, D) encoder tokens.

        Returns:
            (B, L, vocab) next-token logits.
        """
        b, length = tokens.shape
        pos = torch.arange(length, device=tokens.device)
        x = (
            self.tok_emb(tokens)
            + self.type_emb(self.tokenizer.type_ids(tokens))
            + self.pos_emb(pos)[None]
        )
        x = self.drop(x)
        self_mask = self._self_mask(tokens)
        for block in self.blocks:
            x = block(x, memory, self_mask)
        return self.head(self.ln_f(x))

    # =================================================================
    # Decoding
    # =================================================================
    def _step_logits(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        step: int,
        min_objects: int,
    ) -> torch.Tensor:
        """Logits for the next token, with illegal types set to -inf.

        Token-type masking guarantees structurally valid sequences; the
        min_objects clause additionally vetoes EOS until enough objects
        are out (length-controlled decoding).
        """
        logits = self.forward(tokens, memory)[:, -1]  # (B, V)
        valid = self.tokenizer.valid_next_mask(step, device=tokens.device)
        tpo = self.tokenizer.tokens_per_object
        if step % tpo == 0 and (step // tpo) < min_objects:
            valid = valid.clone()
            valid[EOS] = False
        return logits.masked_fill(~valid[None], float("-inf"))

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        # ---- strategy ----
        mode: str = "greedy",
        # ---- cardinality controls ----
        max_objects: int = 12,
        min_objects: int = 0,
        # ---- sampling controls (mode == "sample") ----
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        # ---- beam controls (mode == "beam") ----
        beam_size: int = 4,
    ) -> List[Dict]:
        """Decode object sequences from encoder memory.

        Every emitted P(EOS) at an object boundary is recorded — that is
        the raw material for the EOS-calibration diagnostic, which we
        watch from epoch 1 (early EOS silently drops the quiet masked
        sources this whole project is about).

        Returns:
            One dict per batch element:
                tokens          int array, starts with BOS
                token_logprobs  list, log-prob of each emitted token
                eos_probs       list, P(EOS) at each boundary visited
        """
        if mode == "beam":
            return [
                self._beam_single(memory[i : i + 1], max_objects, min_objects, beam_size)
                for i in range(memory.shape[0])
            ]

        b = memory.shape[0]
        dev = memory.device
        tokens = torch.full((b, 1), BOS, dtype=torch.long, device=dev)
        finished = torch.zeros(b, dtype=torch.bool, device=dev)
        tok_logprobs: List[List[float]] = [[] for _ in range(b)]
        eos_probs: List[List[float]] = [[] for _ in range(b)]
        tpo = self.tokenizer.tokens_per_object

        # +1 step: after max_objects full objects there is exactly one
        # boundary left and EOS is forced there.
        for step in range(max_objects * tpo + 1):
            logits = self._step_logits(tokens, memory, step, min_objects)
            logp = F.log_softmax(logits / temperature, dim=-1)

            if step % tpo == 0:
                p_eos = logp[:, EOS].exp()
                for i in range(b):
                    if not finished[i]:
                        eos_probs[i].append(float(p_eos[i]))

            if step >= max_objects * tpo:
                nxt = torch.full((b,), EOS, dtype=torch.long, device=dev)
            elif mode == "greedy":
                nxt = logp.argmax(dim=-1)
            elif mode == "sample":
                probs = logp.exp()
                if top_p is not None:
                    probs = _top_p_filter(probs, top_p)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                raise ValueError(f"Unknown decode mode '{mode}'")

            step_lp = logp.gather(1, nxt[:, None]).squeeze(1)
            for i in range(b):
                if not finished[i]:
                    tok_logprobs[i].append(float(step_lp[i]))
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            finished = finished | (nxt == EOS)
            if bool(finished.all()):
                break

        out = []
        toks = tokens.cpu().numpy()
        for i in range(b):
            out.append(
                {
                    "tokens": toks[i],
                    "token_logprobs": tok_logprobs[i],
                    "eos_probs": eos_probs[i],
                }
            )
        return out

    def _beam_single(
        self, memory: torch.Tensor, max_objects: int, min_objects: int, beam_size: int
    ) -> Dict:
        """Beam search for ONE scene (batch of 1).

        Eval-time only, so the per-sample Python loop is acceptable:
        sequences are ~18 tokens, beams are small, and clarity beats
        batched beam bookkeeping here. Beams are ranked by
        length-normalized log-prob so longer (more-object) hypotheses
        are not penalized for merely being longer.
        """
        tpo = self.tokenizer.tokens_per_object
        dev = memory.device
        beams = [
            {"tokens": [BOS], "logp": 0.0, "lps": [], "eos": [], "done": False}
        ]
        for step in range(max_objects * tpo + 1):
            if all(bm["done"] for bm in beams):
                break
            candidates = []
            for bm in beams:
                if bm["done"]:
                    candidates.append(bm)
                    continue
                t = torch.tensor([bm["tokens"]], dtype=torch.long, device=dev)
                logits = self._step_logits(t, memory, step, min_objects)
                logp = F.log_softmax(logits, dim=-1)[0]
                eos = bm["eos"] + (
                    [float(logp[EOS].exp())] if step % tpo == 0 else []
                )
                if step >= max_objects * tpo:
                    top_lp, top_ix = logp[[EOS]], torch.tensor([EOS], device=dev)
                else:
                    top_lp, top_ix = logp.topk(min(beam_size, (logp > -math.inf).sum()))
                for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                    candidates.append(
                        {
                            "tokens": bm["tokens"] + [ix],
                            "logp": bm["logp"] + lp,
                            "lps": bm["lps"] + [lp],
                            "eos": eos,
                            "done": ix == EOS,
                        }
                    )
            candidates.sort(key=lambda c: c["logp"] / max(len(c["lps"]), 1), reverse=True)
            beams = candidates[:beam_size]
        best = max(beams, key=lambda c: c["logp"] / max(len(c["lps"]), 1))
        import numpy as np

        return {
            "tokens": np.asarray(best["tokens"], dtype=np.int64),
            "token_logprobs": best["lps"],
            "eos_probs": best["eos"],
        }


# =====================================================================
# Sampling helpers
# =====================================================================
def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest prefix of the sorted
    distribution whose mass reaches top_p, zero the rest, renormalize.
    The top-1 token is always kept (`cum - sorted_p < top_p` is True at
    rank 0 by construction), so the filter can never empty the support.
    """
    sorted_p, idx = probs.sort(dim=-1, descending=True)
    cum = sorted_p.cumsum(dim=-1)
    keep = cum - sorted_p < top_p
    filtered = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, idx, filtered)
