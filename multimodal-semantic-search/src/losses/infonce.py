"""Symmetric InfoNCE contrastive loss with hard-negative weighting.

Two mechanisms combine to sharpen the gradient signal beyond vanilla in-batch
InfoNCE:

1. Hardest-negative up-weighting: within the in-batch similarity matrix, the
   non-matching pair with the highest similarity per anchor gets an extra loss
   weight (`hard_negative_weight`), so the model is penalized more for
   confusing visually/semantically close negatives than for trivially
   dissimilar ones.
2. Memory-bank negatives (optional, supplied externally by
   `mining.hard_negative_mining`): additional hard negative embeddings mined
   offline across the whole dataset are concatenated to the in-batch negatives
   before computing the softmax, widening the negative pool beyond the batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeInfoNCE(nn.Module):
    def __init__(self, hard_negative_weight: float = 0.5):
        super().__init__()
        self.hard_negative_weight = hard_negative_weight

    @staticmethod
    def _weighted_cross_entropy(logits: torch.Tensor, hard_weight: float) -> torch.Tensor:
        """logits: (B, B + extra_negatives), the diagonal (first B columns) holds
        the positive pair similarity for each row. Applies extra weight to the
        hardest (highest-similarity) negative per row before the CE loss.
        """
        batch_size = logits.size(0)
        targets = torch.arange(batch_size, device=logits.device)

        # Identify, per row, the highest-similarity *negative* column and scale it.
        with torch.no_grad():
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size), targets] = False
            neg_logits = logits.masked_fill(~mask, float("-inf"))
            hardest_idx = neg_logits.argmax(dim=1)

        weight_add = torch.zeros_like(logits)
        weight_add[torch.arange(batch_size), hardest_idx] = hard_weight
        weighted_logits = logits + weight_add  # slightly sharpens the hardest negative's logit

        return F.cross_entropy(weighted_logits, targets)

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        logit_scale: torch.Tensor,
        extra_neg_img: torch.Tensor | None = None,
        extra_neg_txt: torch.Tensor | None = None,
    ) -> dict:
        """img_emb, txt_emb: (B, D) L2-normalized.
        extra_neg_img: (K, D) memory-bank hard-negative image embeddings (optional).
        extra_neg_txt: (K, D) memory-bank hard-negative text embeddings (optional).
        """
        # image -> text logits, optionally widened with memory-bank text negatives
        txt_pool = txt_emb if extra_neg_txt is None else torch.cat([txt_emb, extra_neg_txt], dim=0)
        logits_i2t = logit_scale * img_emb @ txt_pool.t()

        # text -> image logits, optionally widened with memory-bank image negatives
        img_pool = img_emb if extra_neg_img is None else torch.cat([img_emb, extra_neg_img], dim=0)
        logits_t2i = logit_scale * txt_emb @ img_pool.t()

        loss_i2t = self._weighted_cross_entropy(logits_i2t, self.hard_negative_weight)
        loss_t2i = self._weighted_cross_entropy(logits_t2i, self.hard_negative_weight)
        loss = 0.5 * (loss_i2t + loss_t2i)

        with torch.no_grad():
            batch_size = img_emb.size(0)
            targets = torch.arange(batch_size, device=img_emb.device)
            acc_i2t = (logits_i2t[:, :batch_size].argmax(dim=1) == targets).float().mean()
            acc_t2i = (logits_t2i[:, :batch_size].argmax(dim=1) == targets).float().mean()

        return {
            "loss": loss,
            "loss_i2t": loss_i2t.detach(),
            "loss_t2i": loss_t2i.detach(),
            "batch_acc_i2t": acc_i2t,
            "batch_acc_t2i": acc_t2i,
        }
