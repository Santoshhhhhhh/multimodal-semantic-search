"""Offline hard-negative mining via a rolling memory bank of embeddings.

In-batch negatives only expose the model to whatever happens to co-occur in a
random batch. To surface *dataset-wide* hard negatives (e.g. two different
beach photos with near-identical captions), we maintain a FIFO memory bank of
recently computed image/text embeddings + their sample ids, refreshed as
training progresses, and query it for the top-K most-similar-but-non-matching
embeddings for each anchor in a batch.
"""
from __future__ import annotations

import torch


class MemoryBank:
    def __init__(self, size: int, embedding_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.img_bank = torch.zeros(size, embedding_dim, device=device)
        self.txt_bank = torch.zeros(size, embedding_dim, device=device)
        self.sample_ids = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def update(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, sample_ids: torch.Tensor):
        """FIFO-enqueue a batch of (already detached) embeddings + their dataset ids."""
        b = img_emb.size(0)
        idx = (torch.arange(b, device=self.device) + self.ptr) % self.size
        self.img_bank[idx] = img_emb.detach()
        self.txt_bank[idx] = txt_emb.detach()
        self.sample_ids[idx] = sample_ids.to(self.device)
        self.ptr = (self.ptr + b) % self.size
        self.filled = min(self.size, self.filled + b)

    @torch.no_grad()
    def get_hard_negatives(
        self,
        anchor_img: torch.Tensor,
        anchor_txt: torch.Tensor,
        anchor_ids: torch.Tensor,
        top_k: int,
    ):
        """For each anchor pair, retrieve the top_k memory-bank text embeddings
        most similar to the anchor image (hard negatives for the i2t direction)
        and the top_k image embeddings most similar to the anchor text
        (hard negatives for the t2i direction). Excludes bank entries that
        share the anchor's own sample id (i.e. not actually a negative).
        """
        if self.filled == 0:
            return None, None

        bank_img = self.img_bank[: self.filled]
        bank_txt = self.txt_bank[: self.filled]
        bank_ids = self.sample_ids[: self.filled]

        # image anchor vs bank text -> hard text negatives
        sim_i2bank_txt = anchor_img @ bank_txt.t()  # (B, filled)
        # text anchor vs bank image -> hard image negatives
        sim_t2bank_img = anchor_txt @ bank_img.t()  # (B, filled)

        same_id_mask = anchor_ids.unsqueeze(1) == bank_ids.unsqueeze(0)  # (B, filled)
        sim_i2bank_txt = sim_i2bank_txt.masked_fill(same_id_mask, float("-inf"))
        sim_t2bank_img = sim_t2bank_img.masked_fill(same_id_mask, float("-inf"))

        k = min(top_k, self.filled)
        hard_txt_idx = sim_i2bank_txt.topk(k, dim=1).indices  # (B, k)
        hard_img_idx = sim_t2bank_img.topk(k, dim=1).indices  # (B, k)

        # Flatten + dedupe to build a compact extra-negative pool for the loss.
        hard_txt_flat = torch.unique(hard_txt_idx.reshape(-1))
        hard_img_flat = torch.unique(hard_img_idx.reshape(-1))

        extra_neg_txt = bank_txt[hard_txt_flat]
        extra_neg_img = bank_img[hard_img_flat]
        return extra_neg_img, extra_neg_txt
