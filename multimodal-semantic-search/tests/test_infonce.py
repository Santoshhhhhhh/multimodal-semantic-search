"""Sanity tests for the hard-negative InfoNCE loss. Pure PyTorch, no pretrained
model downloads required — run with: pytest tests/test_infonce.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.infonce import HardNegativeInfoNCE


def test_loss_is_low_for_perfectly_aligned_embeddings():
    torch.manual_seed(0)
    batch_size, dim = 8, 16
    base = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)
    img_emb = base.clone()
    txt_emb = base.clone()  # identical -> perfect positives, should give near-zero loss
    logit_scale = torch.tensor(10.0)

    criterion = HardNegativeInfoNCE(hard_negative_weight=0.5)
    out = criterion(img_emb, txt_emb, logit_scale)

    assert out["loss"].item() < 0.5
    assert out["batch_acc_i2t"].item() == 1.0
    assert out["batch_acc_t2i"].item() == 1.0


def test_loss_decreases_when_embeddings_move_toward_alignment():
    torch.manual_seed(1)
    batch_size, dim = 8, 16
    img_emb = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)
    txt_emb_far = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1)
    txt_emb_close = torch.nn.functional.normalize(img_emb + 0.05 * torch.randn(batch_size, dim), dim=-1)
    logit_scale = torch.tensor(10.0)

    criterion = HardNegativeInfoNCE(hard_negative_weight=0.5)
    loss_far = criterion(img_emb, txt_emb_far, logit_scale)["loss"].item()
    loss_close = criterion(img_emb, txt_emb_close, logit_scale)["loss"].item()

    assert loss_close < loss_far


def test_memory_bank_negatives_widen_logits_shape():
    from src.mining.hard_negative_mining import MemoryBank

    device = torch.device("cpu")
    dim = 16
    bank = MemoryBank(size=32, embedding_dim=dim, device=device)

    img_emb = torch.nn.functional.normalize(torch.randn(8, dim), dim=-1)
    txt_emb = torch.nn.functional.normalize(torch.randn(8, dim), dim=-1)
    sample_ids = torch.arange(8)
    bank.update(img_emb, txt_emb, sample_ids)

    extra_neg_img, extra_neg_txt = bank.get_hard_negatives(img_emb, txt_emb, sample_ids, top_k=4)
    assert extra_neg_img is not None and extra_neg_txt is not None
    assert extra_neg_img.shape[1] == dim
    assert extra_neg_txt.shape[1] == dim


if __name__ == "__main__":
    test_loss_is_low_for_perfectly_aligned_embeddings()
    test_loss_decreases_when_embeddings_move_toward_alignment()
    test_memory_bank_negatives_widen_logits_shape()
    print("All tests passed.")
