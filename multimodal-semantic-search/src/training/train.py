"""Training entrypoint: fine-tunes LoRA adapters + projection heads on Flickr30k
with hard-negative-weighted InfoNCE and an offline memory-bank hard-negative
miner.

Usage:
    python -m src.training.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.flickr30k_dataset import Flickr30kDataset, collate_train_batch
from src.data.transforms import build_train_transform
from src.models.dual_encoder import DualEncoder
from src.losses.infonce import HardNegativeInfoNCE
from src.mining.hard_negative_mining import MemoryBank
from src.evaluation.retrieval_metrics import evaluate_recall_at_k
from src.data.flickr30k_dataset import Flickr30kEvalDataset
from src.data.transforms import build_eval_transform


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: DualEncoder, cfg: dict):
    params = model.trainable_parameters()
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    train_transform = build_train_transform(cfg["data"]["image_size"])
    train_dataset = Flickr30kDataset(
        images_dir=cfg["data"]["images_dir"],
        captions_file=cfg["data"]["captions_file"],
        split_file=cfg["data"]["train_split_file"],
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_train_batch,
        drop_last=True,
        pin_memory=True,
    )

    val_transform = build_eval_transform(cfg["data"]["image_size"])
    val_dataset = Flickr30kEvalDataset(
        images_dir=cfg["data"]["images_dir"],
        captions_file=cfg["data"]["captions_file"],
        split_file=cfg["data"]["val_split_file"],
        transform=val_transform,
    )

    # --- Model / loss / optim ---
    model = DualEncoder(cfg["model"]).to(device)
    criterion = HardNegativeInfoNCE(hard_negative_weight=cfg["loss"]["hard_negative_weight"])

    memory_bank = None
    if cfg["loss"]["memory_bank"]["enabled"]:
        memory_bank = MemoryBank(
            size=cfg["loss"]["memory_bank"]["size"],
            embedding_dim=cfg["model"]["shared_embedding_dim"],
            device=device,
        )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    optimizer = build_optimizer(model, cfg["training"])
    scheduler = build_scheduler(optimizer, total_steps, cfg["training"]["warmup_ratio"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"])

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    best_recall_at_10 = 0.0
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            sample_ids = batch["sample_ids"].to(device, non_blocking=True)
            tokenized = model.text_encoder.tokenize(batch["captions"], device)

            with torch.cuda.amp.autocast(enabled=cfg["training"]["mixed_precision"]):
                img_emb, txt_emb, logit_scale = model(
                    images, tokenized["input_ids"], tokenized["attention_mask"]
                )

                extra_neg_img, extra_neg_txt = (None, None)
                if memory_bank is not None and memory_bank.filled > 0:
                    extra_neg_img, extra_neg_txt = memory_bank.get_hard_negatives(
                        img_emb, txt_emb, sample_ids, top_k=cfg["loss"]["memory_bank"]["top_k"]
                    )

                loss_dict = criterion(img_emb, txt_emb, logit_scale, extra_neg_img, extra_neg_txt)
                loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.clamp_logit_scale()

            if memory_bank is not None:
                memory_bank.update(img_emb, txt_emb, sample_ids)

            global_step += 1
            if global_step % cfg["training"]["log_every_n_steps"] == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc_i2t=f"{loss_dict['batch_acc_i2t'].item():.3f}",
                    acc_t2i=f"{loss_dict['batch_acc_t2i'].item():.3f}",
                )

        if (epoch + 1) % cfg["training"]["eval_every_n_epochs"] == 0:
            metrics = evaluate_recall_at_k(
                model, val_dataset, device, cfg["evaluation"]["batch_size"], cfg["evaluation"]["recall_ks"]
            )
            print(f"[epoch {epoch}] val metrics: {metrics}")
            r10 = metrics["text_to_image"].get("R@10", 0.0)
            if r10 > best_recall_at_10:
                best_recall_at_10 = r10
                ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], "best.pt")
                torch.save({"model_state_dict": model.state_dict(), "config": cfg, "metrics": metrics}, ckpt_path)
                print(f"  ↳ new best R@10={r10:.4f}, saved to {ckpt_path}")

    final_path = os.path.join(cfg["training"]["checkpoint_dir"], "last.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, final_path)
    print(f"Training complete. Best val R@10={best_recall_at_10:.4f}. Final checkpoint: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
