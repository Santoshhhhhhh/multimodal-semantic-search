"""Dual-encoder ("two-tower") model combining the vision and text encoders into
a single module with a shared learned temperature, mirroring the CLIP training
interface while keeping the towers independently callable for indexing/serving.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder


class DualEncoder(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        v = model_cfg["vision"]
        t = model_cfg["text"]
        embedding_dim = model_cfg["shared_embedding_dim"]

        self.vision_encoder = VisionEncoder(
            backbone_name=v["backbone"],
            embedding_dim=embedding_dim,
            freeze_backbone=v.get("freeze_backbone", True),
            lora_r=v["lora"]["r"],
            lora_alpha=v["lora"]["alpha"],
            lora_dropout=v["lora"]["dropout"],
            lora_target_modules=v["lora"]["target_modules"],
            n_unfrozen_blocks=v["lora"]["n_unfrozen_blocks"],
        )
        self.text_encoder = TextEncoder(
            backbone_name=t["backbone"],
            embedding_dim=embedding_dim,
            max_seq_len=t.get("max_seq_len", 64),
            freeze_backbone=t.get("freeze_backbone", True),
            lora_r=t["lora"]["r"],
            lora_alpha=t["lora"]["alpha"],
            lora_dropout=t["lora"]["dropout"],
            lora_target_modules=t["lora"]["target_modules"],
            n_unfrozen_blocks=t["lora"]["n_unfrozen_blocks"],
        )

        init_temp = model_cfg.get("logit_temperature_init", 0.07)
        # Store as log-scale learnable logit_scale (CLIP convention: logit_scale = 1/temp)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def clamp_logit_scale(self, min_temp: float = 0.01, max_temp: float = 0.5):
        max_scale = math.log(1.0 / min_temp)
        min_scale = math.log(1.0 / max_temp)
        with torch.no_grad():
            self.logit_scale.clamp_(min_scale, max_scale)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(pixel_values)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Returns (img_emb, txt_emb, logit_scale) — all downstream loss/eval
        code operates on the L2-normalized embeddings + scalar temperature.
        """
        img_emb = self.encode_image(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb, self.logit_scale.exp()

    def trainable_parameters(self):
        params = list(self.vision_encoder.trainable_parameters())
        params += list(self.text_encoder.trainable_parameters())
        params.append(self.logit_scale)
        return params
