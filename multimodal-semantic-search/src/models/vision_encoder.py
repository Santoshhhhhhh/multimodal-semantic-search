"""DINOv2-backed vision tower with LoRA adapters and a projection head into the
shared multimodal embedding space.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class VisionEncoder(nn.Module):
    """Wraps a frozen DINOv2 backbone, injects LoRA adapters into the last N
    transformer blocks, and projects the pooled CLS embedding into a shared
    `embedding_dim`-dimensional space.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        n_unfrozen_blocks: int = 4,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Only expose the last `n_unfrozen_blocks` transformer layers to LoRA —
        # keeps the adapter small and focuses capacity on high-level features.
        total_layers = self.backbone.config.num_hidden_layers
        target_layer_idxs = set(range(total_layers - n_unfrozen_blocks, total_layers))
        target_modules = lora_target_modules or ["query", "value"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=self._resolve_target_modules(target_modules, target_layer_idxs),
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_dim),
        )

    def _resolve_target_modules(self, module_suffixes, layer_idxs) -> list[str]:
        """Build explicit module-name patterns so LoRA is only injected into the
        chosen transformer blocks (peft matches on substring/regex over module
        names, e.g. `encoder.layer.10.attention.attention.query`).
        """
        patterns = []
        for idx in layer_idxs:
            for suffix in module_suffixes:
                patterns.append(f"encoder.layer.{idx}.attention.attention.{suffix}")
        return patterns

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) -> returns L2-normalized (B, embedding_dim)."""
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # DINOv2 CLS token
        emb = self.projection(cls_token)
        return nn.functional.normalize(emb, p=2, dim=-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
