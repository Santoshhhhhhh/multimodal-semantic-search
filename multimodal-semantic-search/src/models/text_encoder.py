"""SBERT-backed text tower with LoRA adapters and a projection head into the
shared multimodal embedding space.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard SBERT mean-pooling: average token embeddings, masking out padding."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class TextEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 256,
        max_seq_len: int = 64,
        freeze_backbone: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        n_unfrozen_blocks: int = 4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.max_seq_len = max_seq_len
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
        patterns = []
        for idx in layer_idxs:
            for suffix in module_suffixes:
                patterns.append(f"encoder.layer.{idx}.attention.self.{suffix}")
        return patterns

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        emb = self.projection(pooled)
        return nn.functional.normalize(emb, p=2, dim=-1)

    def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Convenience: tokenize + forward in one call (used at inference time)."""
        batch = self.tokenize(texts, device)
        return self.forward(batch["input_ids"], batch["attention_mask"])

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
