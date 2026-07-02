"""End-to-end natural-language photo search: load a trained DualEncoder
checkpoint, embed a text query with the text tower, and search the persisted
FAISS HNSW gallery index.
"""
from __future__ import annotations

import glob
import os

import torch

from src.indexing.faiss_index import GalleryIndex
from src.models.dual_encoder import DualEncoder
from src.data.transforms import build_eval_transform


class SemanticSearchEngine:
    def __init__(self, checkpoint_path: str, index_path: str | None = None, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = ckpt["config"]
        self.model = DualEncoder(self.cfg["model"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transform = build_eval_transform(self.cfg["data"]["image_size"])

        self.index = None
        if index_path and os.path.exists(index_path):
            self.index = GalleryIndex.load(
                index_path,
                embedding_dim=self.cfg["model"]["shared_embedding_dim"],
                ef_search=self.cfg["indexing"]["hnsw_ef_search"],
            )

    # ---- Indexing (gallery build) ----

    @torch.no_grad()
    def build_index_from_directory(self, image_dir: str, index_out_path: str, batch_size: int = 32):
        from PIL import Image
        import numpy as np

        image_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob.glob(os.path.join(image_dir, "**", ext), recursive=True)
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found under {image_dir}")

        index_cfg = self.cfg["indexing"]
        index = GalleryIndex(
            embedding_dim=index_cfg["embedding_dim"],
            hnsw_M=index_cfg["hnsw_M"],
            ef_construction=index_cfg["hnsw_ef_construction"],
            ef_search=index_cfg["hnsw_ef_search"],
        )

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                tensors.append(self.transform(img))
            batch = torch.stack(tensors).to(self.device)
            embeddings = self.model.encode_image(batch).cpu().numpy().astype(np.float32)
            index.add(embeddings, batch_paths)

        index.save(index_out_path)
        self.index = index
        return index_out_path

    # ---- Query ----

    @torch.no_grad()
    def search(self, query_text: str, k: int = 10):
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index_from_directory() first or pass index_path.")
        query_emb = self.model.text_encoder.encode_texts([query_text], self.device)
        query_emb = query_emb.cpu().numpy()
        return self.index.search(query_emb[0], k=k)
