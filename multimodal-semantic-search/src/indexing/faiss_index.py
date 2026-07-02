"""FAISS HNSW index wrapper for the image gallery.

Vectors are L2-normalized 256-d embeddings, so inner-product search is
equivalent to cosine-similarity ranking. Metadata (FAISS internal id ->
image path) is persisted alongside the index as JSON.
"""
from __future__ import annotations

import json
import os

import faiss
import numpy as np


class GalleryIndex:
    def __init__(self, embedding_dim: int = 256, hnsw_M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.embedding_dim = embedding_dim
        index = faiss.IndexHNSWFlat(embedding_dim, hnsw_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        self.index = index
        self.id_to_path: dict[int, str] = {}

    def add(self, embeddings: np.ndarray, image_paths: list[str]):
        """embeddings: (N, D) float32, L2-normalized. image_paths: len N."""
        assert embeddings.shape[0] == len(image_paths)
        assert embeddings.shape[1] == self.embedding_dim
        start_id = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))
        for offset, path in enumerate(image_paths):
            self.id_to_path[start_id + offset] = path

    def search(self, query_embedding: np.ndarray, k: int = 10):
        """query_embedding: (D,) or (1, D) float32, L2-normalized.
        Returns list of (image_path, score) sorted by descending score.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]
        scores, ids = self.index.search(query_embedding.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.id_to_path[int(idx)], float(score)))
        return results

    def save(self, index_path: str):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        meta_path = index_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({str(k): v for k, v in self.id_to_path.items()}, f)

    @classmethod
    def load(cls, index_path: str, embedding_dim: int = 256, ef_search: int = 64) -> "GalleryIndex":
        obj = cls.__new__(cls)
        obj.embedding_dim = embedding_dim
        obj.index = faiss.read_index(index_path)
        obj.index.hnsw.efSearch = ef_search
        meta_path = index_path + ".meta.json"
        with open(meta_path, "r") as f:
            raw = json.load(f)
        obj.id_to_path = {int(k): v for k, v in raw.items()}
        return obj
