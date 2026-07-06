# Multimodal Semantic Search

A CLIP-style dual-encoder retrieval system that fine-tunes **DINOv2** (vision) and
**SBERT** (text) into a shared embedding space using **LoRA** adapters, **InfoNCE**
contrastive loss, and **hard-negative mining** — trained/evaluated on **Flickr30k**
for text→image retrieval, and deployed as a **FAISS (HNSW)** natural-language photo
search engine.

Benchmark target: **84.1% Recall@10** on Flickr30k text-to-image retrieval.

---

## 1. System Design

### 1.1 Goals & constraints

| Concern | Decision | Why |
|---|---|---|
| Two different pretrained towers (vision + text) | Dual-encoder ("two-tower") architecture, not a fused cross-encoder | Cross-encoders can't be pre-computed/indexed — need independent embeddings for ANN search at query time |
| Full fine-tuning is expensive & risks catastrophic forgetting | **LoRA** adapters on top of frozen DINOv2 + SBERT backbones | Few % of params trainable, fast convergence, backbones keep general visual/language priors |
| Vision and text embeddings live in different spaces/dims | Learned **linear projection heads** → shared `d=256` L2-normalized space | Standard CLIP-style alignment layer, decoupled from backbone dims |
| Only positive pairs are naturally labeled | **In-batch InfoNCE** (symmetric, image→text + text→image) as the base loss | Free negatives from batch, well-understood, differentiable Recall@K proxy |
| In-batch negatives are often "easy" (too dissimilar) → weak gradient signal | **Hard-negative mining**: online in-batch hardest-negative reweighting + an offline memory-bank of top-k confusable pairs, refreshed every epoch | Forces the model to separate visually/semantically similar but non-matching pairs, which is what actually limits Recall@K |
| Query-time latency over a large gallery | **FAISS HNSW** index (cosine/inner-product) over pre-computed image embeddings | Sub-linear ANN search, no exact brute force needed, good recall/latency tradeoff for galleries from 1K to millions |
| Need to serve "free-text query → ranked photos" | Thin `SearchEngine` class + CLI + Streamlit app, index built once and persisted to disk | Encode-once, search-many; index rebuild is decoupled from serving |

### 1.2 Architecture

```
                         ┌─────────────────────────┐
   image ──────────────▶ │   DINOv2 (frozen ViT)    │
                         │   + LoRA adapters (Q,V)  │──▶ pooled patch/CLS ──▶ Linear proj (256d) ──┐
                         └─────────────────────────┘                                              │
                                                                                                     ├─▶ L2-normalize ─▶ shared embedding space
                         ┌─────────────────────────┐                                              │        (cosine similarity)
   caption ─────────────▶│  SBERT (frozen encoder)  │                                              │
                         │  + LoRA adapters (Q,V)   │──▶ mean-pooled token emb ──▶ Linear proj(256d)┘
                         └─────────────────────────┘
```

- **Vision tower**: `facebook/dinov2-base` (ViT-B/14), CLS token pooled, frozen backbone
  with LoRA injected into attention `q_proj`/`v_proj` of the last N transformer blocks.
- **Text tower**: `sentence-transformers/all-mpnet-base-v2` (SBERT), mean-pooled
  token embeddings, frozen backbone with LoRA injected into attention `query`/`value`.
- **Projection heads**: independent `Linear(hidden_dim → 256)` per tower, always trainable.
- **Similarity**: cosine similarity (dot product of L2-normalized vectors) scaled by a
  learned temperature `τ`.

### 1.3 Training pipeline

1. **Batch sampling**: `(image, caption)` positive pairs from Flickr30k
   (5 captions/image → 1 sampled per step, all 5 available for eval).
2. **Encode** both modalities → `img_emb (B,256)`, `txt_emb (B,256)`, L2-normalized.
3. **In-batch similarity matrix** `S = img_emb @ txt_emb.T / τ`.
4. **Symmetric InfoNCE**: `L = 0.5 * (CE(S, diag) + CE(S.T, diag))`.
5. **Hard-negative mining** (two mechanisms, combined):
   - *Online*: within each batch, upweight the loss contribution of the
     highest-similarity non-matching pairs (hardest-negative-weighted InfoNCE).
   - *Offline memory bank*: maintain an embedding cache of the full training set
     (refreshed each epoch), retrieve top-K nearest non-matching captions/images
     per anchor, and inject a fraction of them into each batch as *additional*
     hard negatives beyond the in-batch ones.
6. **Optimizer**: AdamW on LoRA + projection-head params only; cosine LR schedule
   with warmup; backbone stays frozen (`requires_grad=False`).
7. **Checkpointing**: save LoRA adapter weights + projection heads (small, <50MB)
   rather than full backbones.

### 1.4 Evaluation

- Encode all Flickr30k **test** images once, and all 5 captions/image.
- For text→image: rank all images by cosine similarity to each caption, compute
  **Recall@1/5/10** (a hit if the paired image is in the top-K).
- Symmetric image→text Recall@K reported as a secondary metric.

### 1.5 Serving / Search pipeline

```
1K gallery images ─▶ vision tower (eval mode) ─▶ 256-d embeddings ─▶ FAISS HNSW index (persisted to disk)

free-text query ─▶ text tower (eval mode) ─▶ 256-d embedding ─▶ FAISS search(k) ─▶ ranked (image_path, score) list
```

- **Index**: `faiss.IndexHNSWFlat(256, M=32)` with inner product on normalized
  vectors (≡ cosine similarity), `efConstruction=200`, `efSearch=64` (tunable
  recall/latency knob).
- **Metadata store**: simple JSON/SQLite mapping FAISS internal ids → image paths,
  kept alongside the index file.
- **Interfaces**: `scripts/search_cli.py` (terminal), `app/streamlit_app.py`
  (natural-language photo gallery search UI).
