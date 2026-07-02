"""Recall@K computation for text<->image retrieval, following the standard
Flickr30k evaluation protocol: each image has 5 ground-truth captions; a
text→image query is a "hit" if the paired image appears in the top-K ranked
images, and an image→text query is a "hit" if any of its 5 captions appears
in the top-K ranked captions.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def _encode_eval_set(model, dataset, device, batch_size):
    """Encodes all images once, and all captions (flattened, with an
    image-index back-pointer) once. Returns:
        image_embeddings: (N_images, D)
        caption_embeddings: (N_captions, D)
        caption_to_image: (N_captions,) long tensor mapping caption idx -> image idx
    """
    model.eval()

    image_embeddings = []
    all_captions = []
    caption_to_image = []

    # Iterate at the item level (not via DataLoader) since default collate
    # mangles the variable-length list[list[str]] caption field.
    image_cursor = 0
    image_tensors = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        image_tensors.append(item["image"])
        for cap in item["captions"]:
            all_captions.append(cap)
            caption_to_image.append(image_cursor)
        image_cursor += 1

    images_stacked = torch.stack(image_tensors)
    for start in range(0, images_stacked.size(0), batch_size):
        chunk = images_stacked[start:start + batch_size].to(device)
        emb = model.encode_image(chunk)
        image_embeddings.append(emb.cpu())

    caption_embeddings = []
    for start in range(0, len(all_captions), batch_size):
        chunk = all_captions[start:start + batch_size]
        tokenized = model.text_encoder.tokenize(chunk, device)
        emb = model.encode_text(tokenized["input_ids"], tokenized["attention_mask"])
        caption_embeddings.append(emb.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    caption_embeddings = torch.cat(caption_embeddings, dim=0)
    caption_to_image = torch.tensor(caption_to_image, dtype=torch.long)
    return image_embeddings, caption_embeddings, caption_to_image


def _recall_at_k(similarity: torch.Tensor, ground_truth, ks):
    """similarity: (n_queries, n_candidates). ground_truth[i] is either a single
    int (the correct candidate index) or a set/list of acceptable indices.
    Returns {f"R@{k}": recall_fraction}.
    """
    n_queries = similarity.size(0)
    ranked = similarity.argsort(dim=1, descending=True)  # (n_queries, n_candidates)

    results = {k: 0 for k in ks}
    max_k = max(ks)
    top_k_indices = ranked[:, :max_k]

    for i in range(n_queries):
        gt = ground_truth[i]
        gt_set = {gt} if isinstance(gt, int) else set(gt)
        row = top_k_indices[i].tolist()
        for k in ks:
            if gt_set.intersection(row[:k]):
                results[k] += 1

    return {f"R@{k}": results[k] / n_queries for k in ks}


@torch.no_grad()
def evaluate_recall_at_k(model, eval_dataset, device, batch_size, ks) -> dict:
    image_emb, caption_emb, caption_to_image = _encode_eval_set(model, eval_dataset, device, batch_size)

    # text -> image: query = caption, candidates = images, gt = single image index
    sim_t2i = caption_emb @ image_emb.t()
    gt_t2i = caption_to_image.tolist()
    text_to_image = _recall_at_k(sim_t2i, gt_t2i, ks)

    # image -> text: query = image, candidates = captions, gt = set of caption indices for that image
    gt_i2t = [
        [j for j, img_idx in enumerate(caption_to_image.tolist()) if img_idx == i]
        for i in range(image_emb.size(0))
    ]
    sim_i2t = image_emb @ caption_emb.t()
    image_to_text = _recall_at_k(sim_i2t, gt_i2t, ks)

    return {"text_to_image": text_to_image, "image_to_text": image_to_text}
