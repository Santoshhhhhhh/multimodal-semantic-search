"""Flickr30k image-caption dataset.

Expects:
  - `images_dir/`: contains all Flickr30k JPEGs.
  - `captions_file`: tab-separated file in the standard Flickr30k format,
        `<image_name>#<caption_idx>\t<caption text>`
    e.g.  `1000092795.jpg#0\tTwo young guys with shaggy hair look at their hands...`
  - `split_file`: newline-separated list of image filenames belonging to the split.

Each image has `captions_per_image` (default 5) associated captions. During
training we randomly sample one caption per image per epoch (standard practice);
for evaluation, `Flickr30kEvalDataset` exposes all captions for full Recall@K.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset


def _load_captions(captions_file: str) -> dict[str, list[str]]:
    captions_by_image: dict[str, list[str]] = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, caption = line.split("\t")
            image_name = key.split("#")[0]
            captions_by_image[image_name].append(caption.strip())
    return captions_by_image


def _load_split(split_file: str) -> list[str]:
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class Flickr30kDataset(Dataset):
    """Training dataset: one (image, randomly-sampled caption) pair per item."""

    def __init__(self, images_dir: str, captions_file: str, split_file: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        captions_by_image = _load_captions(captions_file)
        split_images = set(_load_split(split_file))
        self.image_names = [name for name in split_images if name in captions_by_image]
        self.captions_by_image = captions_by_image

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = random.choice(self.captions_by_image[image_name])
        return {
            "image": image,
            "caption": caption,
            "sample_id": idx,  # used by the memory bank to identify true positives
        }


class Flickr30kEvalDataset(Dataset):
    """Evaluation dataset: exposes every (image, all captions) pair so Recall@K
    can be computed against the full caption set per image, matching standard
    Flickr30k retrieval benchmark protocol.
    """

    def __init__(self, images_dir: str, captions_file: str, split_file: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        captions_by_image = _load_captions(captions_file)
        split_images = set(_load_split(split_file))
        self.image_names = [name for name in split_images if name in captions_by_image]
        self.captions_by_image = captions_by_image

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "captions": self.captions_by_image[image_name],
            "image_idx": idx,
        }


def collate_train_batch(batch: list[dict]):
    import torch

    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    sample_ids = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)
    return {"images": images, "captions": captions, "sample_ids": sample_ids}
