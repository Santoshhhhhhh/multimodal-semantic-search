"""Evaluate text->image / image->text Recall@K on the Flickr30k test split.

Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.flickr30k_dataset import Flickr30kEvalDataset
from src.data.transforms import build_eval_transform
from src.models.dual_encoder import DualEncoder
from src.evaluation.retrieval_metrics import evaluate_recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoder(cfg["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = Flickr30kEvalDataset(
        images_dir=cfg["data"]["images_dir"],
        captions_file=cfg["data"]["captions_file"],
        split_file=cfg["data"]["test_split_file"],
        transform=build_eval_transform(cfg["data"]["image_size"]),
    )

    metrics = evaluate_recall_at_k(
        model, test_dataset, device, cfg["evaluation"]["batch_size"], cfg["evaluation"]["recall_ks"]
    )

    print("=== Flickr30k test-set retrieval metrics ===")
    for direction, scores in metrics.items():
        print(f"{direction}:")
        for k, v in scores.items():
            print(f"  {k}: {v * 100:.1f}%")


if __name__ == "__main__":
    main()
