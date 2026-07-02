"""Build a FAISS HNSW index over a directory of gallery images.

Usage:
    python scripts/build_index.py \
        --image-dir data/gallery \
        --checkpoint checkpoints/best.pt \
        --index-out index/gallery.hnsw
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.search_engine import SemanticSearchEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory of gallery images (e.g. 1K photos)")
    parser.add_argument("--checkpoint", required=True, help="Path to trained DualEncoder checkpoint (.pt)")
    parser.add_argument("--index-out", required=True, help="Output path for the FAISS index file")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    engine = SemanticSearchEngine(checkpoint_path=args.checkpoint)
    out_path = engine.build_index_from_directory(args.image_dir, args.index_out, batch_size=args.batch_size)
    n_images = len(engine.index.id_to_path)
    print(f"Indexed {n_images} images -> {out_path} (+ {out_path}.meta.json)")


if __name__ == "__main__":
    main()
