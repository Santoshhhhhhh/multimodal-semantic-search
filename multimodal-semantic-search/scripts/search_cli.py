"""Query the gallery index with a free-text description.

Usage:
    python scripts/search_cli.py \
        --index index/gallery.hnsw \
        --checkpoint checkpoints/best.pt \
        --query "two dogs playing in the snow" \
        --k 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.search_engine import SemanticSearchEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to the FAISS index file built by build_index.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    engine = SemanticSearchEngine(checkpoint_path=args.checkpoint, index_path=args.index)
    results = engine.search(args.query, k=args.k)

    print(f'Query: "{args.query}"')
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank:2d}. {score:.4f}  {path}")


if __name__ == "__main__":
    main()
