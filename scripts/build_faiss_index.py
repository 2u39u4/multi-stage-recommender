"""Build the FAISS HNSW index used by the W5 serving stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neorec.serving.faiss_index import build_hnsw, save_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-emb", default="artifacts/recall_oof/two_tower/item_vecs.npy")
    parser.add_argument("--out", default="artifacts/serving/faiss_hnsw.index")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=64)
    args = parser.parse_args()

    item_emb = Path(args.item_emb)
    if not item_emb.exists():
        raise FileNotFoundError(
            f"Missing {item_emb}. Train Two-Tower first, or pass --item-emb explicitly."
        )

    vecs = np.load(item_emb).astype(np.float32)
    index = build_hnsw(
        vecs,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        metric="cosine",
    )
    save_index(index, args.out)
    print(f"Saved FAISS HNSW index with {index.ntotal} vectors to {args.out}")


if __name__ == "__main__":
    main()
