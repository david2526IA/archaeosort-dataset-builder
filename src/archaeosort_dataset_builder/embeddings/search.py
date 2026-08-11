from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from archaeosort_dataset_builder.config.settings import settings
from archaeosort_dataset_builder.embeddings.encoder import encode_image


def search_similar(query_image: Path, k: int = 5):

    embedding_dir = settings.outputs / "embeddings"

    index = faiss.read_index(str(embedding_dir / "dinov2.faiss"))

    with open(
        embedding_dir / "dinov2_paths.json",
        "r",
        encoding="utf8",
    ) as f:
        paths = json.load(f)

    query_vector = encode_image(query_image)

    query_vector = np.expand_dims(
        query_vector.astype("float32"),
        axis=0,
    )

    scores, indices = index.search(query_vector, k)

    print("=" * 60)
    print("VISUAL SIMILARITY SEARCH")
    print("=" * 60)

    print(f"Query: {query_image}")
    print()

    results = []

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        result = {
            "rank": rank,
            "score": float(score),
            "path": paths[int(idx)],
        }

        results.append(result)

        print(f"{rank:02d} | {score:.4f} | {paths[int(idx)]}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=Path)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    search_similar(
        args.image,
        args.k,
    )
