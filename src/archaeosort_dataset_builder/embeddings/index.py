import json

import faiss
import numpy as np

from archaeosort_dataset_builder.config.settings import settings


def build_index():

    embedding_dir = settings.outputs / "embeddings"

    vectors_path = embedding_dir / "dinov2_embeddings.npy"
    paths_path = embedding_dir / "dinov2_paths.json"

    vectors = np.load(vectors_path).astype("float32")

    with open(paths_path, "r", encoding="utf8") as f:
        paths = json.load(f)

    if len(vectors) != len(paths):
        raise ValueError(f"Embeddings ({len(vectors)}) and paths ({len(paths)}) do not match.")

    dimension = vectors.shape[1]

    # Los embeddings ya están L2-normalizados.
    # Inner Product equivale a cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    index_path = embedding_dir / "dinov2.faiss"

    faiss.write_index(index, str(index_path))

    print("=" * 60)
    print("FAISS INDEX")
    print("=" * 60)
    print(f"Vectors    : {index.ntotal}")
    print(f"Dimensions : {dimension}")
    print("Metric     : Cosine similarity (normalized IP)")
    print(f"Index      : {index_path.resolve()}")


if __name__ == "__main__":
    build_index()
