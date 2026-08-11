from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from archaeosort_dataset_builder.config.settings import settings


def get_label(path: str) -> str:
    return Path(path).parent.name


def detect_outliers(k: int = 10, percentile: float = 2.0):

    embedding_dir = settings.outputs / "embeddings"

    vectors = np.load(embedding_dir / "dinov2_embeddings.npy").astype("float32")

    with open(
        embedding_dir / "dinov2_paths.json",
        "r",
        encoding="utf8",
    ) as f:
        paths = json.load(f)

    labels = np.array([get_label(path) for path in paths])

    results = []

    for label in sorted(set(labels)):
        class_indices = np.where(labels == label)[0]
        class_vectors = vectors[class_indices]

        index = faiss.IndexFlatIP(class_vectors.shape[1])

        index.add(class_vectors)

        scores, _ = index.search(class_vectors, min(k + 1, len(class_vectors)))

        # Primera coincidencia = propia imagen.
        mean_similarity = scores[:, 1:].mean(axis=1)

        threshold = np.percentile(mean_similarity, percentile)

        for local_idx, similarity in enumerate(mean_similarity):
            if similarity <= threshold:
                global_idx = int(class_indices[local_idx])

                results.append(
                    {
                        "path": paths[global_idx],
                        "label": label,
                        "mean_similarity": round(float(similarity), 4),
                        "threshold": round(float(threshold), 4),
                    }
                )

    results.sort(key=lambda x: x["mean_similarity"])

    report = {
        "images": len(paths),
        "neighbors": k,
        "percentile": percentile,
        "outliers": len(results),
        "cases": results,
    }

    output = settings.reports / "outliers.json"

    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")

    print("=" * 60)
    print("VISUAL OUTLIER DETECTION")
    print("=" * 60)
    print(f"Images      : {len(paths)}")
    print(f"Neighbors   : {k}")
    print(f"Percentile  : {percentile}")
    print(f"Outliers    : {len(results)}")
    print()
    print("Top outliers:")

    for case in results[:20]:
        print(f"{case['mean_similarity']:.4f} | {case['label']:16} | {Path(case['path']).name}")

    print()
    print(f"Report: {output}")


if __name__ == "__main__":
    detect_outliers()
