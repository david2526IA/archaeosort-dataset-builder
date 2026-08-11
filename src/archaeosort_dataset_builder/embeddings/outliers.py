from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from archaeosort_dataset_builder.config.settings import settings


def get_label(path: str) -> str:
    return Path(path).parent.name


def detect_outliers(k: int = 10, percentile: float = 2.0):
    if k < 1:
        raise ValueError("k must be greater than or equal to 1.")

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    embedding_dir = settings.outputs / "embeddings"
    vectors_path = embedding_dir / "dinov2_embeddings.npy"
    paths_path = embedding_dir / "dinov2_paths.json"

    if not vectors_path.exists():
        raise FileNotFoundError(vectors_path)

    if not paths_path.exists():
        raise FileNotFoundError(paths_path)

    vectors = np.load(vectors_path).astype("float32")

    with paths_path.open("r", encoding="utf8") as file:
        paths = json.load(file)

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(vectors) != len(paths):
        raise ValueError("Number of embeddings does not match number of image paths.")

    labels = np.array([get_label(path) for path in paths])

    results = []
    skipped_classes = []

    for label in sorted(set(labels)):
        class_indices = np.where(labels == label)[0]
        class_vectors = vectors[class_indices]

        if len(class_vectors) < 2:
            skipped_classes.append(label)
            continue

        index = faiss.IndexFlatIP(class_vectors.shape[1])
        index.add(class_vectors)

        neighbors = min(k + 1, len(class_vectors))
        scores, _ = index.search(class_vectors, neighbors)

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

    results.sort(key=lambda item: item["mean_similarity"])

    report = {
        "images": len(paths),
        "neighbors": k,
        "percentile": percentile,
        "outliers": len(results),
        "skipped_classes": skipped_classes,
        "cases": results,
    }

    settings.reports.mkdir(parents=True, exist_ok=True)

    output = settings.reports / "outliers.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    print("=" * 60)
    print("VISUAL OUTLIER DETECTION")
    print("=" * 60)
    print(f"Images      : {len(paths)}")
    print(f"Neighbors   : {k}")
    print(f"Percentile  : {percentile}")
    print(f"Outliers    : {len(results)}")

    if skipped_classes:
        print(f"Skipped     : {len(skipped_classes)}")

    print()
    print("Top outliers:")

    for case in results[:20]:
        print(f"{case['mean_similarity']:.4f} | {case['label']:16} | {Path(case['path']).name}")

    print()
    print(f"Report: {output}")

    return report


if __name__ == "__main__":
    detect_outliers()
