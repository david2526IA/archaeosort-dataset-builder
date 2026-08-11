from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from archaeosort_dataset_builder.config.settings import settings


def get_label(path: str) -> str:
    return Path(path).parent.name


def semantic_audit(k: int = 10):

    embedding_dir = settings.outputs / "embeddings"

    vectors = np.load(embedding_dir / "dinov2_embeddings.npy").astype("float32")

    with open(
        embedding_dir / "dinov2_paths.json",
        "r",
        encoding="utf8",
    ) as f:
        paths = json.load(f)

    index = faiss.read_index(str(embedding_dir / "dinov2.faiss"))

    scores, indices = index.search(vectors, k + 1)

    cases = []

    summary = {
        "consistent": 0,
        "review": 0,
        "high_risk": 0,
    }

    for i, path in enumerate(paths):
        current_label = get_label(path)

        neighbors = []

        for position, idx in enumerate(indices[i]):
            idx = int(idx)

            if idx == i:
                continue

            neighbors.append(
                {
                    "index": idx,
                    "path": paths[idx],
                    "label": get_label(paths[idx]),
                    "similarity": round(float(scores[i][position]), 4),
                }
            )

            if len(neighbors) == k:
                break

        labels = [n["label"] for n in neighbors]

        votes = Counter(labels)

        predicted_label, vote_count = votes.most_common(1)[0]

        agreement = vote_count / len(neighbors)

        if predicted_label == current_label:
            status = "consistent"

        elif agreement >= 0.80:
            status = "high_risk"

        else:
            status = "review"

        summary[status] += 1

        if status != "consistent":
            cases.append(
                {
                    "path": path,
                    "current_label": current_label,
                    "neighbor_prediction": predicted_label,
                    "agreement": round(agreement, 4),
                    "status": status,
                    "neighbors": neighbors,
                }
            )

    cases.sort(key=lambda x: (x["status"] != "high_risk", -x["agreement"]))

    report = {
        "images": len(paths),
        "neighbors": k,
        "summary": summary,
        "cases": cases,
    }

    output = settings.reports / "semantic_audit.json"

    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    print("=" * 60)
    print("SEMANTIC DATASET AUDIT")
    print("=" * 60)

    print(f"Images      : {len(paths)}")
    print(f"Consistent  : {summary['consistent']}")
    print(f"Review      : {summary['review']}")
    print(f"High risk   : {summary['high_risk']}")

    print()

    for case in cases[:20]:
        print(
            f"{case['status'].upper():10} | "
            f"{case['agreement']:.0%} | "
            f"{case['current_label']} -> "
            f"{case['neighbor_prediction']} | "
            f"{Path(case['path']).name}"
        )

    print()
    print(f"Report: {output}")


if __name__ == "__main__":
    semantic_audit()
