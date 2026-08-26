"""Visual-similarity search support for the inference API: loading a
prebuilt FAISS index (from `archaeosort embeddings`/`archaeosort index`) and
searching it with an uploaded image's DINOv2 embedding.

Kept separate from `embeddings/search.py` (the CLI version, which reads a
single query image from disk and relies on the global `settings` singleton)
so the API can load the index once at startup and search it per-request
without touching CLI-only global state. Heavy imports (torch, the DINOv2
model) are deferred into function bodies, matching the rest of this
codebase's lazy-import convention -- `load_similarity_index` itself only
needs faiss/numpy/json, so it's usable without torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SimilarityError(ValueError):
    """Raised when a FAISS index/paths file can't be loaded, or a query
    image can't be embedded."""


@dataclass
class SimilarityIndexBundle:
    """A loaded FAISS index plus the file path for each vector in it, ready
    to search. Loaded once (at API startup) and reused across requests."""

    index: Any
    paths: list[str]


def load_similarity_index(embeddings_dir: Path) -> SimilarityIndexBundle:
    """Load a FAISS index and its paths list, as written by
    `archaeosort embeddings` + `archaeosort index`."""
    import faiss

    embeddings_dir = Path(embeddings_dir).expanduser().resolve()
    index_path = embeddings_dir / "dinov2.faiss"
    paths_path = embeddings_dir / "dinov2_paths.json"

    if not index_path.is_file():
        raise SimilarityError(f"FAISS index not found: {index_path}")
    if not paths_path.is_file():
        raise SimilarityError(f"Paths file not found: {paths_path}")

    try:
        paths = json.loads(paths_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SimilarityError(f"Could not parse {paths_path}: {exc}") from None

    if not isinstance(paths, list) or not paths:
        raise SimilarityError(f"{paths_path} does not contain a non-empty list of paths.")

    index = faiss.read_index(str(index_path))

    if index.ntotal != len(paths):
        raise SimilarityError(
            f"Index/paths mismatch: index has {index.ntotal} vectors, "
            f"paths file lists {len(paths)} -- rebuild with 'archaeosort embeddings' + 'archaeosort index'."
        )

    return SimilarityIndexBundle(index=index, paths=paths)


def embed_image(image: Any) -> Any:
    """Embed a single PIL Image with the same DINOv2 model/normalization
    used to build the index, so query vectors are directly comparable."""
    import torch

    from archaeosort_dataset_builder.embeddings.model import get_model

    runtime = get_model()
    inputs = runtime.processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(runtime.device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = runtime.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy().astype("float32")


def search_similar(bundle: SimilarityIndexBundle, image: Any, k: int = 5) -> list[dict[str, Any]]:
    """Search the loaded index for the k images most visually similar to
    `image`, returning each result's rank, cosine-similarity score, a
    path_id (its position in bundle.paths -- stable for this server's
    lifetime, used by GET /similar-image/{path_id} to fetch the actual
    image without exposing raw filesystem paths to the client) and the
    bare filename for display."""
    import numpy as np

    k = min(k, len(bundle.paths))
    query_vector = np.expand_dims(embed_image(image), axis=0)
    scores, indices = bundle.index.search(query_vector, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
        if idx < 0:
            continue
        path = bundle.paths[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "path_id": int(idx),
                "filename": Path(path).name,
            }
        )
    return results
