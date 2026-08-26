"""Few-shot, image-to-image classification via class prototypes in DINOv2
embedding space -- an alternative to zero_shot.py's CLIP text-prompt
approach.

Why this exists alongside zero_shot.py: the first live test of the CLIP
zero-shot classifier misclassified a genuine natural_rock test image as
stone_artefact. Distinguishing "natural rock" from "worked stone" (or
"natural bone" from "carved bone") from a text description alone asks a
general-purpose vision-language model to capture a fine, domain-specific
visual distinction it was never trained to make. DINOv2 is already
proven on this exact domain -- it powers the /similar visual-similarity
search with strong results on this project's own data -- so comparing a
query image against a handful of REFERENCE PHOTOS per category (a
"prototype": the mean of that category's embeddings), in DINOv2's image
embedding space, is a stronger fit for this specific problem than
text-prompt zero-shot, at the cost of needing a small reference image
set per category (see reference_dataset.py) instead of being purely
zero-shot.

Heavy imports (numpy, and via api.similarity.embed_image, torch) are
deferred into function bodies, matching the rest of this codebase's
lazy-import convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PrototypeError(ValueError):
    """Raised when a prototype file can't be built, saved, or loaded."""


@dataclass
class PrototypeBundle:
    labels: list[str]
    vectors: Any  # np.ndarray, shape (n_classes, embedding_dim), L2-normalized rows


def build_prototypes(reference_images: dict[str, list[Any]]) -> PrototypeBundle:
    """reference_images: {label: [PIL.Image, ...]}. Embeds every reference
    image with the shared DINOv2 model (api.similarity.embed_image, the
    same one used to build the FAISS similarity index) and averages each
    label's embeddings into a single L2-normalized prototype vector."""
    import numpy as np

    from archaeosort_dataset_builder.api.similarity import embed_image

    if not reference_images:
        raise PrototypeError("No reference images provided.")

    labels = []
    vectors = []
    for label, images in reference_images.items():
        if not images:
            raise PrototypeError(f"No reference images provided for label {label!r}.")
        embeddings = np.stack([embed_image(image) for image in images])
        prototype = embeddings.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm == 0:
            raise PrototypeError(f"Degenerate (all-zero) prototype for label {label!r}.")
        labels.append(label)
        vectors.append(prototype / norm)

    return PrototypeBundle(labels=labels, vectors=np.stack(vectors).astype("float32"))


def save_prototypes(bundle: PrototypeBundle, path: Path) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=np.array(bundle.labels), vectors=bundle.vectors)


def load_prototypes(path: Path) -> PrototypeBundle:
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise PrototypeError(f"Prototype file not found: {path}")
    try:
        data = np.load(path, allow_pickle=False)
        labels = [str(label) for label in data["labels"]]
        vectors = data["vectors"].astype("float32")
    except PrototypeError:
        raise
    except Exception as exc:  # surface any corrupt/unreadable file as PrototypeError
        raise PrototypeError(f"Could not load prototype file {path}: {exc}") from exc

    if len(labels) != vectors.shape[0]:
        raise PrototypeError(f"Prototype file {path} has mismatched labels/vectors.")
    return PrototypeBundle(labels=labels, vectors=vectors)


def classify_by_prototype(bundle: PrototypeBundle, image: Any, temperature: float = 20.0) -> dict[str, Any]:
    """Classify `image` by cosine similarity (both sides L2-normalized, so
    this is a plain dot product) to each stored class prototype.
    `temperature` scales the similarities before softmax -- raw cosine
    similarities cluster in a narrow range, so a temperature around 15-25
    is needed to get a confidence spread comparable to a real classifier
    (this is the same role CLIP's learned logit_scale plays, just fixed
    here instead of learned). Returns a dict shaped like
    inference.predict_image's output so both share a response model.
    Confidences are a softmax over scaled similarities, not a calibrated
    probability -- same caveat as the CLIP zero-shot path."""
    import numpy as np

    from archaeosort_dataset_builder.api.similarity import embed_image

    query = embed_image(image)
    similarities = bundle.vectors @ query  # cosine similarity, both sides L2-normalized
    scaled = similarities * temperature

    exp = np.exp(scaled - scaled.max())
    probs = exp / exp.sum()

    probabilities = dict(zip(bundle.labels, probs.tolist(), strict=True))
    predicted_class = max(probabilities, key=probabilities.get)
    return {
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "probabilities": probabilities,
    }
