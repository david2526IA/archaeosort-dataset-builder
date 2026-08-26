"""Inference support for the Fase G ResNet18 baseline classifier: loading a
trained checkpoint and running predictions on a single image.

Kept separate from `classifier/baseline.py` (training only) and from
`api/app.py` (the FastAPI HTTP layer) so each piece can be used/tested on
its own. Heavy imports (torch, torchvision) are deferred into function
bodies, matching the rest of this codebase's lazy-import convention --
`resolve_classes`/`_read_classes_from_evaluation` are import-light and
unit-testable without torch installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InferenceError(ValueError):
    """Raised when a model checkpoint/classes cannot be loaded, or an image
    cannot be classified."""


@dataclass
class ModelBundle:
    """Everything needed to run inference, loaded once (at API startup, or
    by a script) and reused across requests."""

    model: Any
    classes: list[str]
    device: Any
    image_size: int = 224


def _read_classes_from_evaluation(model_path: Path) -> list[str] | None:
    """`archaeosort train-baseline` writes `evaluation.json` next to
    `best_model.pt` with a `classes` field, in the exact index order the
    model's output layer uses. Returns None if that file isn't there or
    doesn't have a usable `classes` list, so callers can fall back to
    requiring an explicit class list instead of guessing wrong."""
    evaluation_path = Path(model_path).parent / "evaluation.json"
    if not evaluation_path.is_file():
        return None

    try:
        data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    classes = data.get("classes")
    if not isinstance(classes, list) or not classes:
        return None
    return [str(name) for name in classes]


def resolve_classes(model_path: Path, classes: list[str] | None) -> list[str]:
    """Resolve the class list for a checkpoint: use the explicit `classes`
    argument if given, otherwise try to read it from a sibling
    evaluation.json. Raises InferenceError if neither source is available --
    silently guessing class names would misalign predictions with labels."""
    if classes:
        return list(classes)

    inferred = _read_classes_from_evaluation(model_path)
    if inferred:
        return inferred

    raise InferenceError(
        f"Could not determine class names for {model_path}: no explicit classes "
        "were given, and no evaluation.json with a 'classes' field was found "
        "alongside the checkpoint."
    )


def load_model_bundle(
    model_path: Path,
    classes: list[str] | None = None,
    image_size: int = 224,
    device: str | None = None,
) -> ModelBundle:
    """Load a trained checkpoint (as written by `train_baseline`) into a
    ready-to-use ModelBundle, in eval mode on the resolved device."""
    import torch

    from archaeosort_dataset_builder.classifier.baseline import build_model

    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise InferenceError(f"Model checkpoint not found: {resolved_path}")

    resolved_classes = resolve_classes(resolved_path, classes)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(len(resolved_classes), freeze_backbone=False)
    state_dict = torch.load(resolved_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    return ModelBundle(
        model=model,
        classes=resolved_classes,
        device=resolved_device,
        image_size=image_size,
    )


def predict_image(bundle: ModelBundle, image: Any) -> dict[str, Any]:
    """Run inference on a single PIL Image, returning the predicted class,
    its confidence, and the full per-class probability distribution."""
    import torch

    from archaeosort_dataset_builder.classifier.baseline import _build_transforms

    transform = _build_transforms(bundle.image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    probability_values = probabilities.tolist()
    best_index = int(probabilities.argmax())

    return {
        "predicted_class": bundle.classes[best_index],
        "confidence": probability_values[best_index],
        "probabilities": dict(zip(bundle.classes, probability_values, strict=True)),
    }
