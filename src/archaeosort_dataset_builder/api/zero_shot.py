"""Open-vocabulary (zero-shot) classification using a CLIP model.

Unlike the ResNet18 baseline in `inference.py` (a closed, trained 2-class
classifier) or the DINOv2/FAISS similarity search in `similarity.py`
(nearest-neighbour lookup against a fixed indexed set), this module
compares an uploaded image against a configurable set of natural-language
prompts using CLIP's shared image/text embedding space -- no training or
indexed dataset required. This makes it the fastest way to extend
ArchaeoSort beyond its original two classes (e.g. bone or wood artefacts)
without collecting and labelling a new dataset, at the cost of typically
lower precision than a purpose-trained classifier.

Note this is a genuinely different model family from DINOv2: DINOv2 is a
vision-only self-supervised model with no text encoder, so it cannot be
compared against text prompts. CLIP is trained specifically to align
image and text embeddings, which is what makes prompt-based zero-shot
classification possible here.

Heavy dependencies (torch, transformers) are imported lazily inside
function bodies, matching the convention in inference.py/similarity.py,
so importing this module (or api/app.py, which imports it) stays cheap
when zero-shot classification isn't configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default open-vocabulary label set: the original two classes plus the
# two extra artefact/natural material pairs the user asked to cover
# (bone, wood) with no dedicated dataset available. Override via the
# ARCHAEOSORT_ZEROSHOT_LABELS env var (see parse_label_prompts) to add or
# change categories without touching code.
DEFAULT_PROMPTS: dict[str, str] = {
    "natural_rock": "a photo of a natural, unworked rock or stone",
    "stone_artefact": "a photo of a stone tool or worked stone artefact",
    "natural_bone": "a photo of a natural, unworked animal bone",
    "bone_artefact": "a photo of a carved or worked bone artefact",
    "natural_wood": "a photo of natural, unworked wood or driftwood",
    "wood_artefact": "a photo of a carved wooden artefact or wood carving",
}

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class ZeroShotError(Exception):
    """Raised when the CLIP model can't be loaded or prompts are invalid."""


@dataclass
class ZeroShotBundle:
    model: Any
    processor: Any
    labels: list[str]
    prompts: list[str]
    device: Any


def parse_label_prompts(raw: str) -> dict[str, str]:
    """Parse an ARCHAEOSORT_ZEROSHOT_LABELS env var value into a
    {label: prompt} dict.

    Format: semicolon-separated "label=prompt text" pairs, e.g.
    "natural_rock=a photo of a natural rock;stone_artefact=a photo of a
    worked stone tool". Raises ZeroShotError on a malformed entry so a
    typo in the env var fails loudly at startup instead of silently
    dropping a category.
    """
    prompts: dict[str, str] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ZeroShotError(
                f"Malformed ARCHAEOSORT_ZEROSHOT_LABELS entry (expected 'label=prompt'): {entry!r}"
            )
        label, _, prompt = entry.partition("=")
        label, prompt = label.strip(), prompt.strip()
        if not label or not prompt:
            raise ZeroShotError(f"Malformed ARCHAEOSORT_ZEROSHOT_LABELS entry: {entry!r}")
        prompts[label] = prompt
    if not prompts:
        raise ZeroShotError("ARCHAEOSORT_ZEROSHOT_LABELS was set but contained no usable entries.")
    return prompts


def load_zero_shot_model(
    model_name: str = DEFAULT_MODEL_NAME,
    prompts: dict[str, str] | None = None,
    device: str | None = None,
) -> ZeroShotBundle:
    """Load a CLIP model/processor and pre-resolve the label list.

    The actual text encoding happens per-request in classify_zero_shot
    (cheap relative to the image forward pass), so this just validates
    the model loads and the prompt set is non-empty.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed
        raise ZeroShotError(f"torch/transformers is required for zero-shot classification: {exc}") from None

    resolved_prompts = prompts or DEFAULT_PROMPTS
    if not resolved_prompts:
        raise ZeroShotError("No zero-shot prompts configured.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = CLIPModel.from_pretrained(model_name).to(resolved_device)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as exc:  # surface any download/load failure as ZeroShotError
        raise ZeroShotError(f"Could not load CLIP model {model_name!r}: {exc}") from exc

    model.eval()
    labels = list(resolved_prompts.keys())
    texts = list(resolved_prompts.values())
    return ZeroShotBundle(model=model, processor=processor, labels=labels, prompts=texts, device=resolved_device)


def classify_zero_shot(bundle: ZeroShotBundle, image: Any) -> dict[str, Any]:
    """Score `image` against the bundle's text prompts and return a dict
    shaped like inference.predict_image's output (predicted_class,
    confidence, probabilities), so both can share a response model.
    """
    import torch

    inputs = bundle.processor(text=bundle.prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(bundle.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bundle.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

    probabilities = dict(zip(bundle.labels, probs, strict=True))
    predicted_class = max(probabilities, key=probabilities.get)
    return {
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "probabilities": probabilities,
    }
