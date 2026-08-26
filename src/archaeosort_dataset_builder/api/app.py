"""FastAPI HTTP layer for the Fase G baseline classifier and the DINOv2/FAISS
visual-similarity search (Fase I backend support).

Thin wrapper around `api/inference.py` and `api/similarity.py`: loads the
classifier checkpoint and/or the FAISS index once at startup (via a
lifespan context manager, reading configuration from environment variables
so `cli.py`'s `serve` command can set them before handing off to uvicorn)
and exposes `/health`, `/predict`, `/similar` and `/similar-image/{path_id}`.
The two features are independent -- a server can run with only a
classifier, only a similarity index, or both; `/health` reports each
separately.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from archaeosort_dataset_builder.api.inference import (
    InferenceError,
    load_model_bundle,
    predict_image,
)
from archaeosort_dataset_builder.api.similarity import (
    SimilarityError,
    load_similarity_index,
    search_similar,
)
from archaeosort_dataset_builder.api.zero_shot import (
    ZeroShotError,
    classify_zero_shot,
    load_zero_shot_model,
    parse_label_prompts,
)
from archaeosort_dataset_builder.prototypes.prototypes import (
    PrototypeError,
    classify_by_prototype,
    load_prototypes,
)

MODEL_PATH_ENV = "ARCHAEOSORT_MODEL_PATH"
MODEL_CLASSES_ENV = "ARCHAEOSORT_MODEL_CLASSES"
MODEL_DEVICE_ENV = "ARCHAEOSORT_MODEL_DEVICE"
EMBEDDINGS_DIR_ENV = "ARCHAEOSORT_EMBEDDINGS_DIR"
CORS_ORIGINS_ENV = "ARCHAEOSORT_CORS_ORIGINS"
ZEROSHOT_MODEL_ENV = "ARCHAEOSORT_ZEROSHOT_MODEL"
ZEROSHOT_LABELS_ENV = "ARCHAEOSORT_ZEROSHOT_LABELS"
ZEROSHOT_DEVICE_ENV = "ARCHAEOSORT_ZEROSHOT_DEVICE"
PROTOTYPES_PATH_ENV = "ARCHAEOSORT_PROTOTYPES_PATH"

# The frontend (Fase I, frontend/) runs as a separate Next.js dev server on
# a different port, so the browser treats it as a different origin and
# blocks fetch() calls to this API unless it's explicitly allowed here.
# Covers both hostnames a dev server might report ("localhost" vs
# "127.0.0.1" are different origins to a browser even though they resolve
# to the same machine). Override/extend via ARCHAEOSORT_CORS_ORIGINS
# (comma-separated) for a non-default frontend port or a deployed origin.
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

_state: dict[str, Any] = {
    "bundle": None,
    "error": None,
    "similarity": None,
    "similarity_error": None,
    "zero_shot": None,
    "zero_shot_error": None,
    "prototypes": None,
    "prototypes_error": None,
}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str] | None = None
    device: str | None = None
    similarity_loaded: bool = False
    similarity_size: int | None = None
    zero_shot_loaded: bool = False
    zero_shot_labels: list[str] | None = None
    prototypes_loaded: bool = False
    prototypes_labels: list[str] | None = None


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class SimilarityResult(BaseModel):
    rank: int
    score: float
    path_id: int
    filename: str


class SimilarityResponse(BaseModel):
    results: list[SimilarityResult]


async def _read_uploaded_image(file: UploadFile) -> Any:
    """Decode an uploaded file into a PIL Image, raising a 400 HTTPException
    (not a raw exception) if it isn't a readable image."""
    from PIL import Image, UnidentifiedImageError

    contents = await file.read()
    try:
        import io

        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from None
    return image


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get(MODEL_PATH_ENV)
    if model_path:
        classes_env = os.environ.get(MODEL_CLASSES_ENV)
        classes = [c.strip() for c in classes_env.split(",") if c.strip()] if classes_env else None
        device = os.environ.get(MODEL_DEVICE_ENV) or None
        try:
            _state["bundle"] = load_model_bundle(model_path, classes=classes, device=device)
            _state["error"] = None
        except InferenceError as exc:
            # Don't crash the whole process on a bad checkpoint/config -- /health
            # will report the failure instead, which is easier to diagnose than
            # uvicorn exiting immediately with a stack trace.
            _state["bundle"] = None
            _state["error"] = str(exc)
    else:
        _state["bundle"] = None
        _state["error"] = f"{MODEL_PATH_ENV} is not set"

    embeddings_dir = os.environ.get(EMBEDDINGS_DIR_ENV)
    if embeddings_dir:
        try:
            _state["similarity"] = load_similarity_index(embeddings_dir)
            _state["similarity_error"] = None
        except SimilarityError as exc:
            _state["similarity"] = None
            _state["similarity_error"] = str(exc)
    else:
        _state["similarity"] = None
        _state["similarity_error"] = f"{EMBEDDINGS_DIR_ENV} is not set"

    zeroshot_model = os.environ.get(ZEROSHOT_MODEL_ENV)
    if zeroshot_model:
        labels_env = os.environ.get(ZEROSHOT_LABELS_ENV)
        try:
            prompts = parse_label_prompts(labels_env) if labels_env else None
            _state["zero_shot"] = load_zero_shot_model(
                zeroshot_model,
                prompts=prompts,
                device=os.environ.get(ZEROSHOT_DEVICE_ENV) or None,
            )
            _state["zero_shot_error"] = None
        except ZeroShotError as exc:
            _state["zero_shot"] = None
            _state["zero_shot_error"] = str(exc)
    else:
        _state["zero_shot"] = None
        _state["zero_shot_error"] = f"{ZEROSHOT_MODEL_ENV} is not set"

    prototypes_path = os.environ.get(PROTOTYPES_PATH_ENV)
    if prototypes_path:
        try:
            _state["prototypes"] = load_prototypes(Path(prototypes_path))
            _state["prototypes_error"] = None
        except PrototypeError as exc:
            _state["prototypes"] = None
            _state["prototypes_error"] = str(exc)
    else:
        _state["prototypes"] = None
        _state["prototypes_error"] = f"{PROTOTYPES_PATH_ENV} is not set"

    yield
    _state["bundle"] = None
    _state["similarity"] = None
    _state["zero_shot"] = None
    _state["prototypes"] = None


app = FastAPI(title="ArchaeoSort Inference API", lifespan=lifespan)

_cors_origins_env = os.environ.get(CORS_ORIGINS_ENV)
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
        if _cors_origins_env
        else DEFAULT_CORS_ORIGINS
    ),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _state.get("bundle")
    similarity = _state.get("similarity")

    if bundle is None:
        status, model_loaded = ("error" if _state.get("error") else "not_loaded", False)
        classes = device = None
    else:
        status, model_loaded = "ok", True
        classes, device = bundle.classes, str(bundle.device)

    zero_shot = _state.get("zero_shot")
    prototypes = _state.get("prototypes")

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        classes=classes,
        device=device,
        similarity_loaded=similarity is not None,
        similarity_size=len(similarity.paths) if similarity is not None else None,
        zero_shot_loaded=zero_shot is not None,
        zero_shot_labels=zero_shot.labels if zero_shot is not None else None,
        prototypes_loaded=prototypes is not None,
        prototypes_labels=prototypes.labels if prototypes is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    bundle = _state.get("bundle")
    if bundle is None:
        detail = _state.get("error") or "Model is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    image = await _read_uploaded_image(file)
    result = predict_image(bundle, image)
    return PredictionResponse(**result)


@app.post("/similar", response_model=SimilarityResponse)
async def similar(file: UploadFile, k: int = 5) -> SimilarityResponse:
    similarity = _state.get("similarity")
    if similarity is None:
        detail = _state.get("similarity_error") or "Similarity index is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    if k < 1:
        raise HTTPException(status_code=400, detail="k must be at least 1.")

    image = await _read_uploaded_image(file)
    results = search_similar(similarity, image, k=k)
    return SimilarityResponse(results=[SimilarityResult(**r) for r in results])


@app.get("/similar-image/{path_id}")
def similar_image(path_id: int) -> FileResponse:
    similarity = _state.get("similarity")
    if similarity is None:
        detail = _state.get("similarity_error") or "Similarity index is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    if path_id < 0 or path_id >= len(similarity.paths):
        raise HTTPException(status_code=404, detail="No image with that path_id.")

    image_path = Path(similarity.paths[path_id])
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Indexed image no longer exists on disk.")

    return FileResponse(image_path)


@app.post("/classify-open", response_model=PredictionResponse)
async def classify_open(file: UploadFile) -> PredictionResponse:
    """Open-vocabulary classification via CLIP text prompts (see
    api/zero_shot.py). Independent of /predict's closed ResNet18
    classifier -- useful for categories the trained model was never
    shown (e.g. bone or wood artefacts) without collecting new training
    data, at the cost of typically lower precision."""
    zero_shot = _state.get("zero_shot")
    if zero_shot is None:
        detail = _state.get("zero_shot_error") or "Zero-shot classifier is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    image = await _read_uploaded_image(file)
    result = classify_zero_shot(zero_shot, image)
    return PredictionResponse(**result)


@app.post("/classify-prototype", response_model=PredictionResponse)
async def classify_prototype(file: UploadFile) -> PredictionResponse:
    """Open-vocabulary classification via DINOv2 image-to-image nearest
    class-prototype (see prototypes/prototypes.py, built with
    'archaeosort build-prototypes'). Compares the uploaded image against
    a handful of real reference photos per category rather than CLIP text
    prompts -- a stronger fit for this project's fine-grained
    natural-vs-worked distinctions, since it reuses the same DINOv2 model
    already proven on this data by /similar."""
    prototypes = _state.get("prototypes")
    if prototypes is None:
        detail = _state.get("prototypes_error") or "Prototype classifier is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    image = await _read_uploaded_image(file)
    result = classify_by_prototype(prototypes, image)
    return PredictionResponse(**result)
