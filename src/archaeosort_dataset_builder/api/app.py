"""FastAPI HTTP layer for the Fase G baseline classifier.

Thin wrapper around `api/inference.py`: loads the model once at startup
(via a lifespan context manager, reading configuration from environment
variables so `cli.py`'s `serve` command can set them before handing off to
uvicorn) and exposes `/health` and `/predict`.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from archaeosort_dataset_builder.api.inference import (
    InferenceError,
    load_model_bundle,
    predict_image,
)

MODEL_PATH_ENV = "ARCHAEOSORT_MODEL_PATH"
MODEL_CLASSES_ENV = "ARCHAEOSORT_MODEL_CLASSES"
MODEL_DEVICE_ENV = "ARCHAEOSORT_MODEL_DEVICE"

_state: dict[str, Any] = {"bundle": None, "error": None}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str] | None = None
    device: str | None = None


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


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
    yield
    _state["bundle"] = None


app = FastAPI(title="ArchaeoSort Inference API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _state.get("bundle")
    if bundle is None:
        return HealthResponse(status="error" if _state.get("error") else "not_loaded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        classes=bundle.classes,
        device=str(bundle.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    bundle = _state.get("bundle")
    if bundle is None:
        detail = _state.get("error") or "Model is not loaded."
        raise HTTPException(status_code=503, detail=detail)

    from PIL import Image, UnidentifiedImageError

    contents = await file.read()
    try:
        import io

        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from None

    result = predict_image(bundle, image)
    return PredictionResponse(**result)
