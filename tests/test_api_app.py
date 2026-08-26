"""HTTP-contract tests for api/app.py, using FastAPI's TestClient with a
monkeypatched fake ModelBundle so no torch/real checkpoint is needed.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from archaeosort_dataset_builder.api import app as app_module
from archaeosort_dataset_builder.api.inference import ModelBundle


def _fake_bundle(*_args, **_kwargs):
    return ModelBundle(model=object(), classes=["natural_rock", "stone_artefact"], device="cpu")


def _fake_predict_image(_bundle, _image):
    return {
        "predicted_class": "stone_artefact",
        "confidence": 0.87,
        "probabilities": {"natural_rock": 0.13, "stone_artefact": 0.87},
    }


def _sample_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client_with_model(monkeypatch):
    monkeypatch.setenv(app_module.MODEL_PATH_ENV, "/fake/best_model.pt")
    monkeypatch.setattr(app_module, "load_model_bundle", _fake_bundle)
    monkeypatch.setattr(app_module, "predict_image", _fake_predict_image)
    with TestClient(app_module.app) as client:
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.delenv(app_module.MODEL_PATH_ENV, raising=False)
    with TestClient(app_module.app) as client:
        yield client


def test_health_reports_loaded_model(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == ["natural_rock", "stone_artefact"]


def test_health_reports_not_loaded_without_model_path(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False


def test_predict_returns_prediction(client_with_model):
    response = client_with_model.post(
        "/predict", files={"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "stone_artefact"
    assert body["confidence"] == pytest.approx(0.87)
    assert body["probabilities"]["stone_artefact"] == pytest.approx(0.87)


def test_predict_503_when_model_not_loaded(client_without_model):
    response = client_without_model.post(
        "/predict", files={"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503


def test_predict_400_on_invalid_image(client_with_model):
    response = client_with_model.post(
        "/predict", files={"file": ("not_an_image.txt", b"this is not an image", "text/plain")}
    )
    assert response.status_code == 400
