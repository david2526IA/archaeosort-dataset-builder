"""HTTP-contract tests for api/app.py, using FastAPI's TestClient with
monkeypatched fake ModelBundle/SimilarityIndexBundle so no torch, real
checkpoint, or real FAISS index is needed.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from archaeosort_dataset_builder.api import app as app_module
from archaeosort_dataset_builder.api.inference import ModelBundle
from archaeosort_dataset_builder.api.similarity import SimilarityIndexBundle


def _fake_bundle(*_args, **_kwargs):
    return ModelBundle(model=object(), classes=["natural_rock", "stone_artefact"], device="cpu")


def _fake_predict_image(_bundle, _image):
    return {
        "predicted_class": "stone_artefact",
        "confidence": 0.87,
        "probabilities": {"natural_rock": 0.13, "stone_artefact": 0.87},
    }


def _fake_similarity_bundle(embeddings_dir):
    return SimilarityIndexBundle(index=object(), paths=["/data/a.jpg", "/data/b.jpg"])


def _fake_search_similar(_bundle, _image, k=5):
    return [
        {"rank": 1, "score": 0.95, "path_id": 0, "filename": "a.jpg"},
        {"rank": 2, "score": 0.80, "path_id": 1, "filename": "b.jpg"},
    ][:k]


def _sample_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _clear_env(monkeypatch):
    monkeypatch.delenv(app_module.MODEL_PATH_ENV, raising=False)
    monkeypatch.delenv(app_module.EMBEDDINGS_DIR_ENV, raising=False)


@pytest.fixture
def client_with_model(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(app_module.MODEL_PATH_ENV, "/fake/best_model.pt")
    monkeypatch.setattr(app_module, "load_model_bundle", _fake_bundle)
    monkeypatch.setattr(app_module, "predict_image", _fake_predict_image)
    with TestClient(app_module.app) as client:
        yield client


@pytest.fixture
def client_without_model(monkeypatch):
    _clear_env(monkeypatch)
    with TestClient(app_module.app) as client:
        yield client


@pytest.fixture
def client_with_similarity(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(app_module.EMBEDDINGS_DIR_ENV, "/fake/embeddings")
    monkeypatch.setattr(app_module, "load_similarity_index", _fake_similarity_bundle)
    monkeypatch.setattr(app_module, "search_similar", _fake_search_similar)
    with TestClient(app_module.app) as client:
        yield client


def test_health_reports_loaded_model(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == ["natural_rock", "stone_artefact"]
    assert body["similarity_loaded"] is False


def test_health_reports_not_loaded_without_model_path(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False
    assert body["similarity_loaded"] is False


def test_health_reports_loaded_similarity_index(client_with_similarity):
    response = client_with_similarity.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["similarity_loaded"] is True
    assert body["similarity_size"] == 2
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


def test_similar_returns_results(client_with_similarity):
    response = client_with_similarity.post(
        "/similar", files={"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "a.jpg"
    assert body["results"][0]["path_id"] == 0


def test_similar_respects_k(client_with_similarity):
    response = client_with_similarity.post(
        "/similar", params={"k": 1}, files={"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_similar_503_when_index_not_loaded(client_without_model):
    response = client_without_model.post(
        "/similar", files={"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 503


def test_similar_400_on_invalid_image(client_with_similarity):
    response = client_with_similarity.post(
        "/similar", files={"file": ("not_an_image.txt", b"nope", "text/plain")}
    )
    assert response.status_code == 400


def test_similar_image_404_out_of_range(client_with_similarity):
    response = client_with_similarity.get("/similar-image/99")
    assert response.status_code == 404


def test_similar_image_404_when_file_missing_on_disk(client_with_similarity):
    # _fake_similarity_bundle's paths ("/data/a.jpg") don't exist on this
    # machine, so a valid path_id still 404s rather than crashing.
    response = client_with_similarity.get("/similar-image/0")
    assert response.status_code == 404


def test_similar_image_503_when_index_not_loaded(client_without_model):
    response = client_without_model.get("/similar-image/0")
    assert response.status_code == 503


def test_cors_allows_default_frontend_origin(client_without_model):
    response = client_without_model.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_unlisted_origin(client_without_model):
    response = client_without_model.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers
