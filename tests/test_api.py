import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_config_and_classify(fake_client) -> None:
    app = create_app("config/classification.yaml", client=fake_client)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        config = client.get("/config").json()
        assert "caller_type" in config
        models = client.get("/models").json()
        assert models["backends"]["openjev"]["models"][0]["name"] == "fake-local-model"
        response = client.post("/classify", json={"transcript": "Caller: Hello"})
        assert response.status_code == 200
        assert response.json()["healthcare_related"]["label"] == "yes"


def test_empty_transcript_rejected(fake_client) -> None:
    app = create_app("config/classification.yaml", client=fake_client)
    response = TestClient(app).post("/classify", json={"transcript": ""})
    assert response.status_code == 422


def test_backend_allow_list_exposes_only_native_gemma(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "openjev_gemma")
    monkeypatch.setenv("INFERENCE_BACKENDS", "openjev_gemma")

    app = create_app("config/classification.yaml")

    assert app.state.default_backend == "openjev_gemma"
    assert set(app.state.decision_clients) == {"openjev_gemma"}
    with TestClient(app) as client:
        assert set(client.get("/models").json()["backends"]) == {"openjev_gemma"}
        assert set(client.get("/health").json()["backends"]) == {"openjev_gemma"}


def test_backend_allow_list_rejects_default_not_enabled(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "openjev")
    monkeypatch.setenv("INFERENCE_BACKENDS", "openjev_gemma")

    with pytest.raises(ValueError, match="is not enabled"):
        create_app("config/classification.yaml")
