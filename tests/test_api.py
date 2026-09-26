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
        response = client.post("/classify", json={"transcript": "Caller: Hello"})
        assert response.status_code == 200
        assert response.json()["healthcare_related"]["label"] == "yes"


def test_empty_transcript_rejected(fake_client) -> None:
    app = create_app("config/classification.yaml", client=fake_client)
    response = TestClient(app).post("/classify", json={"transcript": ""})
    assert response.status_code == 422

