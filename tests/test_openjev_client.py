import json

import httpx

from app.classifier.openjev_client import OpenJevClient


def test_openjev_instance_can_override_backend_name_and_method() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["options"]["method"] == "generation"
        return httpx.Response(
            200,
            json={
                "model": "gemma4:e4b",
                "answers": {
                    "task": {
                        "choice": "a",
                        "probabilities": {"a": 0.75, "b": 0.25},
                        "confidence": 0.5,
                        "method": "generation",
                    }
                },
                "usage": {},
            },
        )

    client = OpenJevClient(
        "http://test",
        backend_name="openjev_ollama",
        method_override="generation",
    )
    client._client.close()
    client._client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    try:
        result = client.decide(
            "transcript",
            {"task": {"criteria": {"a": "A", "b": "B"}}},
            method="direct",
        )
    finally:
        client.close()

    assert result.backend == "openjev_ollama"
    assert result.answers["task"].method == "generation"
