import json

import httpx

from app.classifier.ollama_client import OllamaClient


def test_lists_only_completion_models_and_returns_label_only_decisions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "embedding-only:latest",
                            "capabilities": ["embedding"],
                            "details": {},
                        },
                        {
                            "name": "classifier:latest",
                            "size": 123,
                            "capabilities": ["completion"],
                            "details": {
                                "family": "test",
                                "parameter_size": "1B",
                                "quantization_level": "Q4_K_M",
                            },
                        },
                    ]
                },
            )
        if request.url.path == "/api/chat":
            payload = json.loads(request.content)
            assert payload["format"]["properties"]["task"]["enum"] == ["a", "b"]
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"task":"a"}'},
                    "total_duration": 2_000_000,
                    "prompt_eval_count": 12,
                },
            )
        return httpx.Response(404)

    client = OllamaClient(
        default_model="classifier:latest",
        transport=httpx.MockTransport(handler),
    )
    assert [model["name"] for model in client.available_models()] == ["classifier:latest"]
    result = client.decide(
        "Caller: hello",
        {"task": {"instructions": "Choose", "criteria": {"a": "A", "b": "B"}}},
        model="classifier:latest",
    )
    assert result.backend == "ollama"
    assert result.answers["task"].label == "a"
    assert result.answers["task"].confidence is None
    assert result.answers["task"].probabilities is None

