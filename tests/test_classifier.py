from typing import Any

from app.classifier.decision_client import DecisionAnswer, DecisionResult
from app.classifier.generic_classifier import GenericDecisionClassifier
from app.config.loader import ConfigStore


class FakeOllamaClient:
    backend_name = "ollama"

    def __init__(self) -> None:
        self.calls = 0

    def health(self) -> dict[str, Any]:
        return {"ready": True}

    def available_models(self) -> list[dict[str, Any]]:
        return [{"name": "fake-ollama", "capabilities": ["completion"]}]

    def decide(self, state, questions, method="direct", model=None) -> DecisionResult:
        self.calls += 1
        answers = {
            name: DecisionAnswer(
                label=list(question["criteria"])[-1],
                probabilities=None,
                confidence=None,
                method="structured_generation",
                inference_ms=25.0,
            )
            for name, question in questions.items()
        }
        return DecisionResult(model or "fake-ollama", "ollama", answers, 25, 20)


class MarginalConfidenceOpenJevClient:
    backend_name = "openjev"

    def decide(self, state, questions, method="direct", model=None) -> DecisionResult:
        answers = {}
        for name, question in questions.items():
            labels = list(question["criteria"])
            probabilities = {label: 0.38 / (len(labels) - 1) for label in labels}
            probabilities[labels[0]] = 0.62
            answers[name] = DecisionAnswer(
                label=labels[0],
                probabilities=probabilities,
                confidence=0.62,
                method="direct",
                inference_ms=5.0,
            )
        return DecisionResult("fake-native", "openjev", answers, 5, 10)


def test_classifier_returns_every_dynamic_task(fake_client) -> None:
    classifier = GenericDecisionClassifier(ConfigStore("config/classification.yaml"), fake_client)
    result = classifier.classify("Caller: This is a test transcript.")
    assert result["healthcare_related"]["label"] == "yes"
    assert set(result["caller_type"]["probabilities"]) == set(
        classifier.config_store.public_view()["caller_type"]
    )
    assert result["model"]["probability_source"] == "token_logprobs"
    assert result["transcript_processing"]["truncated"] is False
    assert result["model"]["name"] == "fake-local-model"


def test_optional_chunk_details(fake_client) -> None:
    classifier = GenericDecisionClassifier(
        ConfigStore("config/classification.yaml"), fake_client, return_chunk_details=True
    )
    result = classifier.classify("Caller: Test.")
    assert result["debug"]["chunks"][0]["chunk"] == 1


def test_hybrid_returns_confident_native_result_without_calling_ollama(fake_client) -> None:
    ollama = FakeOllamaClient()
    classifier = GenericDecisionClassifier(
        ConfigStore("config/classification.yaml"),
        {"openjev": fake_client, "ollama": ollama},
    )

    result = classifier.classify("Caller: Test.", backend="hybrid", model="fake-ollama")

    assert result["model"]["backend"] == "hybrid"
    assert result["model"]["routing"]["selected_backend"] == "openjev"
    assert result["model"]["routing"]["fallback_used"] is False
    assert ollama.calls == 0


def test_hybrid_falls_back_to_selected_ollama_model() -> None:
    ollama = FakeOllamaClient()
    classifier = GenericDecisionClassifier(
        ConfigStore("config/classification.yaml"),
        {"openjev": MarginalConfidenceOpenJevClient(), "ollama": ollama},
    )

    result = classifier.classify("Caller: Test.", backend="hybrid", model="fake-ollama")

    assert result["model"]["name"] == "fake-ollama"
    assert result["model"]["backend"] == "hybrid"
    assert result["model"]["routing"]["selected_backend"] == "ollama"
    assert result["model"]["routing"]["fallback_used"] is True
    assert {item["task"] for item in result["model"]["routing"]["fallback_tasks"]} == {
        "healthcare_related",
        "caller_type",
        "intent",
    }
    assert all(
        item["reason"] == "low_confidence"
        for item in result["model"]["routing"]["fallback_tasks"]
    )
    assert all(
        item["confidence"] == 0.62
        for item in result["model"]["routing"]["fallback_tasks"]
    )
    assert ollama.calls == 1
