from __future__ import annotations

from typing import Any

import pytest

from app.classifier.decision_client import DecisionAnswer, DecisionResult


class FakeOpenJevClient:
    backend_name = "openjev"

    def health(self) -> dict[str, Any]:
        return {"ready": True, "model": "fake-local-model", "llama_reachable": True}

    def available_models(self) -> list[dict[str, Any]]:
        return [{"name": "fake-local-model", "capabilities": ["classification"]}]

    def decide(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
        method: str = "direct",
        model: str | None = None,
    ) -> DecisionResult:
        answers: dict[str, DecisionAnswer] = {}
        for task_name, question in questions.items():
            labels = list(question["criteria"])
            selected = labels[0]
            probabilities = {label: 0.1 / (len(labels) - 1) for label in labels}
            probabilities[selected] = 0.9
            answers[task_name] = DecisionAnswer(
                selected, probabilities, 0.8, method, 1.0
            )
        return DecisionResult("fake-local-model", self.backend_name, answers, 3, 100)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_client() -> FakeOpenJevClient:
    return FakeOpenJevClient()
