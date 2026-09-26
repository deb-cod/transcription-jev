from __future__ import annotations

from typing import Any

import pytest

from app.classifier.openjev_client import OpenJevAnswer, OpenJevResult


class FakeOpenJevClient:
    def health(self) -> dict[str, Any]:
        return {"ready": True, "model": "fake-local-model", "llama_reachable": True}

    def decide(
        self, state: str, questions: dict[str, dict[str, Any]], method: str = "direct"
    ) -> OpenJevResult:
        answers: dict[str, OpenJevAnswer] = {}
        for task_name, question in questions.items():
            labels = list(question["criteria"])
            selected = labels[0]
            probabilities = {label: 0.1 / (len(labels) - 1) for label in labels}
            probabilities[selected] = 0.9
            answers[task_name] = OpenJevAnswer(
                selected, probabilities, 0.8, method, 1.0
            )
        return OpenJevResult("fake-local-model", answers, 3, 100)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_client() -> FakeOpenJevClient:
    return FakeOpenJevClient()

