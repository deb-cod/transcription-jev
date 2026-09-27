from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class DecisionBackendError(RuntimeError):
    """Raised when a local decision backend cannot return a valid decision."""


@dataclass(frozen=True)
class DecisionAnswer:
    label: str
    probabilities: dict[str, float] | None
    confidence: float | None
    method: str
    inference_ms: float | None


@dataclass(frozen=True)
class DecisionResult:
    model: str
    backend: str
    answers: dict[str, DecisionAnswer]
    elapsed_ms: int
    input_tokens: int


class DecisionClient(Protocol):
    backend_name: str

    def health(self) -> dict[str, Any]: ...

    def available_models(self) -> list[dict[str, Any]]: ...

    def decide(
        self,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        method: str = "direct",
        model: str | None = None,
    ) -> DecisionResult: ...

