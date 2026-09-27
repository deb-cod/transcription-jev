from __future__ import annotations

from typing import Any, Mapping

import httpx

from app.classifier.decision_client import (
    DecisionAnswer,
    DecisionBackendError,
    DecisionResult,
)


class OpenJevError(DecisionBackendError):
    """Raised when the local OpenJev service returns an unusable result."""


class OpenJevClient:
    """Adapter for axsh/openjev's documented local HTTP API."""

    backend_name = "openjev"

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120,
        backend_name: str = "openjev",
        method_override: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_name = backend_name
        self.method_override = method_override
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def available_models(self) -> list[dict[str, Any]]:
        health = self.health()
        model = str(health.get("model", ""))
        return [{"name": model, "capabilities": ["classification"]}] if model else []

    def decide(
        self,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        method: str = "direct",
        model: str | None = None,
    ) -> DecisionResult:
        effective_method = self.method_override or method
        payload = {
            "state": state,
            "questions": questions,
            "options": {"method": effective_method},
        }
        if model:
            payload["model"] = model
        try:
            response = self._client.post("/v1/systemone", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            detail = getattr(exc.response, "text", "") if isinstance(exc, httpx.HTTPStatusError) else ""
            raise OpenJevError(f"OpenJev request failed: {exc}; {detail[:500]}") from exc
        data = response.json()
        parsed: dict[str, DecisionAnswer] = {}
        for question_id, answer in data.get("answers", {}).items():
            probabilities = answer.get("probabilities")
            choice = answer.get("choice")
            if not isinstance(choice, str) or not isinstance(probabilities, dict):
                raise OpenJevError(f"question {question_id!r} did not return choice probabilities")
            parsed[question_id] = DecisionAnswer(
                label=choice,
                probabilities={str(k): float(v) for k, v in probabilities.items()},
                confidence=float(answer["confidence"]) if answer.get("confidence") is not None else None,
                method=str(answer.get("method", effective_method)),
                inference_ms=_direct_ms(answer),
            )
        missing = set(questions) - set(parsed)
        if missing:
            raise OpenJevError(f"OpenJev omitted answers for: {', '.join(sorted(missing))}")
        usage = data.get("usage", {})
        return DecisionResult(
            model=str(data.get("model", "")),
            backend=self.backend_name,
            answers=parsed,
            elapsed_ms=int(data.get("elapsedMs", 0)),
            input_tokens=int(usage.get("input_tokens", 0)),
        )


def _direct_ms(answer: Mapping[str, Any]) -> float | None:
    value = answer.get("timings", {}).get("direct_ms")
    return float(value) if value is not None else None
