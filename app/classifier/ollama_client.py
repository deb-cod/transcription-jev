from __future__ import annotations

import json
import time
from typing import Any, Mapping

import httpx

from app.classifier.decision_client import (
    DecisionAnswer,
    DecisionBackendError,
    DecisionResult,
)


class OllamaError(DecisionBackendError):
    """Raised when Ollama cannot produce a valid constrained decision."""


class OllamaClient:
    backend_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "",
        timeout_seconds: float = 600,
        keep_alive: str = "10m",
        num_ctx: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        models = self.available_models()
        names = [item["name"] for item in models]
        default_available = not self.default_model or self.default_model in names
        return {
            "ready": bool(models) and default_available,
            "default_model": self.default_model,
            "default_model_available": default_available,
            "completion_models": names,
        }

    def available_models(self) -> list[dict[str, Any]]:
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"cannot list Ollama models: {exc}") from exc
        available: list[dict[str, Any]] = []
        for item in response.json().get("models", []):
            capabilities = [str(value) for value in item.get("capabilities", [])]
            if "completion" not in capabilities:
                continue
            details = item.get("details", {})
            available.append(
                {
                    "name": str(item.get("name") or item.get("model")),
                    "size": int(item.get("size", 0)),
                    "capabilities": capabilities,
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                }
            )
        return available

    def decide(
        self,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        method: str = "direct",
        model: str | None = None,
    ) -> DecisionResult:
        selected_model = model or self.default_model
        if not selected_model:
            raise OllamaError("no Ollama model was selected")
        installed = {item["name"] for item in self.available_models()}
        if selected_model not in installed:
            raise OllamaError(
                f"Ollama model {selected_model!r} is not installed or lacks completion capability"
            )
        schema = _response_schema(questions)
        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict classification system, not a chatbot. "
                        "Use only the supplied transcript. Choose exactly one allowed label "
                        "per task. Use an unknown label when evidence is insufficient. "
                        "For caller type, classify who the caller represents, not the person "
                        "being discussed. Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": _user_prompt(state, questions)},
            ],
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "keep_alive": self.keep_alive,
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            detail = getattr(exc.response, "text", "") if isinstance(exc, httpx.HTTPStatusError) else ""
            raise OllamaError(f"Ollama request failed: {exc}; {detail[:500]}") from exc
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        data = response.json()
        content = data.get("message", {}).get("content", "")
        try:
            choices = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama returned invalid structured JSON: {str(content)[:500]}") from exc
        answers: dict[str, DecisionAnswer] = {}
        for task_name, question in questions.items():
            allowed = set(question.get("criteria", {}))
            label = choices.get(task_name)
            if not isinstance(label, str) or label not in allowed:
                raise OllamaError(
                    f"Ollama returned invalid label {label!r} for task {task_name!r}"
                )
            answers[task_name] = DecisionAnswer(
                label=label,
                probabilities=None,
                confidence=None,
                method="structured_generation",
                inference_ms=float(elapsed_ms),
            )
        duration_ns = data.get("total_duration")
        reported_ms = round(float(duration_ns) / 1_000_000) if duration_ns is not None else elapsed_ms
        return DecisionResult(
            model=selected_model,
            backend=self.backend_name,
            answers=answers,
            elapsed_ms=reported_ms,
            input_tokens=int(data.get("prompt_eval_count", 0)),
        )


def _response_schema(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "enum": list(question.get("criteria", {}))}
            for name, question in questions.items()
        },
        "required": list(questions),
        "additionalProperties": False,
    }


def _user_prompt(state: str, questions: Mapping[str, Mapping[str, Any]]) -> str:
    task_lines: list[str] = []
    for name, question in questions.items():
        task_lines.append(f"TASK {name}: {question.get('instructions', '')}")
        for label, description in question.get("criteria", {}).items():
            task_lines.append(f"- {label}: {description}")
    return (
        "TRANSCRIPT\n---\n"
        + state
        + "\n---\n\nCLASSIFICATION TASKS\n"
        + "\n".join(task_lines)
    )

