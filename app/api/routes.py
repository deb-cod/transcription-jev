from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.classifier.generic_classifier import TranscriptTooLongError
from app.classifier.decision_client import DecisionBackendError
from app.classifier.schemas import ClassifyRequest


logger = logging.getLogger("openjev_call_classifier.api")
router = APIRouter()


@router.post("/classify")
def classify(payload: ClassifyRequest, request: Request) -> dict[str, Any]:
    logger.info("request_received transcript_characters=%d", len(payload.transcript))
    try:
        result = request.app.state.classifier.classify(
            payload.transcript, backend=payload.backend, model=payload.model
        )
    except TranscriptTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DecisionBackendError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    labels = {
        name: (
            f"{value['label']} ({value['confidence']:.3f})"
            if value.get("confidence") is not None
            else f"{value['label']} (confidence unavailable)"
        )
        for name, value in result.items()
        if isinstance(value, dict) and "label" in value
    }
    logger.info(
        "classification_complete chunks=%d labels=%s latency_ms=%.3f",
        result["transcript_processing"]["chunks"],
        labels,
        result["model"]["latency_ms"],
    )
    return result


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    backends: dict[str, Any] = {}
    for name, client in request.app.state.decision_clients.items():
        try:
            backends[name] = client.health()
        except Exception as exc:  # Health must describe failure rather than crash.
            backends[name] = {"ready": False, "error": str(exc)}
    if {"openjev", "ollama"}.issubset(request.app.state.decision_clients):
        runtime = request.app.state.config_store.snapshot()
        backends["hybrid"] = {
            "ready": bool(backends.get("openjev", {}).get("ready"))
            and bool(backends.get("ollama", {}).get("ready")),
            "strategy": "native_first",
            "fallback_confidence": runtime.hybrid_fallback_confidence,
            "fallback_on_unknown": runtime.hybrid_fallback_on_unknown,
        }
    selected = request.app.state.default_backend
    loaded = bool(backends.get(selected, {}).get("ready"))
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "gpu_available": _gpu_available(),
        "default_backend": selected,
        "backends": backends,
    }


@router.get("/models")
def models(request: Request) -> dict[str, Any]:
    backends: dict[str, Any] = {}
    for name, client in request.app.state.decision_clients.items():
        try:
            backends[name] = {"models": client.available_models()}
        except Exception as exc:
            backends[name] = {"models": [], "error": str(exc)}
    if {"openjev", "ollama"}.issubset(request.app.state.decision_clients):
        native_models = backends.get("openjev", {}).get("models", [])
        ollama_models = backends.get("ollama", {}).get("models", [])
        hybrid_models = []
        if native_models and ollama_models:
            for item in ollama_models:
                hybrid_model = dict(item)
                hybrid_model["capabilities"] = list(item.get("capabilities", [])) + [
                    "native_first_fallback"
                ]
                hybrid_models.append(hybrid_model)
        backends["hybrid"] = {
            "models": hybrid_models,
            "strategy": "native_first",
            "fallback_model_source": "ollama",
        }
    return {"default_backend": request.app.state.default_backend, "backends": backends}


@router.get("/config")
def config(request: Request) -> dict[str, list[str]]:
    return request.app.state.config_store.public_view()


@router.post("/config/reload")
def reload_config(request: Request) -> dict[str, Any]:
    if not request.app.state.enable_config_reload:
        raise HTTPException(status_code=404, detail="configuration reload is disabled")
    try:
        runtime = request.app.state.config_store.reload()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info("configuration_reloaded tasks=%s", [task.name for task in runtime.tasks])
    return {"status": "reloaded", "classifications": request.app.state.config_store.public_view()}


def _gpu_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        return subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).returncode == 0
    except OSError:
        return False
