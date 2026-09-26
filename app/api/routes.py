from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.classifier.generic_classifier import TranscriptTooLongError
from app.classifier.openjev_client import OpenJevError
from app.classifier.schemas import ClassifyRequest


logger = logging.getLogger("openjev_call_classifier.api")
router = APIRouter()


@router.post("/classify")
def classify(payload: ClassifyRequest, request: Request) -> dict[str, Any]:
    logger.info("request_received transcript_characters=%d", len(payload.transcript))
    try:
        result = request.app.state.classifier.classify(payload.transcript)
    except TranscriptTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OpenJevError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    labels = {
        name: f"{value['label']} ({value['confidence']:.3f})"
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
    try:
        backend = request.app.state.openjev_client.health()
        loaded = bool(backend.get("ready"))
    except Exception as exc:  # Health must describe failure rather than crash.
        backend = {"error": str(exc)}
        loaded = False
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "gpu_available": _gpu_available(),
        "backend": backend,
    }


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

