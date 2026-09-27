from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.api.routes import router
from app.classifier.generic_classifier import GenericDecisionClassifier
from app.classifier.openjev_client import OpenJevClient
from app.classifier.ollama_client import OllamaClient
from app.config.loader import ConfigStore
from app.config.settings import ENABLE_CONFIG_RELOAD, RETURN_CHUNK_DETAILS, config_path
from app.utils.logging import configure_logging


_CLIENT_BACKENDS = {"openjev", "openjev_gemma", "openjev_ollama", "ollama"}


def _configured_client_backends() -> set[str] | None:
    """Return the optional comma-separated runtime backend allow-list."""
    raw = os.getenv("INFERENCE_BACKENDS", "").strip()
    if not raw:
        return None

    enabled = {name.strip().lower() for name in raw.split(",") if name.strip()}
    unknown = enabled - _CLIENT_BACKENDS
    if unknown:
        raise ValueError(
            "INFERENCE_BACKENDS contains unsupported values: "
            + ", ".join(sorted(unknown))
        )
    if not enabled:
        raise ValueError("INFERENCE_BACKENDS must enable at least one inference backend")
    return enabled


def _validate_decision_clients(
    clients: dict[str, Any], default_backend: str
) -> tuple[dict[str, Any], str]:
    if default_backend == "hybrid":
        if not {"openjev", "ollama"}.issubset(clients):
            raise ValueError(
                "hybrid requires openjev and ollama in INFERENCE_BACKENDS"
            )
    elif default_backend not in clients:
        raise ValueError(
            f"INFERENCE_BACKEND {default_backend!r} is not enabled by INFERENCE_BACKENDS"
        )
    return clients, default_backend


def create_app(
    path: str | Path | None = None,
    client: Any | None = None,
    return_chunk_details: bool | None = None,
) -> FastAPI:
    configure_logging()
    store = ConfigStore(path or config_path())
    runtime = store.snapshot()
    default_backend = os.getenv("INFERENCE_BACKEND", runtime.default_backend).lower()
    if client is not None:
        backend_name = getattr(client, "backend_name", default_backend)
        decision_clients = {backend_name: client}
        default_backend = backend_name
    else:
        enabled_backends = _configured_client_backends()
        decision_clients = {}
        if enabled_backends is None or "openjev" in enabled_backends:
            decision_clients["openjev"] = OpenJevClient(
                os.getenv("OPENJEV_URL", runtime.openjev_url),
                runtime.request_timeout_seconds,
                backend_name="openjev",
                method_override="direct",
            )
        if enabled_backends is None or "openjev_ollama" in enabled_backends:
            decision_clients["openjev_ollama"] = OpenJevClient(
                os.getenv("OPENJEV_OLLAMA_SERVICE_URL", runtime.openjev_ollama_url),
                runtime.ollama_timeout_seconds,
                backend_name="openjev_ollama",
                method_override="generation",
            )
        if enabled_backends is None or "openjev_gemma" in enabled_backends:
            decision_clients["openjev_gemma"] = OpenJevClient(
                os.getenv("OPENJEV_GEMMA_URL", runtime.openjev_gemma_url),
                runtime.ollama_timeout_seconds,
                backend_name="openjev_gemma",
                method_override="direct",
            )
        if enabled_backends is None or "ollama" in enabled_backends:
            decision_clients["ollama"] = OllamaClient(
                os.getenv("OLLAMA_URL", runtime.ollama_url),
                os.getenv("OLLAMA_MODEL", runtime.ollama_model),
                runtime.ollama_timeout_seconds,
                runtime.ollama_keep_alive,
                runtime.ollama_num_ctx,
            )
        decision_clients, default_backend = _validate_decision_clients(
            decision_clients, default_backend
        )
    classifier = GenericDecisionClassifier(
        store,
        decision_clients,
        RETURN_CHUNK_DETAILS if return_chunk_details is None else return_chunk_details,
        default_backend,
    )
    app = FastAPI(
        title="Local OpenJev Call Classifier",
        version="0.1.0",
        description="Local structured transcript classification; no external LLM APIs.",
    )
    app.state.config_store = store
    app.state.decision_clients = decision_clients
    app.state.default_backend = default_backend
    app.state.classifier = classifier
    app.state.enable_config_reload = ENABLE_CONFIG_RELOAD
    app.include_router(router)
    return app


app = create_app()
