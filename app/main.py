from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.api.routes import router
from app.classifier.generic_classifier import GenericDecisionClassifier
from app.classifier.openjev_client import OpenJevClient
from app.config.loader import ConfigStore
from app.config.settings import ENABLE_CONFIG_RELOAD, RETURN_CHUNK_DETAILS, config_path
from app.utils.logging import configure_logging


def create_app(
    path: str | Path | None = None,
    client: Any | None = None,
    return_chunk_details: bool | None = None,
) -> FastAPI:
    configure_logging()
    store = ConfigStore(path or config_path())
    runtime = store.snapshot()
    openjev_client = client or OpenJevClient(
        os.getenv("OPENJEV_URL", runtime.openjev_url), runtime.request_timeout_seconds
    )
    classifier = GenericDecisionClassifier(
        store,
        openjev_client,
        RETURN_CHUNK_DETAILS if return_chunk_details is None else return_chunk_details,
    )
    app = FastAPI(
        title="Local OpenJev Call Classifier",
        version="0.1.0",
        description="Local structured transcript classification; no external LLM APIs.",
    )
    app.state.config_store = store
    app.state.openjev_client = openjev_client
    app.state.classifier = classifier
    app.state.enable_config_reload = ENABLE_CONFIG_RELOAD
    app.include_router(router)
    return app


app = create_app()

