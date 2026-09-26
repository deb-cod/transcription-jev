from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def config_path() -> Path:
    return Path(os.getenv("CLASSIFICATION_CONFIG", "config/classification.yaml"))


RETURN_CHUNK_DETAILS = env_bool("RETURN_CHUNK_DETAILS")
LOG_TRANSCRIPTS = env_bool("LOG_TRANSCRIPTS")
ENABLE_CONFIG_RELOAD = env_bool("ENABLE_CONFIG_RELOAD", True)

