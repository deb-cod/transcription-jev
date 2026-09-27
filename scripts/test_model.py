from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier.generic_classifier import GenericDecisionClassifier
from app.classifier.openjev_client import OpenJevClient
from app.classifier.ollama_client import OllamaClient
from app.config.loader import ConfigStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live local OpenJev classification")
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--backend", choices=("openjev", "ollama"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--text",
        default=(
            "Caller: This is Maria from Dr. Patel's cardiology office. "
            "I'm checking the prior authorization for an echocardiogram."
        ),
    )
    args = parser.parse_args()
    store = ConfigStore(args.config)
    runtime = store.snapshot()
    backend = args.backend or runtime.default_backend
    client = (
        OpenJevClient(runtime.openjev_url, runtime.request_timeout_seconds)
        if backend == "openjev"
        else OllamaClient(
            runtime.ollama_url,
            runtime.ollama_model,
            runtime.ollama_timeout_seconds,
            runtime.ollama_keep_alive,
            runtime.ollama_num_ctx,
        )
    )
    try:
        result = GenericDecisionClassifier(store, client, default_backend=backend).classify(
            args.text, backend=backend, model=args.model
        )
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
