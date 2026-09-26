from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier.generic_classifier import GenericDecisionClassifier
from app.classifier.openjev_client import OpenJevClient
from app.config.loader import ConfigStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark combined vs separate OpenJev questions")
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    store = ConfigStore(args.config)
    runtime = store.snapshot()
    client = OpenJevClient(runtime.openjev_url, runtime.request_timeout_seconds)
    questions = GenericDecisionClassifier(store, client).build_questions(runtime)
    state = (
        "Caller: This is Maria from Dr. Patel's cardiology office. "
        "We are checking the prior authorization for an echocardiogram."
    )
    combined: list[float] = []
    separate: list[float] = []
    try:
        for _ in range(args.runs):
            started = time.perf_counter()
            client.decide(state, questions, runtime.method)
            combined.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            for name, question in questions.items():
                client.decide(state, {name: question}, runtime.method)
            separate.append((time.perf_counter() - started) * 1000)
    finally:
        client.close()
    print(f"runs: {args.runs}")
    print(f"combined average ms: {statistics.fmean(combined):.3f}")
    print(f"separate average ms: {statistics.fmean(separate):.3f}")
    print("selected: combined (one request per chunk; OpenJev schedules its questions)")


if __name__ == "__main__":
    main()
