from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier.generic_classifier import GenericDecisionClassifier
from app.classifier.openjev_client import OpenJevClient
from app.config.loader import ConfigStore


def materialize(example: dict[str, Any]) -> str:
    filler = str(example.get("filler", ""))
    repeated = "\n".join(filler for _ in range(int(example.get("filler_repeat", 0))))
    return "\n".join(
        part for part in [str(example.get("prefix", "")), repeated, str(example.get("evidence", ""))] if part
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local classifier")
    parser.add_argument("--dataset", default="samples/sample_transcripts.json")
    parser.add_argument("--config", default="config/classification.yaml")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    examples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        examples = examples[: args.limit]
    store = ConfigStore(args.config)
    runtime = store.snapshot()
    client = OpenJevClient(runtime.openjev_url, runtime.request_timeout_seconds)
    classifier = GenericDecisionClassifier(store, client)
    records: list[dict[str, Any]] = []
    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = _gpu_memory_mb()
    started = time.perf_counter()
    try:
        for index, example in enumerate(examples, 1):
            transcript = materialize(example)
            item_started = time.perf_counter()
            result = classifier.classify(transcript)
            elapsed = (time.perf_counter() - item_started) * 1000
            records.append(
                {
                    "id": example["id"],
                    "length": example["length"],
                    "expected": example["expected"],
                    "predicted": {task.name: result[task.name]["label"] for task in runtime.tasks},
                    "latency_ms": elapsed,
                    "chunks": result["transcript_processing"]["chunks"],
                }
            )
            print(f"[{index}/{len(examples)}] {example['id']}: {elapsed:.1f} ms")
    finally:
        client.close()
    report = build_report(records)
    report["performance"] = {
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "average_latency_ms": _mean(record["latency_ms"] for record in records),
        "average_latency_per_chunk_ms": _mean(
            record["latency_ms"] / record["chunks"] for record in records
        ),
        "average_chunks": _mean(record["chunks"] for record in records),
        "process_rss_before_mb": round(rss_before / 1024**2, 2),
        "process_rss_after_mb": round(process.memory_info().rss / 1024**2, 2),
        "gpu_vram_before_mb": gpu_before,
        "gpu_vram_after_mb": _gpu_memory_mb(),
        "model_load_time_ms": None,
        "model_load_note": "Model is persistent and was already loaded; run.ps1 reports cold load time.",
    }
    print(json.dumps(report, indent=2))


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({task for record in records for task in record["expected"]})
    metrics = {task: task_metrics(records, task) for task in tasks}
    by_length: dict[str, Any] = {}
    for length in ("short", "medium", "long"):
        group = [record for record in records if record["length"] == length]
        by_length[length] = {
            "examples": len(group),
            "accuracy": {
                task: _accuracy(group, task) for task in tasks
            },
            "average_latency_ms": _mean(record["latency_ms"] for record in group),
            "average_chunks": _mean(record["chunks"] for record in group),
        }
    return {"examples": len(records), "tasks": metrics, "by_length": by_length}


def task_metrics(records: list[dict[str, Any]], task: str) -> dict[str, Any]:
    labels = sorted(
        {record["expected"][task] for record in records}
        | {record["predicted"][task] for record in records}
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_label: dict[str, Any] = {}
    for record in records:
        confusion[record["expected"][task]][record["predicted"][task]] += 1
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "support": sum(confusion[label].values()),
        }
    return {
        "accuracy": _accuracy(records, task),
        "per_label": per_label,
        "confusion_matrix": {actual: dict(confusion[actual]) for actual in labels},
    }


def _accuracy(records: list[dict[str, Any]], task: str) -> float | None:
    if not records:
        return None
    correct = sum(record["expected"][task] == record["predicted"][task] for record in records)
    return round(correct / len(records), 4)


def _mean(values: Any) -> float | None:
    items = list(values)
    return round(statistics.fmean(items), 3) if items else None


def _gpu_memory_mb() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else None
    except (OSError, ValueError, IndexError):
        return None


if __name__ == "__main__":
    main()
