from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import httpx


@dataclass(frozen=True)
class Backend:
    name: str
    model: str | None = None


BACKENDS = (
    Backend("hybrid", "gemma4:e4b"),
    Backend("openjev", "minicpm5-2b-q4_k_m"),
    Backend("openjev_gemma", "gemma4-e4b-native"),
    Backend("ollama", "gemma4:e4b"),
    Backend("openjev_ollama", "gemma4:e4b"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark every selectable inference backend through FastAPI."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--transcript", default="samples/very_long_benchmark_transcript.txt"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--watts", type=float, default=100)
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=0.15)
    parser.add_argument(
        "--backend",
        action="append",
        choices=[item.name for item in BACKENDS],
        help="Benchmark only this backend; repeat the option to select several.",
    )
    return parser.parse_args()


def classify(
    client: httpx.Client, api_url: str, transcript: str, backend: Backend
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "transcript": transcript,
        "backend": backend.name,
    }
    if backend.model:
        payload["model"] = backend.model
    started = time.perf_counter()
    response = client.post(f"{api_url.rstrip('/')}/classify", json=payload)
    wall_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), wall_ms


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def usd_for_requests(
    latency_ms: float, watts: float, usd_per_kwh: float, requests: int
) -> float:
    kwh_per_request = watts * (latency_ms / 1000) / 3_600_000
    return kwh_per_request * usd_per_kwh * requests


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.runs < 1:
        raise SystemExit("--warmups must be >= 0 and --runs must be >= 1")
    transcript_path = Path(args.transcript)
    transcript = transcript_path.read_text(encoding="utf-8")
    report: dict[str, Any] = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "method": {
            "warmups_per_backend": args.warmups,
            "measured_runs_per_backend": args.runs,
            "sequential_requests": True,
            "cost_assumed_watts": args.watts,
            "cost_assumed_usd_per_kwh": args.electricity_usd_per_kwh,
        },
        "transcript": {
            "path": str(transcript_path),
            "characters": len(transcript),
            "sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        },
        "results": [],
    }
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{args.api_url.rstrip('/')}/health")
        health.raise_for_status()
        report["health_before_test"] = health.json()
        selected_backends = (
            tuple(item for item in BACKENDS if item.name in args.backend)
            if args.backend
            else BACKENDS
        )
        for backend in selected_backends:
            entry: dict[str, Any] = {
                "backend": backend.name,
                "requested_model": backend.model,
            }
            try:
                for _ in range(args.warmups):
                    classify(client, args.api_url, transcript, backend)
                samples: list[dict[str, Any]] = []
                for run in range(1, args.runs + 1):
                    result, wall_ms = classify(
                        client, args.api_url, transcript, backend
                    )
                    samples.append(
                        {
                            "run": run,
                            "application_latency_ms": result["model"]["latency_ms"],
                            "client_wall_ms": round(wall_ms, 3),
                            "backend_elapsed_ms": result["model"].get(
                                "backend_elapsed_ms"
                            ),
                            "average_chunk_inference_ms": result["model"].get(
                                "average_chunk_inference_ms"
                            ),
                            "routing": result["model"].get("routing"),
                        }
                    )
                latencies = [x["application_latency_ms"] for x in samples]
                mean_ms = statistics.fmean(latencies)
                last_result = result
                entry.update(
                    {
                        "status": "ok",
                        "returned_model": last_result["model"]["name"],
                        "chunks": last_result["transcript_processing"]["chunks"],
                        "input_tokens_reported": last_result[
                            "transcript_processing"
                        ]["input_tokens_reported"],
                        "labels": {
                            task: {
                                "label": last_result[task]["label"],
                                "confidence": last_result[task]["confidence"],
                            }
                            for task in (
                                "healthcare_related",
                                "caller_type",
                                "intent",
                            )
                        },
                        "samples": samples,
                        "application_latency_ms": {
                            "mean": round(mean_ms, 3),
                            "median": round(statistics.median(latencies), 3),
                            "min": round(min(latencies), 3),
                            "max": round(max(latencies), 3),
                            "p95_interpolated": round(percentile(latencies, 0.95), 3),
                        },
                        "illustrative_energy_cost_usd": {
                            "per_10_000_requests": round(
                                usd_for_requests(
                                    mean_ms,
                                    args.watts,
                                    args.electricity_usd_per_kwh,
                                    10_000,
                                ),
                                6,
                            ),
                            "per_1_000_000_requests": round(
                                usd_for_requests(
                                    mean_ms,
                                    args.watts,
                                    args.electricity_usd_per_kwh,
                                    1_000_000,
                                ),
                                4,
                            ),
                        },
                    }
                )
            except Exception as exc:
                entry.update({"status": "error", "error": str(exc)})
            report["results"].append(entry)
            print(json.dumps(entry, indent=2), flush=True)
    print("BENCHMARK_REPORT_JSON")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
