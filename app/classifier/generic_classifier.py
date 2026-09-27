from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

from app.classifier.decision_client import (
    DecisionAnswer,
    DecisionBackendError,
    DecisionClient,
)
from app.config.loader import ConfigStore, RuntimeConfig, TaskConfig
from app.transcript.chunker import chunk_transcript
from app.transcript.evidence_aggregator import (
    ChunkEvidence,
    aggregate_categorical_evidence,
    aggregate_evidence,
)
from app.transcript.preprocessor import preprocess_transcript


class TranscriptTooLongError(ValueError):
    pass


@dataclass
class GenericDecisionClassifier:
    config_store: ConfigStore
    client: DecisionClient | Mapping[str, DecisionClient]
    return_chunk_details: bool = False
    default_backend: str | None = None

    def build_questions(self, runtime: RuntimeConfig | None = None) -> dict[str, dict[str, Any]]:
        active = runtime or self.config_store.snapshot()
        return {
            task.name: {
                "type": "choice",
                "instructions": task.description,
                "criteria": {label.name: label.description for label in task.labels},
            }
            for task in active.tasks
        }

    def classify(
        self,
        transcript: str,
        backend: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        runtime = self.config_store.snapshot()
        clients = (
            dict(self.client)
            if isinstance(self.client, Mapping)
            else {getattr(self.client, "backend_name", runtime.default_backend): self.client}
        )
        selected_backend = (backend or self.default_backend or runtime.default_backend).lower()
        if selected_backend == "hybrid":
            return self._classify_hybrid(transcript, model, runtime, clients, started)
        if selected_backend not in clients:
            raise ValueError(
                f"backend {selected_backend!r} is unavailable; choose one of {sorted(clients)}"
            )
        decision_client = clients[selected_backend]
        if not transcript or not transcript.strip():
            raise ValueError("transcript must not be empty")
        if len(transcript) > runtime.max_total_characters:
            raise TranscriptTooLongError(
                f"transcript has {len(transcript)} characters; configured maximum is "
                f"{runtime.max_total_characters}. Input was rejected, not truncated."
            )
        processed = preprocess_transcript(transcript)
        chunks = chunk_transcript(processed, runtime.chunking)
        questions = self.build_questions(runtime)
        evidence: dict[str, list[tuple[int, DecisionAnswer]]] = {
            task.name: [] for task in runtime.tasks
        }
        debug_chunks: list[dict[str, Any]] = []
        model_name = ""
        backend_elapsed = 0
        input_tokens = 0
        per_chunk_ms: list[float] = []
        methods: set[str] = set()
        result_backend = selected_backend

        for chunk in chunks:
            result = decision_client.decide(
                chunk.text, questions, runtime.method, model=model
            )
            model_name = result.model or model_name
            result_backend = result.backend
            backend_elapsed += result.elapsed_ms
            per_chunk_ms.append(float(result.elapsed_ms))
            input_tokens += result.input_tokens
            details: dict[str, Any] = {"chunk": chunk.index + 1}
            for task in runtime.tasks:
                answer = result.answers[task.name]
                evidence[task.name].append((chunk.index, answer))
                methods.add(answer.method)
                details[task.name] = {
                    "label": answer.label,
                    "confidence": answer.confidence,
                    "probabilities": answer.probabilities,
                }
            debug_chunks.append(details)

        output: dict[str, Any] = {}
        for task in runtime.tasks:
            output[task.name] = self._aggregate_task(task, evidence[task.name], runtime)
        output["transcript_processing"] = {
            "characters": len(transcript),
            "processed_characters": len(processed),
            "chunks": len(chunks),
            "truncated": False,
            "input_tokens_reported": input_tokens,
        }
        output["model"] = {
            "name": model_name,
            "backend": result_backend,
            "method": next(iter(methods)) if len(methods) == 1 else sorted(methods),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "backend_elapsed_ms": backend_elapsed,
            "average_chunk_inference_ms": (
                round(sum(per_chunk_ms) / len(per_chunk_ms), 3) if per_chunk_ms else None
            ),
            "probability_source": _probability_source(evidence, methods),
        }
        if self.return_chunk_details:
            output["debug"] = {"chunks": debug_chunks}
        return output

    def _classify_hybrid(
        self,
        transcript: str,
        model: str | None,
        runtime: RuntimeConfig,
        clients: Mapping[str, DecisionClient],
        started: float,
    ) -> dict[str, Any]:
        if "openjev" not in clients or "ollama" not in clients:
            raise ValueError("hybrid backend requires native OpenJev and Ollama")

        fallback_reasons: list[str] = []
        fallback_tasks: list[dict[str, Any]] = []
        native: dict[str, Any] | None = None
        try:
            native = self.classify(transcript, backend="openjev")
            for task in runtime.tasks:
                decision = native[task.name]
                confidence = decision.get("confidence")
                if runtime.hybrid_fallback_on_unknown and decision.get("label") == "unknown":
                    fallback_reasons.append(f"{task.name}:unknown")
                    fallback_tasks.append(
                        {
                            "task": task.name,
                            "reason": "unknown",
                            "label": decision.get("label"),
                            "raw_label": decision.get("raw_label"),
                            "confidence": confidence,
                            "task_threshold": decision.get("threshold"),
                            "hybrid_threshold": runtime.hybrid_fallback_confidence,
                        }
                    )
                elif confidence is None or confidence < runtime.hybrid_fallback_confidence:
                    fallback_reasons.append(
                        f"{task.name}:confidence={confidence if confidence is not None else 'unavailable'}"
                    )
                    fallback_tasks.append(
                        {
                            "task": task.name,
                            "reason": (
                                "confidence_unavailable"
                                if confidence is None
                                else "low_confidence"
                            ),
                            "label": decision.get("label"),
                            "raw_label": decision.get("raw_label"),
                            "confidence": confidence,
                            "task_threshold": decision.get("threshold"),
                            "hybrid_threshold": runtime.hybrid_fallback_confidence,
                        }
                    )
        except DecisionBackendError as exc:
            fallback_reasons.append(f"openjev_error:{exc}")

        if native is not None and not fallback_reasons:
            native_latency = native["model"]["latency_ms"]
            native["model"].update(
                {
                    "backend": "hybrid",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "routing": {
                        "strategy": "native_first",
                        "selected_backend": "openjev",
                        "fallback_used": False,
                        "fallback_confidence": runtime.hybrid_fallback_confidence,
                        "native_latency_ms": native_latency,
                    },
                }
            )
            return native

        fallback = self.classify(transcript, backend="ollama", model=model)
        fallback_latency = fallback["model"]["latency_ms"]
        native_latency = native["model"]["latency_ms"] if native is not None else None
        native_elapsed = native["model"].get("backend_elapsed_ms", 0) if native else 0
        fallback["model"].update(
            {
                "backend": "hybrid",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "backend_elapsed_ms": native_elapsed
                + fallback["model"].get("backend_elapsed_ms", 0),
                "routing": {
                    "strategy": "native_first",
                    "selected_backend": "ollama",
                    "fallback_used": True,
                    "fallback_confidence": runtime.hybrid_fallback_confidence,
                    "fallback_reasons": fallback_reasons,
                    "fallback_tasks": fallback_tasks,
                    "native_latency_ms": native_latency,
                    "ollama_latency_ms": fallback_latency,
                },
            }
        )
        return fallback

    def _aggregate_task(
        self,
        task: TaskConfig,
        evidence: list[tuple[int, DecisionAnswer]],
        runtime: RuntimeConfig,
    ) -> dict[str, Any]:
        probability_flags = {answer.probabilities is not None for _, answer in evidence}
        if len(probability_flags) != 1:
            raise ValueError(f"task {task.name!r} mixed probabilistic and categorical evidence")
        if probability_flags == {False}:
            decision = aggregate_categorical_evidence(
                [answer.label for _, answer in evidence],
                [label.name for label in task.labels],
                runtime.aggregation_for(task.name),
            )
            return {
                "raw_label": decision.label,
                "label": decision.label,
                "confidence": None,
                "threshold": task.threshold,
                "threshold_applied": False,
                "probabilities": None,
                "vote_counts": decision.vote_counts,
            }
        decision = aggregate_evidence(
            [
                ChunkEvidence(index, answer.probabilities or {}, answer.confidence)
                for index, answer in evidence
            ],
            [label.name for label in task.labels],
            task.threshold,
            runtime.aggregation_for(task.name),
        )
        return {
            "raw_label": decision.raw_label,
            "label": decision.label,
            "confidence": decision.confidence,
            "threshold": task.threshold,
            "threshold_applied": True,
            "probabilities": decision.probabilities,
        }


def _probability_source(
    evidence: Mapping[str, list[tuple[int, DecisionAnswer]]], methods: set[str]
) -> str:
    if not all(
        answer.probabilities is not None
        for task_evidence in evidence.values()
        for _, answer in task_evidence
    ):
        return "unavailable"
    if methods == {"generation"}:
        return "model_generated_distribution"
    if methods.issubset({"direct", "both"}):
        return "token_logprobs"
    return "mixed"
