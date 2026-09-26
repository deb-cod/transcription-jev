from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from app.classifier.openjev_client import OpenJevResult
from app.config.loader import ConfigStore, RuntimeConfig, TaskConfig
from app.transcript.chunker import chunk_transcript
from app.transcript.evidence_aggregator import ChunkEvidence, aggregate_evidence
from app.transcript.preprocessor import preprocess_transcript


class DecisionClient(Protocol):
    def decide(self, state: str, questions: dict[str, dict[str, Any]], method: str) -> OpenJevResult: ...


class TranscriptTooLongError(ValueError):
    pass


@dataclass
class GenericDecisionClassifier:
    config_store: ConfigStore
    client: DecisionClient
    return_chunk_details: bool = False

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

    def classify(self, transcript: str) -> dict[str, Any]:
        started = time.perf_counter()
        runtime = self.config_store.snapshot()
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
        evidence: dict[str, list[ChunkEvidence]] = {task.name: [] for task in runtime.tasks}
        debug_chunks: list[dict[str, Any]] = []
        model_name = ""
        openjev_elapsed = 0
        input_tokens = 0
        per_chunk_ms: list[float] = []

        for chunk in chunks:
            result = self.client.decide(chunk.text, questions, runtime.method)
            model_name = result.model or model_name
            openjev_elapsed += result.elapsed_ms
            input_tokens += result.input_tokens
            details: dict[str, Any] = {"chunk": chunk.index + 1}
            for task in runtime.tasks:
                answer = result.answers[task.name]
                evidence[task.name].append(
                    ChunkEvidence(chunk.index, answer.probabilities, answer.confidence)
                )
                details[task.name] = {
                    "label": answer.label,
                    "confidence": answer.confidence,
                    "probabilities": answer.probabilities,
                }
                if answer.direct_ms is not None:
                    per_chunk_ms.append(answer.direct_ms)
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
            "backend": "axsh/openjev + llama.cpp",
            "method": runtime.method,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "openjev_elapsed_ms": openjev_elapsed,
            "average_question_inference_ms": (
                round(sum(per_chunk_ms) / len(per_chunk_ms), 3) if per_chunk_ms else None
            ),
        }
        if self.return_chunk_details:
            output["debug"] = {"chunks": debug_chunks}
        return output

    def _aggregate_task(
        self,
        task: TaskConfig,
        evidence: list[ChunkEvidence],
        runtime: RuntimeConfig,
    ) -> dict[str, Any]:
        decision = aggregate_evidence(
            evidence,
            [label.name for label in task.labels],
            task.threshold,
            runtime.aggregation_for(task.name),
        )
        return {
            "raw_label": decision.raw_label,
            "label": decision.label,
            "confidence": decision.confidence,
            "threshold": task.threshold,
            "probabilities": decision.probabilities,
        }

