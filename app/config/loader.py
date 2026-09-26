from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when classification configuration cannot be used safely."""


@dataclass(frozen=True)
class LabelConfig:
    name: str
    description: str


@dataclass(frozen=True)
class TaskConfig:
    name: str
    kind: str
    description: str
    threshold: float
    labels: tuple[LabelConfig, ...]


@dataclass(frozen=True)
class ChunkingConfig:
    enabled: bool
    max_characters: int
    overlap_characters: int
    preserve_speaker_turns: bool


@dataclass(frozen=True)
class RuntimeConfig:
    openjev_url: str
    method: str
    request_timeout_seconds: float
    max_total_characters: int
    chunking: ChunkingConfig
    tasks: tuple[TaskConfig, ...]
    aggregation: Mapping[str, Mapping[str, Any]]

    def aggregation_for(self, task_name: str) -> dict[str, Any]:
        merged = dict(self.aggregation.get("default", {}))
        merged.update(self.aggregation.get(task_name, {}))
        return merged


class ConfigStore:
    """Thread-safe, reloadable source of validated runtime configuration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._lock = RLock()
        self._raw: dict[str, Any] = {}
        self._runtime: RuntimeConfig | None = None
        self.reload()

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            if self._runtime is None:  # pragma: no cover - defensive
                raise ConfigError("configuration is not loaded")
            return self._runtime

    def raw_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._raw)

    def reload(self) -> RuntimeConfig:
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot load {self.path}: {exc}") from exc
        runtime = _validate(raw)
        with self._lock:
            self._raw = raw
            self._runtime = runtime
        return runtime

    def public_view(self) -> dict[str, list[str]]:
        return {
            task.name: [label.name for label in task.labels]
            for task in self.snapshot().tasks
        }


def _validate(raw: Any) -> RuntimeConfig:
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    openjev = _mapping(raw.get("openjev", {}), "openjev")
    url = str(openjev.get("url", "http://127.0.0.1:8090")).rstrip("/")
    method = str(openjev.get("method", "direct"))
    if method not in {"direct", "generation", "both"}:
        raise ConfigError("openjev.method must be direct, generation, or both")

    transcript = _mapping(raw.get("transcript", {}), "transcript")
    chunking_raw = _mapping(transcript.get("chunking", {}), "transcript.chunking")
    max_chars = _positive_int(chunking_raw.get("max_characters", 4000), "max_characters")
    overlap = int(chunking_raw.get("overlap_characters", 400))
    if overlap < 0 or overlap >= max_chars:
        raise ConfigError("overlap_characters must be >= 0 and smaller than max_characters")

    tasks_raw = _mapping(raw.get("classifications"), "classifications")
    tasks: list[TaskConfig] = []
    for task_name, task_value in tasks_raw.items():
        task = _mapping(task_value, f"classifications.{task_name}")
        if not task.get("enabled", True):
            continue
        kind = str(task.get("type", "choice"))
        if kind not in {"choice", "binary"}:
            raise ConfigError(f"{task_name}: supported types are choice and binary")
        description = str(task.get("description", "")).strip()
        if not description:
            raise ConfigError(f"{task_name}: description is required")
        threshold = float(task.get("threshold", 0.6))
        if not 0.0 <= threshold <= 1.0:
            raise ConfigError(f"{task_name}: threshold must be between 0 and 1")
        labels_raw = _mapping(task.get("labels"), f"{task_name}.labels")
        labels: list[LabelConfig] = []
        for label_name, label_value in labels_raw.items():
            label = _mapping(label_value, f"{task_name}.labels.{label_name}")
            if not label.get("enabled", True):
                continue
            label_description = str(label.get("description", "")).strip()
            if not label_description:
                raise ConfigError(f"{task_name}.{label_name}: description is required")
            labels.append(LabelConfig(str(label_name), label_description))
        # OpenJev currently validates choice criteria at 2..20 options.
        if not 2 <= len(labels) <= 20:
            raise ConfigError(f"{task_name}: OpenJev choice tasks require 2 to 20 enabled labels")
        tasks.append(TaskConfig(str(task_name), kind, description, threshold, tuple(labels)))
    if not tasks:
        raise ConfigError("at least one classification task must be enabled")

    aggregation = _mapping(raw.get("aggregation", {}), "aggregation")
    return RuntimeConfig(
        openjev_url=url,
        method=method,
        request_timeout_seconds=float(openjev.get("request_timeout_seconds", 120)),
        max_total_characters=_positive_int(
            transcript.get("max_total_characters", 200000), "max_total_characters"
        ),
        chunking=ChunkingConfig(
            enabled=bool(chunking_raw.get("enabled", True)),
            max_characters=max_chars,
            overlap_characters=overlap,
            preserve_speaker_turns=bool(chunking_raw.get("preserve_speaker_turns", True)),
        ),
        tasks=tuple(tasks),
        aggregation=deepcopy(aggregation),
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise ConfigError(f"{name} must be positive")
    return converted

