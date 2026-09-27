from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_index: int
    probabilities: Mapping[str, float]
    model_confidence: float | None


@dataclass(frozen=True)
class AggregatedDecision:
    raw_label: str
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class CategoricalDecision:
    label: str
    vote_counts: dict[str, int]


def aggregate_evidence(
    evidence: Sequence[ChunkEvidence],
    label_names: Sequence[str],
    threshold: float,
    options: Mapping[str, object],
) -> AggregatedDecision:
    if not evidence:
        raise ValueError("cannot aggregate empty evidence")
    strategy = str(options.get("strategy", "confidence_weighted_average"))
    if strategy == "positive_evidence_max":
        return _positive_evidence_max(evidence, label_names, threshold, options)
    if strategy not in {"confidence_weighted_average", "confidence_powered_average"}:
        raise ValueError(f"unsupported aggregation strategy: {strategy}")
    minimum_weight = float(options.get("minimum_chunk_weight", 0.05))
    confidence_power = (
        float(options.get("confidence_power", 1.0))
        if strategy == "confidence_powered_average"
        else 1.0
    )
    intro_boost = float(options.get("introductory_chunk_boost", 1.0))
    totals = {label: 0.0 for label in label_names}
    total_weight = 0.0
    for item in evidence:
        confidence = item.model_confidence
        if confidence is None or not math.isfinite(confidence):
            confidence = max(item.probabilities.values(), default=minimum_weight)
        weight = max(minimum_weight, confidence) ** confidence_power
        if item.chunk_index == 0:
            weight *= intro_boost
        for label in label_names:
            totals[label] += weight * max(0.0, float(item.probabilities.get(label, 0.0)))
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("aggregation weights must be positive")
    probabilities = {label: value / total_weight for label, value in totals.items()}
    normalizer = sum(probabilities.values())
    if normalizer <= 0:
        raise ValueError("aggregated probability mass is zero")
    probabilities = {label: value / normalizer for label, value in probabilities.items()}
    raw_label = max(label_names, key=probabilities.__getitem__)
    top_probability = probabilities[raw_label]
    label = raw_label if top_probability >= threshold else "unknown"
    return AggregatedDecision(raw_label, label, top_probability, probabilities)


def _positive_evidence_max(
    evidence: Sequence[ChunkEvidence],
    label_names: Sequence[str],
    threshold: float,
    options: Mapping[str, object],
) -> AggregatedDecision:
    if len(label_names) != 2:
        raise ValueError("positive_evidence_max requires exactly two labels")
    positive = str(options.get("positive_label", ""))
    if positive not in label_names:
        raise ValueError("positive_evidence_max positive_label is not configured for this task")
    negative = next(label for label in label_names if label != positive)
    positive_probability = max(
        max(0.0, min(1.0, float(item.probabilities.get(positive, 0.0))))
        for item in evidence
    )
    probabilities = {positive: positive_probability, negative: 1.0 - positive_probability}
    raw_label = max(label_names, key=probabilities.__getitem__)
    top_probability = probabilities[raw_label]
    label = raw_label if top_probability >= threshold else "unknown"
    return AggregatedDecision(raw_label, label, top_probability, probabilities)


def aggregate_categorical_evidence(
    labels: Sequence[str],
    label_names: Sequence[str],
    options: Mapping[str, object],
) -> CategoricalDecision:
    """Aggregate label-only backends without presenting vote shares as probabilities."""
    if not labels:
        raise ValueError("cannot aggregate empty categorical evidence")
    allowed = set(label_names)
    invalid = [label for label in labels if label not in allowed]
    if invalid:
        raise ValueError(f"categorical evidence contains invalid labels: {invalid}")
    counts = Counter(labels)
    strategy = str(options.get("strategy", "confidence_powered_average"))
    if strategy == "positive_evidence_max":
        positive = str(options.get("positive_label", ""))
        if positive not in allowed or len(label_names) != 2:
            raise ValueError("positive_evidence_max requires a valid positive label and two labels")
        if counts[positive] > 0:
            return CategoricalDecision(positive, dict(counts))

    candidates = [label for label in label_names if counts[label] > 0 and label != "unknown"]
    if not candidates:
        return CategoricalDecision("unknown" if "unknown" in allowed else labels[0], dict(counts))

    # Caller identity is commonly introduced early; apply the configured boost
    # only as a tie-breaking weight. All raw counts remain visible in output.
    weights = {label: float(counts[label]) for label in candidates}
    first = labels[0]
    if first in weights:
        weights[first] += max(0.0, float(options.get("introductory_chunk_boost", 1.0)) - 1.0)
    winner = max(candidates, key=lambda label: (weights[label], -label_names.index(label)))
    return CategoricalDecision(winner, dict(counts))
