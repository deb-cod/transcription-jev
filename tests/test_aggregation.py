import pytest

from app.transcript.evidence_aggregator import ChunkEvidence, aggregate_evidence


def test_confident_chunk_outweighs_uncertain_chunks() -> None:
    result = aggregate_evidence(
        [
            ChunkEvidence(0, {"a": 0.51, "b": 0.49}, 0.02),
            ChunkEvidence(1, {"a": 0.05, "b": 0.95}, 0.8),
        ],
        ["a", "b"],
        0.6,
        {"strategy": "confidence_weighted_average", "minimum_chunk_weight": 0.05},
    )
    assert result.label == "b"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_threshold_returns_unknown_but_preserves_raw_label() -> None:
    result = aggregate_evidence(
        [ChunkEvidence(0, {"a": 0.55, "b": 0.45}, 0.1)],
        ["a", "b"],
        0.6,
        {},
    )
    assert result.raw_label == "a"
    assert result.label == "unknown"
    assert result.confidence == pytest.approx(0.55)


def test_positive_evidence_is_not_diluted_by_many_negative_chunks() -> None:
    result = aggregate_evidence(
        [
            ChunkEvidence(0, {"yes": 0.05, "no": 0.95}, 0.7),
            ChunkEvidence(1, {"yes": 0.03, "no": 0.97}, 0.8),
            ChunkEvidence(2, {"yes": 0.88, "no": 0.12}, 0.4),
        ],
        ["yes", "no"],
        0.6,
        {"strategy": "positive_evidence_max", "positive_label": "yes"},
    )
    assert result.label == "yes"
    assert result.confidence == pytest.approx(0.88)


def test_powered_average_favors_sharp_evidence() -> None:
    result = aggregate_evidence(
        [
            ChunkEvidence(0, {"target": 0.45, "unknown": 0.55}, 0.6),
            ChunkEvidence(1, {"target": 0.95, "unknown": 0.05}, 0.9),
        ],
        ["target", "unknown"],
        0.6,
        {"strategy": "confidence_powered_average", "confidence_power": 8},
    )
    assert result.label == "target"
