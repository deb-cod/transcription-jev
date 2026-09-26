from app.classifier.generic_classifier import GenericDecisionClassifier
from app.config.loader import ConfigStore


def test_classifier_returns_every_dynamic_task(fake_client) -> None:
    classifier = GenericDecisionClassifier(ConfigStore("config/classification.yaml"), fake_client)
    result = classifier.classify("Caller: This is a test transcript.")
    assert result["healthcare_related"]["label"] == "yes"
    assert set(result["caller_type"]["probabilities"]) == set(
        classifier.config_store.public_view()["caller_type"]
    )
    assert result["transcript_processing"]["truncated"] is False
    assert result["model"]["name"] == "fake-local-model"


def test_optional_chunk_details(fake_client) -> None:
    classifier = GenericDecisionClassifier(
        ConfigStore("config/classification.yaml"), fake_client, return_chunk_details=True
    )
    result = classifier.classify("Caller: Test.")
    assert result["debug"]["chunks"][0]["chunk"] == 1

