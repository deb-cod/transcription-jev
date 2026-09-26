from pathlib import Path

import yaml

from app.classifier.generic_classifier import GenericDecisionClassifier
from app.config.loader import ConfigStore


def test_new_task_and_label_are_loaded_without_source_change(tmp_path: Path, fake_client) -> None:
    raw = yaml.safe_load(Path("config/classification.yaml").read_text(encoding="utf-8"))
    raw["classifications"]["caller_type"]["labels"]["dental_office"] = {
        "description": "Dental clinic or dentist office."
    }
    raw["classifications"]["call_priority"] = {
        "enabled": True,
        "type": "choice",
        "description": "Determine urgency.",
        "threshold": 0.65,
        "labels": {
            "low": {"description": "Routine."},
            "high": {"description": "Prompt attention."},
        },
    }
    target = tmp_path / "classification.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    store = ConfigStore(target)
    questions = GenericDecisionClassifier(store, fake_client).build_questions()
    assert "dental_office" in questions["caller_type"]["criteria"]
    assert "call_priority" in questions


def test_disabled_label_is_excluded(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("config/classification.yaml").read_text(encoding="utf-8"))
    raw["classifications"]["caller_type"]["labels"]["pharmacy"]["enabled"] = False
    target = tmp_path / "classification.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    view = ConfigStore(target).public_view()
    assert "pharmacy" not in view["caller_type"]

