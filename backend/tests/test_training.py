import pytest

from training.common import LABEL2ID, normalize_label
from training.datasets import adapt_arafa_record


def test_training_label_mapping_is_stable() -> None:
    assert LABEL2ID == {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_INFORMATION": 2}
    assert normalize_label("neutral") == "NOT_ENOUGH_INFORMATION"


def test_arafa_adapter_accepts_common_schema() -> None:
    result = adapt_arafa_record(
        {"claim": "العدد 500", "evidence": "بلغ العدد 500", "label": "supported"}
    )
    assert result["label"] == 0
    assert result["group"]


def test_arafa_adapter_rejects_missing_evidence() -> None:
    with pytest.raises(ValueError):
        adapt_arafa_record({"claim": "ادعاء", "label": "supported"})
