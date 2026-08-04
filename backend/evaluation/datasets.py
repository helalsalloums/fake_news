from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from training.common import LABEL2ID, normalize_label
from training.datasets import (
    CLAIM_FIELDS,
    EVIDENCE_FIELDS,
    LABEL_FIELDS,
    adapt_arafa_record,
    read_json_records,
)


@dataclass(frozen=True)
class FactCheckingExample:
    claim: str
    evidence: str
    label: int
    source: str


class FactCheckingDataset(Protocol):
    name: str

    def load(self, split: str | None = None) -> list[FactCheckingExample]: ...


class ArafaDataset:
    name = "arafa"

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, split: str | None = None) -> list[FactCheckingExample]:
        return [
            FactCheckingExample(item["claim"], item["evidence"], item["label"], self.name)
            for item in (adapt_arafa_record(record) for record in read_json_records(self.path))
        ]


class PreparedDataset:
    def __init__(self, path: Path, name: str = "prepared") -> None:
        self.path = path
        self.name = name

    def load(self, split: str | None = "test") -> list[FactCheckingExample]:
        from datasets import load_from_disk

        dataset = load_from_disk(str(self.path))[split or "test"]
        return [
            FactCheckingExample(row["claim"], row["evidence"], int(row["label"]), self.name)
            for row in dataset
        ]


class XFactDataset:
    name = "x-fact"

    def __init__(self, path: Path, label_mapping: dict[str, str] | None = None) -> None:
        self.path = path
        self.label_mapping = label_mapping or {
            "TRUE": "SUPPORTED",
            "MOSTLY TRUE": "SUPPORTED",
            "FALSE": "REFUTED",
            "MOSTLY FALSE": "REFUTED",
            "PARTLY FALSE": "NOT_ENOUGH_INFORMATION",
            "MIXTURE": "NOT_ENOUGH_INFORMATION",
            "OTHER": "NOT_ENOUGH_INFORMATION",
        }

    def load(self, split: str | None = None) -> list[FactCheckingExample]:
        return _load_claim_evidence_json(self.path, self.name, self.label_mapping)


class AraFactsDataset(XFactDataset):
    name = "arafacts"

    def __init__(self, path: Path, label_mapping: dict[str, str] | None = None) -> None:
        super().__init__(
            path,
            label_mapping
            or {
                "TRUE": "SUPPORTED",
                "CORRECT": "SUPPORTED",
                "ACCURATE": "SUPPORTED",
                "FALSE": "REFUTED",
                "INCORRECT": "REFUTED",
                "MISLEADING": "NOT_ENOUGH_INFORMATION",
                "PARTLY FALSE": "NOT_ENOUGH_INFORMATION",
                "PARTLY TRUE": "NOT_ENOUGH_INFORMATION",
            },
        )


def _first(record: dict[str, object], fields: tuple[str, ...]) -> object | None:
    for field in fields:
        if record.get(field) not in (None, ""):
            return record[field]
    return None


def _load_claim_evidence_json(
    path: Path, name: str, mapping: dict[str, str]
) -> list[FactCheckingExample]:
    examples: list[FactCheckingExample] = []
    evidence_fields = EVIDENCE_FIELDS + (
        "fact_checking_article",
        "article_content",
        "description",
        "explanation",
    )
    for record in read_json_records(path):
        claim = _first(record, CLAIM_FIELDS)
        evidence = _first(record, evidence_fields)
        raw_label = _first(record, LABEL_FIELDS + ("rating", "normalized_rating"))
        if claim is None or evidence is None or raw_label is None:
            continue
        source_label = str(raw_label).strip().upper().replace("_", " ")
        if source_label not in mapping:
            raise ValueError(f"unmapped {name} label: {raw_label!r}")
        label = LABEL2ID[normalize_label(mapping[source_label])]
        examples.append(FactCheckingExample(str(claim), str(evidence), label, name))
    return examples
