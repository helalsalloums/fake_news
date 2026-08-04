from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.services.arabic import normalize_arabic
from training.common import LABEL2ID, file_checksum, normalize_label, write_json

CLAIM_FIELDS = ("claim", "claim_text", "statement", "hypothesis")
EVIDENCE_FIELDS = ("evidence", "evidence_text", "context", "premise", "passage")
LABEL_FIELDS = (
    "label",
    "labels",
    "verdict",
    "class",
    "stance",
    "judgement",
    "judgment",
)

GROUP_FIELDS = ("source_id", "article_id", "page_id", "document_id", "source_url", "url")


def _first(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def read_json_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    for key in ("data", "records", "examples", "claims"):
        if isinstance(raw.get(key), list):
            return raw[key]
    if all(isinstance(value, list) for value in raw.values()):
        return [item for rows in raw.values() for item in rows]
    raise ValueError("unsupported dataset JSON structure")


def adapt_arafa_record(record: dict[str, Any]) -> dict[str, Any]:
    claim = _first(record, CLAIM_FIELDS)
    evidence = _first(record, EVIDENCE_FIELDS)
    label = _first(record, LABEL_FIELDS)
    if isinstance(evidence, list):
        evidence = " ".join(str(item) for item in evidence)
    if claim is None or evidence is None or label is None:
        raise ValueError("record is missing claim, evidence, or label")
    normalized_claim = normalize_arabic(str(claim))
    normalized_evidence = normalize_arabic(str(evidence))
    if not normalized_claim or not normalized_evidence:
        raise ValueError("record contains empty normalized text")
    group = _first(record, GROUP_FIELDS)
    if group is None:
        group = normalized_evidence
    canonical_label = normalize_label(label)
    return {
        "claim": str(claim).strip(),
        "evidence": str(evidence).strip(),
        "label": LABEL2ID[canonical_label],
        "label_name": canonical_label,
        "group": normalize_arabic(str(group)),
    }


def prepare_arafa(source: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import GroupShuffleSplit

    raw_json = json.loads(source.read_text(encoding="utf-8"))
    official_splits = (
        raw_json
        if isinstance(raw_json, dict)
        and all(isinstance(raw_json.get(name), list) for name in ("train", "validation", "test"))
        else None
    )
    adapted: list[dict[str, Any]] = []
    rejected = 0
    seen: set[tuple[str, str]] = set()

    def adapt_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal rejected
        output_rows: list[dict[str, Any]] = []
        for raw in records:
            try:
                item = adapt_arafa_record(raw)
            except ValueError:
                rejected += 1
                continue
            key = (normalize_arabic(item["claim"]), normalize_arabic(item["evidence"]))
            if key in seen:
                continue
            seen.add(key)
            output_rows.append(item)
        return output_rows

    if official_splits is not None:
        splits = {
            name: adapt_rows(official_splits[name]) for name in ("train", "validation", "test")
        }
        adapted = [row for rows in splits.values() for row in rows]
    else:
        adapted = adapt_rows(read_json_records(source))

    if len(adapted) < 30:
        raise ValueError("too few valid records after validation")

    if official_splits is None:
        groups = [item["group"] for item in adapted]
        first_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_indexes, holdout_indexes = next(first_split.split(adapted, groups=groups))
        holdout = [adapted[index] for index in holdout_indexes]
        holdout_groups = [item["group"] for item in holdout]
        second_split = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        validation_local, test_local = next(second_split.split(holdout, groups=holdout_groups))
        splits = {
            "train": [adapted[index] for index in train_indexes],
            "validation": [holdout[index] for index in validation_local],
            "test": [holdout[index] for index in test_local],
        }
    dataset = DatasetDict(
        {
            name: Dataset.from_list(
                [{k: v for k, v in row.items() if k != "group"} for row in rows]
            )
            for name, rows in splits.items()
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    manifest = {
        "dataset": "ARAFA",
        "source": str(source),
        "source_sha256": file_checksum(source),
        "seed": seed,
        "used_official_splits": official_splits is not None,
        "rejected_records": rejected,
        "splits": {
            name: {"count": len(rows), "labels": dict(Counter(row["label_name"] for row in rows))}
            for name, rows in splits.items()
        },
    }
    write_json(output / "split_manifest.json", manifest)
    return manifest


def assert_no_split_leakage(dataset_path: Path) -> None:
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_path))
    hashes: dict[str, set[str]] = defaultdict(set)
    for split, rows in dataset.items():
        for row in rows:
            key = f"{normalize_arabic(row['claim'])}\0{normalize_arabic(row['evidence'])}"
            hashes[split].add(key)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = hashes[left] & hashes[right]
        if overlap:
            raise ValueError(
                f"dataset leakage detected between {left} and {right}: {len(overlap)} pairs"
            )
