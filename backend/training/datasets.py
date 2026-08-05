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


def _extract_records(raw: Any) -> list[dict[str, Any]]:
    """Extract a flat list of records from an already-parsed JSON object."""
    if isinstance(raw, list):
        return raw
    for key in ("data", "records", "examples", "claims"):
        if isinstance(raw.get(key), list):
            return raw[key]
    if all(isinstance(value, list) for value in raw.values()):
        return [item for rows in raw.values() for item in rows]
    raise ValueError("unsupported dataset JSON structure")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read + parse a JSON file from disk and extract its records.

    Kept for callers that only have a path. prepare_arafa avoids this and
    reuses its own already-parsed JSON instead, to avoid double-parsing
    large files.
    """
    return _extract_records(json.loads(path.read_text(encoding="utf-8")))


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
    from sklearn.model_selection import train_test_split
    import gc
    import time

    t0 = time.time()

    print("[1/8] Reading JSON...")
    raw_json = json.loads(source.read_text(encoding="utf-8"))
    print(f"      Done in {time.time() - t0:.1f}s")

    official_splits = (
        raw_json
        if isinstance(raw_json, dict)
        and all(isinstance(raw_json.get(name), list) for name in ("train", "validation", "test"))
        else None
    )

    rejected = 0
    seen: set[tuple[str, str]] = set()

    def adapt_rows(records):
        nonlocal rejected

        rows = []
        start = time.time()

        for i, raw in enumerate(records):
            if i and i % 10000 == 0:
                print(f"      {i:,}/{len(records):,} ({time.time()-start:.1f}s)")

            try:
                item = adapt_arafa_record(raw)
            except ValueError:
                rejected += 1
                continue

            key = (
                normalize_arabic(item["claim"]),
                normalize_arabic(item["evidence"]),
            )

            if key in seen:
                continue

            seen.add(key)
            rows.append(item)

        return rows

    print("[2/8] Adapting records...")

    if official_splits is not None:
        splits = {
            name: adapt_rows(official_splits[name])
            for name in ("train", "validation", "test")
        }
    else:
        records = _extract_records(raw_json)
        raw_json = None
        gc.collect()

        adapted = adapt_rows(records)

        records = None
        gc.collect()

        print(f"[3/8] Adapted {len(adapted):,} records")

        if len(adapted) < 30:
            raise ValueError("too few valid records after validation")

        print("[4/8] Splitting dataset...")

        labels = [x["label"] for x in adapted]

        train_rows, holdout = train_test_split(
            adapted,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
        )

        holdout_labels = [x["label"] for x in holdout]

        validation_rows, test_rows = train_test_split(
            holdout,
            test_size=0.50,
            random_state=seed,
            stratify=holdout_labels,
        )

        splits = {
            "train": train_rows,
            "validation": validation_rows,
            "test": test_rows,
        }

    print("[5/8] Creating HuggingFace Dataset...")

    dataset = DatasetDict(
        {
            name: Dataset.from_list(
                [{k: v for k, v in row.items() if k != "group"} for row in rows]
            )
            for name, rows in splits.items()
        }
    )

    print("[6/8] Saving dataset...")

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
            name: {
                "count": len(rows),
                "labels": dict(Counter(row["label_name"] for row in rows)),
            }
            for name, rows in splits.items()
        },
    }

    write_json(output / "split_manifest.json", manifest)

    print(f"[7/8] Finished in {time.time()-t0:.1f}s")

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