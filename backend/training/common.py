from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

LABEL2ID = {"SUPPORTED": 0, "REFUTED": 1, "NOT_ENOUGH_INFORMATION": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
LABEL_ALIASES = {
    "SUPPORTED": "SUPPORTED",
    "SUPPORT": "SUPPORTED",
    "ENTAILMENT": "SUPPORTED",
    "TRUE": "SUPPORTED",
    "REFUTED": "REFUTED",
    "REFUTE": "REFUTED",
    "CONTRADICTION": "REFUTED",
    "FALSE": "REFUTED",
    "NEI": "NOT_ENOUGH_INFORMATION",
    "NEUTRAL": "NOT_ENOUGH_INFORMATION",
    "NOT ENOUGH INFORMATION": "NOT_ENOUGH_INFORMATION",
    "NOT_ENOUGH_INFORMATION": "NOT_ENOUGH_INFORMATION",
}


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    dataset_path: str
    output_dir: str
    max_length: int = 384
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    seed: int = 42
    early_stopping_patience: int = 2
    save_total_limit: int = 2
    logging_steps: int = 50
    eval_steps: int = 500

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    def dump(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


def normalize_label(value: object) -> str:
    normalized = str(value).strip().upper().replace("-", "_")
    try:
        return LABEL_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"unknown fact-checking label: {value!r}") from error


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
