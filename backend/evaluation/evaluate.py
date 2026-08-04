from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.datasets import PreparedDataset
from training.calibrate import expected_calibration_error
from training.common import ID2LABEL


def evaluate_model(checkpoint: Path, dataset_path: Path, batch_size: int = 16) -> dict[str, object]:
    examples = PreparedDataset(dataset_path).load("test")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    temperature_path = checkpoint / "calibration.json"
    temperature = 1.0
    if temperature_path.exists():
        temperature = float(json.loads(temperature_path.read_text())["temperature"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = tokenizer(
                [item.evidence for item in batch],
                [item.claim for item in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            logits = model(**{key: value.to(device) for key, value in inputs.items()}).logits
            probabilities.extend(torch.softmax(logits / temperature, dim=-1).cpu().numpy())
            labels.extend(item.label for item in batch)
    probability_array = np.asarray(probabilities)
    label_array = np.asarray(labels)
    predictions = probability_array.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        label_array, predictions, labels=list(ID2LABEL), zero_division=0
    )
    per_class = {
        ID2LABEL[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in ID2LABEL
    }
    return {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            label_array, predictions, labels=list(ID2LABEL)
        ).tolist(),
        "expected_calibration_error": expected_calibration_error(probability_array, label_array),
        "brier_score": float(
            np.mean(np.sum((probability_array - np.eye(3)[label_array]) ** 2, axis=1))
        ),
        "count": len(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Arabic evidence verifier")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/verifier"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/datasets/arafa"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_model(args.checkpoint, args.dataset)
    serialized = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
