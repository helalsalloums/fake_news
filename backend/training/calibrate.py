from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from training.common import write_json


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidences > lower) & (confidences <= upper)
        if selected.any():
            accuracy = (predictions[selected] == labels[selected]).mean()
            error += selected.mean() * abs(accuracy - confidences[selected].mean())
    return float(error)


def collect_logits(model_path: Path, dataset_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    from datasets import load_from_disk

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = load_from_disk(str(dataset_path))["validation"]

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["evidence"], batch["claim"], truncation=True, max_length=384)

    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=["claim", "evidence", "label_name"]
    )
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits.cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10.0))


def calibrate(model_path: Path, dataset_path: Path) -> dict[str, float]:
    logits, labels = collect_logits(model_path, dataset_path)
    temperature = fit_temperature(logits, labels)
    before = torch.softmax(logits, dim=-1).numpy()
    after = torch.softmax(logits / temperature, dim=-1).numpy()
    label_values = labels.numpy()
    result = {
        "temperature": temperature,
        "ece_before": expected_calibration_error(before, label_values),
        "ece_after": expected_calibration_error(after, label_values),
        "nll_before": float(log_loss(label_values, before)),
        "nll_after": float(log_loss(label_values, after)),
        "brier_after": float(np.mean(np.sum((after - np.eye(3)[label_values]) ** 2, axis=1))),
    }
    write_json(model_path / "calibration.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-calibrate a verifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(calibrate(args.checkpoint, args.dataset))


if __name__ == "__main__":
    main()
