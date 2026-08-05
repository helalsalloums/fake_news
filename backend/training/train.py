from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from training.common import ID2LABEL, LABEL2ID, TrainingConfig, write_json
from training.datasets import assert_no_split_leakage


def compute_metrics(prediction: Any) -> dict[str, float]:
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average=None, labels=list(ID2LABEL), zero_division=0
    )
    output: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    for index, label in ID2LABEL.items():
        output[f"f1_{label.lower()}"] = float(f1[index])
    return output


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        labels = inputs["labels"]

        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k != "labels"
        }

        outputs = model(**model_inputs)

        loss = functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )

        return (loss, outputs) if return_outputs else loss


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    values = [math.sqrt(total / (len(LABEL2ID) * counts[index])) for index in range(len(LABEL2ID))]
    mean = sum(values) / len(values)
    return torch.tensor([value / mean for value in values], dtype=torch.float32)


def run_training(config: TrainingConfig) -> Path:
    from datasets import load_from_disk

    dataset_path = Path(config.dataset_path)
    assert_no_split_leakage(dataset_path)
    dataset = load_from_disk(str(dataset_path))
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["evidence"],
            batch["claim"],
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = dataset.map(
        tokenize, batched=True, remove_columns=["claim", "evidence", "label_name"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    output = Path(config.output_dir)
    arguments = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        bf16=config.bf16,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=config.save_total_limit,
        report_to=["tensorboard"],
        seed=config.seed,
        data_seed=config.seed,
        max_grad_norm=1.0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
        class_weights=class_weights(dataset["train"]["label"]),
        callbacks=[EarlyStoppingCallback(config.early_stopping_patience)],
    )

    trainer.train()

    final_path = output / "best"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    write_json(final_path / "test_metrics.json", metrics)

    config.dump(final_path / "training_config.yaml")

    (final_path / "training_metadata.json").write_text(
        json.dumps(
            {
                "base_model": config.base_model,
                "dataset": "ARAFA",
                "license_notice": "Research/non-commercial use; review CC-BY-NC-SA-4.0.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (final_path / "README.md").write_text(
        "# Arabic Evidence Verifier\n\n"
        "A non-generative three-class claim–evidence classifier fine-tuned on ARAFA.\n\n"
        "## Labels\n\n`SUPPORTED`, `REFUTED`, and `NOT_ENOUGH_INFORMATION`.\n\n"
        "## Intended use\n\nResearch evaluation of Arabic evidence relationships. "
        "This model does not "
        "determine objective truth and must be combined with retrieved, cited evidence.\n\n"
        "## License and data\n\nARAFA is CC-BY-NC-SA-4.0. Treat these weights as "
        "research-only/non-commercial unless qualified legal review determines otherwise.\n\n"
        "## Limitations\n\nARAFA is synthetic and Wikipedia-derived. Evaluate on natural "
        "Arabic claims, dialects, temporal statements, quantities, and negation "
        "before deployment.\n",
        encoding="utf-8",
    )

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Arabic evidence verifier")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    result = run_training(TrainingConfig.load(args.config))
    print(result)


if __name__ == "__main__":
    main()