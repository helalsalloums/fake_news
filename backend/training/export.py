from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def export_onnx(checkpoint: Path, output: Path) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    inputs = tokenizer(
        "هذا نص الدليل.",
        "هذا هو الادعاء.",
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "verifier.onnx"
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        target,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
        "calibration.json",
        "training_config.yaml",
        "test_metrics.json",
        "training_metadata.json",
    ):
        source = checkpoint / filename
        if source.exists():
            shutil.copy2(source, output / filename)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export verifier to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/verifier-onnx"))
    args = parser.parse_args()
    print(export_onnx(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
