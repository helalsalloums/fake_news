from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.schemas.domain import (
    ClassProbabilities,
    ExtractedClaim,
    Passage,
    Verdict,
    VerificationSignal,
)

from app.services.arabic import analyze_arabic, claim_coverage_ratio, normalize_arabic, normalized_tokens, similarity_ratio

NEGATIONS = {"لا", "لم", "لن", "ليس", "ليست"}

import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!؟?؛;])\s+")

def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / max(temperature, 1e-6) for value in values]
    maximum = max(scaled)
    exponents = [math.exp(value - maximum) for value in scaled]
    total = sum(exponents)
    return [value / total for value in exponents]

def _numeric_findings(claim: ExtractedClaim, passage: Passage) -> tuple[Verdict | None, list[str]]:
    passage_numbers = set(analyze_arabic(passage.text).numbers)
    claim_numbers = set(claim.numbers)
    if not claim_numbers:
        return None, []
    passage_tokens = normalized_tokens(passage.text)
    entities_present = (
        all(normalized_tokens(entity).issubset(passage_tokens) for entity in claim.entities)
        if claim.entities
        else True
    )
    if claim_numbers.issubset(passage_numbers) and entities_present:
        return Verdict.SUPPORTED, ["MATCHING_NUMBERS"]
    if passage_numbers and claim_coverage_ratio(claim.normalized_text, passage.text) >= 0.18:
        return Verdict.REFUTED, ["CONFLICTING_NUMBERS"]
    return None, ["CLAIM_NUMBERS_NOT_FOUND"]

def _negation_finding(claim: ExtractedClaim, passage: Passage) -> tuple[Verdict | None, list[str]]:
    claim_words = set(normalize_arabic(claim.original_text).split())
    claim_negated = bool(claim_words & NEGATIONS)
    sentences = [s for s in SENTENCE_SPLIT.split(passage.text) if s.strip()]
    if not sentences:
        return None, []
    best_sentence = max(sentences, key=lambda s: claim_coverage_ratio(claim.normalized_text, s))
    best_overlap = claim_coverage_ratio(claim.normalized_text, best_sentence)
    passage_negated = bool(set(normalize_arabic(best_sentence).split()) & NEGATIONS)
    if best_overlap >= 0.5 and claim_negated != passage_negated:
        return Verdict.REFUTED, ["NEGATION_MISMATCH"]
    return None, []

class RuleBasedVerifier:
    version = "rules-v1"

    async def verify(self, claim: ExtractedClaim, passage: Passage) -> VerificationSignal:
        findings: list[str] = []
        decisions: list[Verdict] = []
        for checker in (_numeric_findings, _negation_finding):
            decision, new_findings = checker(claim, passage)
            findings.extend(new_findings)
            if decision:
                decisions.append(decision)
        relevance = claim_coverage_ratio(claim.normalized_text, passage.text)
        if Verdict.REFUTED in decisions:
            verdict = Verdict.REFUTED
            probabilities = [0.08, 0.84, 0.08]
            confidence = 0.84
        elif Verdict.SUPPORTED in decisions and relevance >= 0.18:
            verdict = Verdict.SUPPORTED
            confidence = min(0.88, 0.68 + relevance)
            probabilities = [confidence, 0.04, 1 - confidence - 0.04]
        elif relevance >= 0.55:
            verdict = Verdict.SUPPORTED
            confidence = min(0.78, 0.55 + relevance * 0.3)
            probabilities = [confidence, 0.05, 1 - confidence - 0.05]
            findings.append("HIGH_LEXICAL_OVERLAP")
        else:
            verdict = Verdict.NOT_ENOUGH_INFORMATION
            confidence = max(0.55, 0.82 - relevance)
            probabilities = [(1 - confidence) * 0.6, (1 - confidence) * 0.4, confidence]
            findings.append("INSUFFICIENT_DIRECT_MATCH")
        return VerificationSignal(
            model_verdict=verdict,
            model_confidence=confidence,
            class_probabilities=ClassProbabilities.model_validate(
                dict(
                    zip(
                        ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFORMATION"),
                        probabilities,
                        strict=True,
                    )
                )
            ),
            model_version=self.version,
            rule_verdict=verdict,
            rule_confidence=confidence,
            rule_findings=findings,
        )


class TransformerNliVerifier:
    """Lazy non-generative NLI inference with optional temperature calibration."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._tokenizer: Any = None
        self._model: Any = None
        calibration_path = Path(model_path) / "calibration.json"
        self.temperature = 1.0
        if calibration_path.exists():
            self.temperature = float(json.loads(calibration_path.read_text())["temperature"])

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()
        return self._tokenizer, self._model

    async def verify(self, claim: ExtractedClaim, passage: Passage) -> VerificationSignal:
        import torch

        tokenizer, model = self._load()
        inputs = tokenizer(
            passage.text, claim.original_text, truncation=True, max_length=384, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[0].detach().cpu().tolist()
        calibrated = softmax(logits, self.temperature)
        id2label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}
        canonical: dict[str, float] = {label: 0.0 for label in Verdict}
        aliases = {
            "ENTAILMENT": Verdict.SUPPORTED,
            "SUPPORTED": Verdict.SUPPORTED,
            "CONTRADICTION": Verdict.REFUTED,
            "REFUTED": Verdict.REFUTED,
            "NEUTRAL": Verdict.NOT_ENOUGH_INFORMATION,
            "NOT_ENOUGH_INFORMATION": Verdict.NOT_ENOUGH_INFORMATION,
        }
        for index, probability in enumerate(calibrated):
            label = aliases.get(id2label[index])
            if label:
                canonical[label.value] = probability
        verdict = max(Verdict, key=lambda label: canonical[label.value])
        rule_signal = await RuleBasedVerifier().verify(claim, passage)
        return VerificationSignal(
            model_verdict=verdict,
            model_confidence=canonical[verdict.value],
            class_probabilities=ClassProbabilities.model_validate(canonical),
            model_version=self.model_path,
            rule_verdict=rule_signal.rule_verdict,
            rule_confidence=rule_signal.rule_confidence,
            rule_findings=rule_signal.rule_findings,
        )
