from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from urllib.parse import urlparse

from app.schemas.domain import (
    ClaimResult,
    ClassProbabilities,
    ConfidenceBasis,
    EvidenceItem,
    EvidenceScores,
    EvidenceStance,
    ExtractedClaim,
    FactCheckResponse,
    Passage,
    Verdict,
    VerificationSignal,
)


@dataclass(frozen=True)
class EvaluatedPassage:
    passage: Passage
    scores: EvidenceScores
    signal: VerificationSignal


def _probabilities(signal: VerificationSignal) -> dict[Verdict, float]:
    values = signal.class_probabilities
    return {
        Verdict.SUPPORTED: values.supported,
        Verdict.REFUTED: values.refuted,
        Verdict.NOT_ENOUGH_INFORMATION: values.not_enough_information,
    }


class ConservativeVerdictAggregator:
    def __init__(
        self,
        model_threshold: float = 0.70,
        margin_threshold: float = 0.20,
        evidence_threshold: float = 0.65,
    ) -> None:
        self.model_threshold = model_threshold
        self.margin_threshold = margin_threshold
        self.evidence_threshold = evidence_threshold

    def _stance(self, item: EvaluatedPassage) -> EvidenceStance:
        probabilities = _probabilities(item.signal)
        ordered = sorted(probabilities.values(), reverse=True)
        strong = (
            item.signal.model_confidence >= self.model_threshold
            and ordered[0] - ordered[1] >= self.margin_threshold
            and item.scores.overall >= self.evidence_threshold
        )
        if not strong:
            return EvidenceStance.NEUTRAL
        if item.signal.model_verdict == Verdict.SUPPORTED:
            return EvidenceStance.SUPPORTING
        if item.signal.model_verdict == Verdict.REFUTED:
            return EvidenceStance.CONTRADICTING
        return EvidenceStance.NEUTRAL

    def aggregate_claim(
        self, claim: ExtractedClaim, evaluated: list[EvaluatedPassage]
    ) -> ClaimResult:
        evidence: list[EvidenceItem] = []
        for item in evaluated:
            stance = self._stance(item)
            evidence.append(
                EvidenceItem(
                    id=item.passage.id,
                    title=item.passage.title,
                    url=item.passage.document_url,
                    snippet=item.passage.text[:500],
                    source=item.passage.source
                    or (urlparse(item.passage.document_url).hostname or ""),
                    published_at=item.passage.published_at,
                    retrieved_at=item.passage.retrieved_at,
                    stance=stance,
                    scores=item.scores,
                    verification=item.signal,
                )
            )

        supporting = [item for item in evidence if item.stance == EvidenceStance.SUPPORTING]
        contradicting = [item for item in evidence if item.stance == EvidenceStance.CONTRADICTING]
        conflict = bool(supporting and contradicting)
        best_evaluated = evaluated[0] if evaluated else None
        if best_evaluated:
            model_verdict = best_evaluated.signal.model_verdict
            model_confidence = best_evaluated.signal.model_confidence
            probabilities = best_evaluated.signal.class_probabilities
        else:
            model_verdict = Verdict.NOT_ENOUGH_INFORMATION
            model_confidence = 0.55
            probabilities = ClassProbabilities.model_validate(
                {"SUPPORTED": 0.225, "REFUTED": 0.225, "NOT_ENOUGH_INFORMATION": 0.55}
            )

        quality = fmean(item.scores.overall for item in evidence[:3]) if evidence else 0.0
        if conflict:
            verdict = Verdict.NOT_ENOUGH_INFORMATION
            confidence = min(0.85, 0.60 + 0.15 * min(len(supporting), len(contradicting)))
            basis = ConfidenceBasis.CONFLICTING_CREDIBLE_EVIDENCE
            explanation = "توجد أدلة موثوقة متعارضة، لذلك لا يمكن إصدار حكم حاسم حالياً."
        elif supporting:
            verdict = Verdict.SUPPORTED
            chosen = max(supporting, key=lambda item: item.scores.overall)
            confidence = 0.55 * chosen.verification.model_confidence + 0.45 * chosen.scores.overall
            basis = (
                ConfidenceBasis.CORROBORATED_EVIDENCE
                if len({urlparse(item.url).hostname for item in supporting}) > 1
                else ConfidenceBasis.DIRECT_EVIDENCE
            )
            explanation = "تؤكد الأدلة المتاحة مضمون الادعاء بشكل مباشر."
        elif contradicting:
            verdict = Verdict.REFUTED
            chosen = max(contradicting, key=lambda item: item.scores.overall)
            confidence = 0.55 * chosen.verification.model_confidence + 0.45 * chosen.scores.overall
            basis = ConfidenceBasis.DIRECT_EVIDENCE
            explanation = "تتعارض الأدلة المتاحة مباشرةً مع مضمون الادعاء."
        else:
            verdict = Verdict.NOT_ENOUGH_INFORMATION
            confidence = min(0.65, 0.45 + 0.20 * (1.0 - quality))
            basis = ConfidenceBasis.INSUFFICIENT_EVIDENCE
            explanation = "لا تتوافر أدلة مباشرة وموثوقة كافية لتأييد الادعاء أو تفنيده حالياً."

        return ClaimResult(
            claim=claim.original_text,
            claim_type=claim.claim_type,
            verdict=verdict,
            confidence=round(confidence, 4),
            confidence_basis=basis,
            model_verdict=model_verdict,
            model_confidence=round(model_confidence, 4),
            class_probabilities=probabilities,
            evidence_quality=round(quality, 4),
            conflict_detected=conflict,
            evidence=evidence,
            explanation=explanation,
        )

    def aggregate_fact_check(self, text: str, claims: list[ClaimResult]) -> FactCheckResponse:
        refuted = [
            item for item in claims if item.verdict == Verdict.REFUTED and item.confidence >= 0.70
        ]
        all_supported = bool(claims) and all(item.verdict == Verdict.SUPPORTED for item in claims)
        conflict = any(item.conflict_detected for item in claims)
        if refuted:
            verdict = Verdict.REFUTED
            relevant = refuted
            basis = ConfidenceBasis.DIRECT_EVIDENCE
            explanation = (
                "استناداً إلى الأدلة المتاحة، يحتوي النص على ادعاء واحد على الأقل تعارضه الأدلة."
            )
        elif all_supported:
            verdict = Verdict.SUPPORTED
            relevant = claims
            basis = ConfidenceBasis.CORROBORATED_EVIDENCE
            explanation = "استناداً إلى الأدلة المتاحة، تدعم المصادر الادعاءات المستخرجة من النص."
        else:
            verdict = Verdict.NOT_ENOUGH_INFORMATION
            relevant = claims
            basis = (
                ConfidenceBasis.CONFLICTING_CREDIBLE_EVIDENCE
                if conflict
                else ConfidenceBasis.INSUFFICIENT_EVIDENCE
            )
            explanation = (
                "استناداً إلى الأدلة المتاحة، لا يمكن التوصل إلى حكم حاسم على جميع الادعاءات."
            )
        weights = [max(item.evidence_quality, 0.1) for item in relevant]
        confidence = (
            sum(item.confidence * weight for item, weight in zip(relevant, weights, strict=True))
            / sum(weights)
            if relevant
            else 0.5
        )
        all_evidence = {item.id: item for claim in claims for item in claim.evidence}
        return FactCheckResponse(
            verdict=verdict,
            confidence=round(confidence, 4),
            confidence_basis=basis,
            claims=claims,
            evidence=list(all_evidence.values()),
            explanation=explanation,
            conflict_detected=conflict,
            metadata={"input_length": len(text), "claim_count": len(claims)},
        )
