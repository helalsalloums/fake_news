from datetime import UTC, datetime

import pytest

from app.schemas.domain import (
    ClaimType,
    EvidenceScores,
    ExtractedClaim,
    Passage,
    Verdict,
)
from app.services.aggregation import ConservativeVerdictAggregator, EvaluatedPassage
from app.services.verification import RuleBasedVerifier


def make_passage(identifier: str, text: str, url: str) -> Passage:
    return Passage(
        id=identifier,
        document_url=url,
        title="خبر",
        source="مصدر",
        text=text,
        retrieved_at=datetime.now(UTC),
    )


def strong_scores() -> EvidenceScores:
    return EvidenceScores(
        relevance=0.9,
        source=0.9,
        recency=0.9,
        directness=0.9,
        agreement=1.0,
        overall=0.91,
    )


@pytest.mark.asyncio
async def test_numeric_mismatch_refutes_claim() -> None:
    claim = ExtractedClaim(
        original_text="بلغ عدد الإصابات 500 حالة",
        normalized_text="بلغ عدد الاصابات 500 حالة",
        claim_type=ClaimType.STATISTIC,
        numbers=["500"],
    )
    signal = await RuleBasedVerifier().verify(
        claim, make_passage("a", "بلغ عدد الإصابات 300 حالة فقط.", "https://one.example/a")
    )
    assert signal.model_verdict == Verdict.REFUTED
    assert "CONFLICTING_NUMBERS" in signal.rule_findings


@pytest.mark.asyncio
async def test_conflicting_credible_evidence_forces_nei() -> None:
    claim = ExtractedClaim(
        original_text="بلغ عدد الإصابات 500 حالة",
        normalized_text="بلغ عدد الاصابات 500 حالة",
        claim_type=ClaimType.STATISTIC,
        numbers=["500"],
    )
    verifier = RuleBasedVerifier()
    support_passage = make_passage(
        "support", "أكد البيان أن عدد الإصابات بلغ 500 حالة.", "https://one.example/a"
    )
    refute_passage = make_passage(
        "refute", "أكد البيان أن عدد الإصابات بلغ 300 حالة.", "https://two.example/a"
    )
    support = await verifier.verify(claim, support_passage)
    refute = await verifier.verify(claim, refute_passage)
    result = ConservativeVerdictAggregator().aggregate_claim(
        claim,
        [
            EvaluatedPassage(support_passage, strong_scores(), support),
            EvaluatedPassage(refute_passage, strong_scores(), refute),
        ],
    )
    assert result.verdict == Verdict.NOT_ENOUGH_INFORMATION
    assert result.conflict_detected is True
    assert result.model_confidence >= 0.7
