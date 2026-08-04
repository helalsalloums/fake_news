from datetime import UTC, datetime

import pytest

from app.schemas.domain import ClaimType, ExtractedClaim, Passage
from app.services.scoring import ConfigurableEvidenceRanker, ScoringWeights, SourceReliability


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(relevance=0.9).validate()


def test_direct_reliable_evidence_scores_highly() -> None:
    claim = ExtractedClaim(
        original_text="بلغ عدد الإصابات 500 حالة",
        normalized_text="بلغ عدد الاصابات 500 حالة",
        claim_type=ClaimType.STATISTIC,
        numbers=["500"],
    )
    passage = Passage(
        id="one",
        document_url="https://health.gov/test",
        source="وزارة الصحة",
        text="أكد البيان الرسمي أن عدد الإصابات بلغ 500 حالة.",
        published_at=datetime.now(UTC),
    )
    scored = ConfigurableEvidenceRanker(SourceReliability()).rank(claim, [passage], None)
    assert scored[0][1].source == 1.0
    assert scored[0][1].overall >= 0.65
