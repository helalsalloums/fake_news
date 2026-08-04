import pytest

from app.schemas.domain import ClaimType
from app.services.claims import RuleBasedClaimExtractor
from app.services.query import ArabicQueryGenerator


@pytest.mark.asyncio
async def test_extracts_multiple_statistical_claims() -> None:
    text = "قالت وزارة الصحة إن عدد الإصابات بلغ 500 حالة، وأضافت أن 20 شخصاً دخلوا المستشفى."
    claims = await RuleBasedClaimExtractor().extract(text)
    assert len(claims) == 2
    assert all(claim.claim_type == ClaimType.STATISTIC for claim in claims)
    assert claims[0].numbers == ["500"]
    assert claims[1].numbers == ["20"]


@pytest.mark.asyncio
async def test_query_generation_keeps_numbers() -> None:
    claim = (
        await RuleBasedClaimExtractor().extract("أعلنت وزارة الصحة أن الإصابات بلغت ٥٠٠ حالة.")
    )[0]
    queries = ArabicQueryGenerator().generate(claim)
    assert any('"500"' in query for query in queries)
    assert len(queries) == len(set(queries))
