import hashlib

import pytest

from app.schemas.domain import Document, SearchResult, Verdict
from app.services.aggregation import ConservativeVerdictAggregator
from app.services.claims import RuleBasedClaimExtractor
from app.services.pipeline import FactCheckingPipeline
from app.services.query import ArabicQueryGenerator
from app.services.retrieval import LexicalEvidenceRetriever
from app.services.scoring import ConfigurableEvidenceRanker, SourceReliability
from app.services.verification import RuleBasedVerifier


class FakeSearch:
    name = "fake"

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return [SearchResult(title="بيان", url="https://health.gov/a", source="وزارة الصحة")]


class FakeFetcher:
    async def fetch(self, result: SearchResult) -> Document:
        text = "أكدت وزارة الصحة في بيان رسمي أن عدد الإصابات بلغ 500 حالة."
        return Document(
            url=result.url,
            title=result.title,
            source=result.source,
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )


@pytest.mark.asyncio
async def test_complete_pipeline_with_mocked_external_services() -> None:
    pipeline = FactCheckingPipeline(
        RuleBasedClaimExtractor(),
        ArabicQueryGenerator(),
        FakeSearch(),
        FakeFetcher(),
        LexicalEvidenceRetriever(),
        ConfigurableEvidenceRanker(SourceReliability()),
        RuleBasedVerifier(),
        ConservativeVerdictAggregator(evidence_threshold=0.55),
    )
    result = await pipeline.run("أعلنت وزارة الصحة أن عدد الإصابات بلغ 500 حالة")
    assert result.verdict == Verdict.SUPPORTED
    assert result.claims[0].model_confidence >= 0.7
    assert result.claims[0].class_probabilities.supported >= 0.7
    assert result.evidence
