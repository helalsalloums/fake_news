from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from datetime import datetime
from urllib.parse import urlparse

import structlog

from app.schemas.domain import Document, FactCheckResponse, SearchResult
from app.services.aggregation import ConservativeVerdictAggregator, EvaluatedPassage
from app.services.interfaces import (
    ClaimExtractor,
    DocumentFetcher,
    EvidenceRetriever,
    FactVerifier,
    QueryGenerator,
    SearchProvider,
)
from app.services.scoring import ConfigurableEvidenceRanker

logger = structlog.get_logger(__name__)


class FactCheckingPipeline:
    def __init__(
        self,
        extractor: ClaimExtractor,
        query_generator: QueryGenerator,
        search_provider: SearchProvider,
        fetcher: DocumentFetcher,
        retriever: EvidenceRetriever,
        ranker: ConfigurableEvidenceRanker,
        verifier: FactVerifier,
        aggregator: ConservativeVerdictAggregator,
    ) -> None:
        self.extractor = extractor
        self.query_generator = query_generator
        self.search_provider = search_provider
        self.fetcher = fetcher
        self.retriever = retriever
        self.ranker = ranker
        self.verifier = verifier
        self.aggregator = aggregator

    async def _search(self, queries: Sequence[str]) -> list[SearchResult]:
        batches = await asyncio.gather(
            *(self.search_provider.search(query, limit=10) for query in queries),
            return_exceptions=True,
        )
        unique: dict[str, SearchResult] = {}
        for batch in batches:
            if isinstance(batch, BaseException):
                await logger.awarning(
                    "search_failed", provider=self.search_provider.name, error=type(batch).__name__
                )
                continue
            for item in batch:
                unique.setdefault(item.url, item)
        return list(unique.values())

    async def _documents(self, results: Sequence[SearchResult]) -> list[Document]:
        fetched = await asyncio.gather(
            *(self.fetcher.fetch(item) for item in results[:12]), return_exceptions=True
        )
        documents: dict[str, Document] = {}
        for result, value in zip(results[:12], fetched, strict=True):
            if isinstance(value, BaseException):
                await logger.ainfo(
                    "document_fetch_skipped", url=result.url, error=type(value).__name__
                )
                continue
            if value is not None:
                documents.setdefault(value.content_hash, value)
            elif self.search_provider.name == "local" and len(result.snippet) >= 40:
                digest = hashlib.sha256(result.snippet.encode("utf-8")).hexdigest()
                documents.setdefault(
                    digest,
                    Document(
                        url=result.url,
                        title=result.title,
                        source=result.source,
                        text=result.snippet,
                        published_at=result.published_at,
                        content_hash=digest,
                    ),
                )
        return list(documents.values())

    async def run(self, text: str, claimed_at: datetime | None = None) -> FactCheckResponse:
        started = time.perf_counter()
        extracted = await self.extractor.extract(text)
        results = []
        for claim in extracted:
            queries = self.query_generator.generate(claim)
            search_results = await self._search(queries)
            documents = await self._documents(search_results)
            passages = await self.retriever.retrieve(claim, documents)
            ranked = self.ranker.rank(claim, passages, claimed_at)
            evaluated: list[EvaluatedPassage] = []
            for passage, scores in ranked[:5]:
                signal = await self.verifier.verify(claim, passage)
                evaluated.append(EvaluatedPassage(passage, scores, signal))
            stance_domains: dict[str, set[str]] = {}
            for item in evaluated:
                domain = urlparse(item.passage.document_url).hostname or item.passage.document_url
                stance_domains.setdefault(item.signal.model_verdict.value, set()).add(domain)
            all_domains = set().union(*stance_domains.values()) if stance_domains else set()
            for item in evaluated:
                agreement = len(stance_domains[item.signal.model_verdict.value]) / max(
                    len(all_domains), 1
                )
                item.scores.overall += self.ranker.weights.agreement * (
                    agreement - item.scores.agreement
                )
                item.scores.agreement = agreement
            evaluated.sort(key=lambda item: item.scores.overall, reverse=True)
            results.append(self.aggregator.aggregate_claim(claim, evaluated))
        response = self.aggregator.aggregate_fact_check(text, results)
        response.metadata.update(
            {
                "search_provider": self.search_provider.name,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        await logger.ainfo(
            "fact_check_completed",
            fact_check_id=str(response.id),
            claim_count=len(results),
            verdict=response.verdict,
            confidence=response.confidence,
            latency_ms=response.metadata["latency_ms"],
        )
        return response
