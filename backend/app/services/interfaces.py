from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.schemas.domain import (
    ClaimResult,
    Document,
    ExtractedClaim,
    FactCheckResponse,
    Passage,
    SearchResult,
    VerificationSignal,
)


class ClaimExtractor(Protocol):
    async def extract(self, text: str) -> list[ExtractedClaim]: ...


class QueryGenerator(Protocol):
    def generate(self, claim: ExtractedClaim, limit: int = 4) -> list[str]: ...


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]: ...


class DocumentFetcher(Protocol):
    async def fetch(self, result: SearchResult) -> Document | None: ...


class DocumentCleaner(Protocol):
    def clean(self, html: str, url: str, result: SearchResult) -> Document | None: ...


class EvidenceRetriever(Protocol):
    async def retrieve(
        self, claim: ExtractedClaim, documents: Sequence[Document], limit: int = 8
    ) -> list[Passage]: ...


class EvidenceRanker(Protocol):
    def rank(
        self,
        claim: ExtractedClaim,
        passages: Sequence[Passage],
        claimed_at: datetime | None,
    ) -> list[tuple[Passage, object]]: ...


class FactVerifier(Protocol):
    async def verify(self, claim: ExtractedClaim, passage: Passage) -> VerificationSignal: ...


class VerdictAggregator(Protocol):
    def aggregate_claim(self, claim: ExtractedClaim, evidence: list[object]) -> ClaimResult: ...

    def aggregate_fact_check(self, text: str, claims: list[ClaimResult]) -> FactCheckResponse: ...


class VectorStore(Protocol):
    async def add(self, passages: Sequence[Passage]) -> None: ...

    async def search(self, query: str, limit: int) -> list[tuple[str, float]]: ...


class FactCheckingDataset(Protocol):
    name: str

    def load(self, split: str | None = None) -> object: ...
