from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from app.schemas.domain import Document, ExtractedClaim, Passage
from app.services.arabic import normalize_arabic, normalized_tokens
from app.services.vector import FaissVectorStore, SentenceTransformerEncoder

SENTENCE_SPLIT = re.compile(r"(?<=[.!؟?؛;])\s+|\n+")


def split_document(
    document: Document, max_chars: int = 1200, overlap_sentences: int = 1
) -> list[Passage]:
    sentences = [item.strip() for item in SENTENCE_SPLIT.split(document.text) if item.strip()]
    passages: list[Passage] = []
    current: list[str] = []
    for sentence in sentences:
        if current and sum(len(item) for item in current) + len(sentence) > max_chars:
            text = " ".join(current)
            index = len(passages)
            passages.append(
                Passage(
                    id=hashlib.sha256(f"{document.url}:{index}:{text}".encode()).hexdigest()[:20],
                    document_url=document.url,
                    title=document.title,
                    source=document.source,
                    text=text,
                    published_at=document.published_at,
                    retrieved_at=document.retrieved_at,
                    passage_index=index,
                )
            )
            current = current[-overlap_sentences:]
        current.append(sentence)
    if current:
        text = " ".join(current)
        index = len(passages)
        passages.append(
            Passage(
                id=hashlib.sha256(f"{document.url}:{index}:{text}".encode()).hexdigest()[:20],
                document_url=document.url,
                title=document.title,
                source=document.source,
                text=text,
                published_at=document.published_at,
                retrieved_at=document.retrieved_at,
                passage_index=index,
            )
        )
    return passages


def bm25_scores(
    query: str, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    query_tokens = list(normalized_tokens(query))
    documents = [normalize_arabic(item.text).split() for item in passages]
    if not query_tokens or not documents:
        return [0.0] * len(passages)
    average_length = sum(map(len, documents)) / len(documents)
    document_frequency = Counter(
        token for document in documents for token in set(document) if token in query_tokens
    )
    raw: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(document) / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        raw.append(score)
    maximum = max(raw, default=0.0)
    return [value / maximum if maximum else 0.0 for value in raw]


class LexicalEvidenceRetriever:
    async def retrieve(
        self, claim: ExtractedClaim, documents: Sequence[Document], limit: int = 8
    ) -> list[Passage]:
        passages = [passage for document in documents for passage in split_document(document)]
        scores = bm25_scores(claim.normalized_text, passages)
        ranked = sorted(zip(passages, scores, strict=True), key=lambda item: item[1], reverse=True)
        return [passage for passage, _ in ranked[:limit]]


class HybridEvidenceRetriever:
    def __init__(self, embedding_model: str, dense_weight: float = 0.6) -> None:
        self.store = FaissVectorStore(SentenceTransformerEncoder(embedding_model))
        self.dense_weight = dense_weight

    async def retrieve(
        self, claim: ExtractedClaim, documents: Sequence[Document], limit: int = 8
    ) -> list[Passage]:
        passages = [passage for document in documents for passage in split_document(document)]
        if not passages:
            return []
        lexical = dict(
            zip(
                (item.id for item in passages),
                bm25_scores(claim.normalized_text, passages),
                strict=True,
            )
        )
        await self.store.add(passages)
        dense = dict(await self.store.search(claim.normalized_text, len(passages)))
        combined = {
            item.id: self.dense_weight * dense.get(item.id, 0.0)
            + (1.0 - self.dense_weight) * lexical.get(item.id, 0.0)
            for item in passages
        }
        return sorted(passages, key=lambda item: combined[item.id], reverse=True)[:limit]
