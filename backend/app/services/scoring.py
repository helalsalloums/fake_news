from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from app.schemas.domain import EvidenceScores, ExtractedClaim, Passage
from app.services.arabic import claim_coverage_ratio, normalized_tokens, similarity_ratio

@dataclass(frozen=True)
class ScoringWeights:
    relevance: float = 0.40
    source: float = 0.25
    recency: float = 0.15
    directness: float = 0.10
    agreement: float = 0.10

    def validate(self) -> None:
        if not math.isclose(sum(self.__dict__.values()), 1.0, abs_tol=1e-6):
            raise ValueError("evidence scoring weights must sum to 1.0")

    @classmethod
    def from_yaml(cls, path: Path | None) -> ScoringWeights:
        if path is None or not path.exists():
            return cls()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = loaded.get("weights", {}) if isinstance(loaded, dict) else {}
        result = cls(**values)
        result.validate()
        return result


class SourceReliability:
    def __init__(self, config_path: Path | None = None) -> None:
        data: dict[str, Any] = {}
        if config_path and config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        self.categories = {
            "government": 1.0,
            "official_organization": 0.95,
            "major_news": 0.85,
            "established_media": 0.80,
            "blog": 0.40,
            "unknown": 0.50,
            "social_media": 0.20,
            **dict(data.get("categories", {})),
        }
        self.domain_categories: dict[str, str] = dict(data.get("domain_categories", {}))
        self.government_suffixes: tuple[str, ...] = tuple(data.get("government_suffixes", [".gov"]))
        self.social_domains: set[str] = set(data.get("social_domains", []))
        self.official_domains: set[str] = set(data.get("official_domains", []))
        self.news_domains: set[str] = set(data.get("news_domains", []))
        self.blog_domains: set[str] = set(data.get("blog_domains", []))

    def category(self, url: str) -> str:
        domain = (urlparse(url).hostname or "").lower().removeprefix("www.")
        if domain in self.domain_categories:
            return str(self.domain_categories[domain])
        if any(domain == item or domain.endswith(f".{item}") for item in self.social_domains):
            return "social_media"
        if any(domain.endswith(suffix) for suffix in self.government_suffixes):
            return "government"
        if any(domain == item or domain.endswith(f".{item}") for item in self.official_domains):
            return "official_organization"
        if any(domain == item or domain.endswith(f".{item}") for item in self.news_domains):
            return "established_media"
        if any(domain == item or domain.endswith(f".{item}") for item in self.blog_domains):
            return "blog"
        if "news" in domain.split("."):
            return "established_media"
        return "unknown"

    def score(self, url: str) -> float:
        return float(self.categories.get(self.category(url), self.categories["unknown"]))


def temporal_score(published_at: datetime | None, claimed_at: datetime | None) -> float:
    if published_at is None:
        return 0.45
    reference = claimed_at or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    published = published_at
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    distance_days = abs((published - reference).total_seconds()) / 86_400
    if distance_days <= 2:
        return 1.0
    if distance_days <= 30:
        return 0.85
    if distance_days <= 365:
        return 0.65
    return 0.40


def directness_score(claim: ExtractedClaim, passage: Passage) -> float:
    passage_tokens = normalized_tokens(passage.text)
    entity_hits = sum(
        1 for entity in claim.entities if normalized_tokens(entity).issubset(passage_tokens)
    )
    number_hits = sum(
        1
        for number in claim.numbers
        if number in passage.text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    )

    components = [claim_coverage_ratio(claim.normalized_text, passage.text)]

    if claim.entities:
        components.append(entity_hits / len(claim.entities))
    if claim.numbers:
        components.append(number_hits / len(claim.numbers))
    return min(1.0, sum(components) / len(components) * 1.4)


class ConfigurableEvidenceRanker:
    def __init__(
        self,
        source_reliability: SourceReliability,
        weights: ScoringWeights | None = None,
    ) -> None:
        self.source_reliability = source_reliability
        self.weights = weights or ScoringWeights()
        self.weights.validate()

    def rank(
        self,
        claim: ExtractedClaim,
        passages: list[Passage],
        claimed_at: datetime | None,
    ) -> list[tuple[Passage, EvidenceScores]]:
        domains = [(urlparse(item.document_url).hostname or "").lower() for item in passages]
        domain_counts = Counter(domains)
        output: list[tuple[Passage, EvidenceScores]] = []
        for passage, domain in zip(passages, domains, strict=True):
            relevance = min(1.0, claim_coverage_ratio(claim.normalized_text, passage.text) * 1.4)
            source = self.source_reliability.score(passage.document_url)
            recency = temporal_score(passage.published_at, claimed_at)
            directness = directness_score(claim, passage)
            agreement = 1.0 / max(1, domain_counts[domain])
            overall = (
                relevance * self.weights.relevance
                + source * self.weights.source
                + recency * self.weights.recency
                + directness * self.weights.directness
                + agreement * self.weights.agreement
            )
            output.append(
                (
                    passage,
                    EvidenceScores(
                        relevance=relevance,
                        source=source,
                        recency=recency,
                        directness=directness,
                        agreement=agreement,
                        overall=overall,
                    ),
                )
            )
        return sorted(output, key=lambda item: item[1].overall, reverse=True)
