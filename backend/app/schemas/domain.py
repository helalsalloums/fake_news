from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFORMATION = "NOT_ENOUGH_INFORMATION"


class ClaimType(StrEnum):
    STATISTIC = "STATISTIC"
    EVENT = "EVENT"
    ATTRIBUTION = "ATTRIBUTION"
    DATE = "DATE"
    LOCATION = "LOCATION"
    OTHER = "OTHER"


class EvidenceStance(StrEnum):
    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    NEUTRAL = "NEUTRAL"


class ConfidenceBasis(StrEnum):
    CORROBORATED_EVIDENCE = "CORROBORATED_EVIDENCE"
    DIRECT_EVIDENCE = "DIRECT_EVIDENCE"
    CONFLICTING_CREDIBLE_EVIDENCE = "CONFLICTING_CREDIBLE_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MODEL_RULE_DISAGREEMENT = "MODEL_RULE_DISAGREEMENT"
    RULE_BASED_FALLBACK = "RULE_BASED_FALLBACK"


class ExtractedClaim(BaseModel):
    original_text: str
    normalized_text: str
    claim_type: ClaimType
    sentence_index: int = 0
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    importance: float = Field(default=1.0, ge=0, le=1)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published_at: datetime | None = None
    provider: str = "local"
    rank: int = 0


class Document(BaseModel):
    url: str
    title: str = ""
    source: str = ""
    text: str
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_hash: str


class Passage(BaseModel):
    id: str
    document_url: str
    title: str = ""
    source: str = ""
    text: str
    published_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    passage_index: int = 0


class EvidenceScores(BaseModel):
    relevance: float = Field(ge=0, le=1)
    source: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    directness: float = Field(ge=0, le=1)
    agreement: float = Field(default=0.5, ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class ClassProbabilities(BaseModel):
    supported: float = Field(alias="SUPPORTED", ge=0, le=1)
    refuted: float = Field(alias="REFUTED", ge=0, le=1)
    not_enough_information: float = Field(alias="NOT_ENOUGH_INFORMATION", ge=0, le=1)

    model_config = ConfigDict(populate_by_name=True)


class VerificationSignal(BaseModel):
    model_verdict: Verdict
    model_confidence: float = Field(ge=0, le=1)
    class_probabilities: ClassProbabilities
    model_version: str
    rule_verdict: Verdict | None = None
    rule_confidence: float | None = Field(default=None, ge=0, le=1)
    rule_findings: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    text: str | None = None
    source: str
    published_at: datetime | None = None
    retrieved_at: datetime
    stance: EvidenceStance
    scores: EvidenceScores
    verification: VerificationSignal


class ClaimResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    claim: str
    claim_type: ClaimType
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    confidence_basis: ConfidenceBasis
    model_verdict: Verdict
    model_confidence: float = Field(ge=0, le=1)
    class_probabilities: ClassProbabilities
    evidence_quality: float = Field(ge=0, le=1)
    conflict_detected: bool = False
    evidence: list[EvidenceItem] = Field(default_factory=list)
    explanation: str


class FactCheckRequest(BaseModel):
    text: str = Field(min_length=3)
    language: str = Field(default="ar", pattern="^(ar|ara)$")
    claimed_at: datetime | None = None
    force_refresh: bool = False

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be blank")
        return value


class FactCheckResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    language: str = "ar"
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    confidence_basis: ConfidenceBasis
    claims: list[ClaimResult]
    evidence: list[EvidenceItem]
    explanation: str
    conflict_detected: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    searched_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
