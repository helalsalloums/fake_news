from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.schemas.domain import utc_now


class FactCheckORM(Base):
    __tablename__ = "fact_checks"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    input_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8), default="ar")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    searched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    overall_verdict: Mapped[str] = mapped_column(String(32), index=True)
    overall_confidence: Mapped[float] = mapped_column(Float)
    confidence_basis: Mapped[str] = mapped_column(String(64))
    conflict_detected: Mapped[bool] = mapped_column(default=False)
    explanation: Mapped[str] = mapped_column(Text)
    response_json: Mapped[dict[str, object]] = mapped_column(JSON)

    claims: Mapped[list[ClaimORM]] = relationship(
        back_populates="fact_check", cascade="all, delete-orphan"
    )


class ClaimORM(Base):
    __tablename__ = "claims"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    fact_check_id: Mapped[UUID] = mapped_column(
        ForeignKey("fact_checks.id", ondelete="CASCADE"), index=True
    )
    claim_text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(32))
    verdict: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    model_verdict: Mapped[str] = mapped_column(String(32))
    model_confidence: Mapped[float] = mapped_column(Float)
    class_probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    evidence_quality: Mapped[float] = mapped_column(Float)
    conflict_detected: Mapped[bool] = mapped_column(default=False)
    explanation: Mapped[str] = mapped_column(Text)

    fact_check: Mapped[FactCheckORM] = relationship(back_populates="claims")
    evidence: Mapped[list[EvidenceORM]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )
    verifications: Mapped[list[VerificationORM]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )


class EvidenceORM(Base):
    __tablename__ = "evidence"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    external_id: Mapped[str] = mapped_column(String(64), index=True)
    claim_id: Mapped[UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text: Mapped[str | None] = mapped_column(Text)
    snippet: Mapped[str] = mapped_column(Text)
    stance: Mapped[str] = mapped_column(String(32))
    relevance_score: Mapped[float] = mapped_column(Float)
    source_score: Mapped[float] = mapped_column(Float)
    recency_score: Mapped[float] = mapped_column(Float)
    directness_score: Mapped[float] = mapped_column(Float)
    agreement_score: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float)

    claim: Mapped[ClaimORM] = relationship(back_populates="evidence")


class VerificationORM(Base):
    __tablename__ = "verifications"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    claim_id: Mapped[UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    evidence_external_id: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    verdict: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    class_probabilities: Mapped[dict[str, float]] = mapped_column(JSON)
    rule_findings: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    claim: Mapped[ClaimORM] = relationship(back_populates="verifications")


Index("ix_fact_checks_created_verdict", FactCheckORM.created_at, FactCheckORM.overall_verdict)
