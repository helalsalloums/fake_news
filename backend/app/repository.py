from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import ClaimORM, EvidenceORM, FactCheckORM, VerificationORM
from app.schemas.domain import FactCheckResponse


class FactCheckRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, input_text: str, result: FactCheckResponse) -> None:
        record = FactCheckORM(
            id=result.id,
            input_text=input_text,
            language=result.language,
            created_at=result.created_at,
            searched_at=result.searched_at,
            overall_verdict=result.verdict.value,
            overall_confidence=result.confidence,
            confidence_basis=result.confidence_basis.value,
            conflict_detected=result.conflict_detected,
            explanation=result.explanation,
            response_json=result.model_dump(mode="json", by_alias=True),
        )
        for claim in result.claims:
            claim_record = ClaimORM(
                id=claim.id,
                claim_text=claim.claim,
                claim_type=claim.claim_type.value,
                verdict=claim.verdict.value,
                confidence=claim.confidence,
                model_verdict=claim.model_verdict.value,
                model_confidence=claim.model_confidence,
                class_probabilities=claim.class_probabilities.model_dump(by_alias=True),
                evidence_quality=claim.evidence_quality,
                conflict_detected=claim.conflict_detected,
                explanation=claim.explanation,
            )
            for evidence in claim.evidence:
                scores = evidence.scores
                claim_record.evidence.append(
                    EvidenceORM(
                        external_id=evidence.id,
                        url=evidence.url,
                        title=evidence.title,
                        source=evidence.source,
                        published_at=evidence.published_at,
                        retrieved_at=evidence.retrieved_at,
                        text=evidence.text,
                        snippet=evidence.snippet,
                        stance=evidence.stance.value,
                        relevance_score=scores.relevance,
                        source_score=scores.source,
                        recency_score=scores.recency,
                        directness_score=scores.directness,
                        agreement_score=scores.agreement,
                        overall_score=scores.overall,
                    )
                )
                signal = evidence.verification
                claim_record.verifications.append(
                    VerificationORM(
                        evidence_external_id=evidence.id,
                        model=signal.model_version,
                        verdict=signal.model_verdict.value,
                        confidence=signal.model_confidence,
                        class_probabilities=signal.class_probabilities.model_dump(by_alias=True),
                        rule_findings=signal.rule_findings,
                    )
                )
            record.claims.append(claim_record)
        self.session.add(record)
        await self.session.commit()

    async def get(self, fact_check_id: UUID) -> FactCheckResponse | None:
        statement = (
            select(FactCheckORM)
            .where(FactCheckORM.id == fact_check_id)
            .options(selectinload(FactCheckORM.claims))
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        return FactCheckResponse.model_validate(record.response_json) if record else None
