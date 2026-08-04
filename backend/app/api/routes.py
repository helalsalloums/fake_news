from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db import get_session
from app.repository import FactCheckRepository
from app.schemas.domain import ExtractedClaim, FactCheckRequest, FactCheckResponse
from app.services.cache import RedisCache
from app.services.pipeline import FactCheckingPipeline

router = APIRouter()


class ClaimsResponse(BaseModel):
    claims: list[ExtractedClaim]


def get_pipeline(request: Request) -> FactCheckingPipeline:
    return request.app.state.pipeline


@router.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(
    payload: FactCheckRequest,
    request: Request,
    pipeline: Annotated[FactCheckingPipeline, Depends(get_pipeline)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FactCheckResponse:
    if len(payload.text) > settings.max_input_chars:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "input text is too large")
    cache: RedisCache = request.app.state.cache
    cache_key = cache.key(
        "fact-check",
        f"{payload.language}:{payload.claimed_at}:{pipeline.verifier.__class__.__name__}:{payload.text}",
    )
    if not payload.force_refresh:
        cached = await cache.get_json(cache_key)
        if cached is not None:
            return FactCheckResponse.model_validate(cached)
    result = await pipeline.run(payload.text, payload.claimed_at)
    await FactCheckRepository(session).save(payload.text, result)
    await cache.set_json(
        cache_key,
        result.model_dump(mode="json", by_alias=True),
        settings.fact_check_cache_ttl_seconds,
    )
    return result


@router.post("/claims", response_model=ClaimsResponse)
async def extract_claims(
    payload: FactCheckRequest,
    pipeline: Annotated[FactCheckingPipeline, Depends(get_pipeline)],
) -> ClaimsResponse:
    return ClaimsResponse(claims=await pipeline.extractor.extract(payload.text))


@router.get("/fact-check/{fact_check_id}", response_model=FactCheckResponse)
async def get_fact_check(
    fact_check_id: UUID, session: Annotated[AsyncSession, Depends(get_session)]
) -> FactCheckResponse:
    result = await FactCheckRepository(session).get(fact_check_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "fact check not found")
    return result


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "search_provider": request.app.state.pipeline.search_provider.name,
        "verifier": request.app.state.pipeline.verifier.__class__.__name__,
    }


@router.get("/config")
async def config(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    return settings.public_dict()
