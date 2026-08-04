from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db import build_engine, build_session_factory, create_schema
from app.services.aggregation import ConservativeVerdictAggregator
from app.services.cache import CachedDocumentFetcher, CachedSearchProvider, RedisCache
from app.services.claims import RuleBasedClaimExtractor
from app.services.fetch import SecureDocumentFetcher
from app.services.pipeline import FactCheckingPipeline
from app.services.query import ArabicQueryGenerator
from app.services.retrieval import HybridEvidenceRetriever, LexicalEvidenceRetriever
from app.services.scoring import ConfigurableEvidenceRanker, ScoringWeights, SourceReliability
from app.services.search import build_search_provider
from app.services.verification import RuleBasedVerifier, TransformerNliVerifier

REQUESTS = Counter("factchecker_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("factchecker_http_request_seconds", "HTTP request latency", ["method", "path"])
logger = structlog.get_logger(__name__)


def build_pipeline(settings: Settings, cache: RedisCache | None = None) -> FactCheckingPipeline:
    verifier = (
        TransformerNliVerifier(settings.verifier_model)
        if settings.enable_neural_models
        else RuleBasedVerifier()
    )
    reliability_path = settings.source_reliability_path
    if not reliability_path.exists():
        reliability_path = Path(__file__).resolve().parents[2] / "configs/source_reliability.yaml"
    scoring_path = settings.evidence_scoring_path
    if not scoring_path.exists():
        scoring_path = Path(__file__).resolve().parents[2] / "configs/evidence_scoring.yaml"
    search_provider: Any = build_search_provider(settings)
    fetcher: Any = SecureDocumentFetcher(
        settings.request_timeout_seconds, settings.max_document_bytes
    )
    retriever = (
        HybridEvidenceRetriever(settings.embedding_model)
        if settings.enable_neural_models
        else LexicalEvidenceRetriever()
    )
    if cache is not None:
        search_provider = CachedSearchProvider(
            search_provider, cache, settings.search_cache_ttl_seconds
        )
        fetcher = CachedDocumentFetcher(fetcher, cache, settings.page_cache_ttl_seconds)
    return FactCheckingPipeline(
        extractor=RuleBasedClaimExtractor(),
        query_generator=ArabicQueryGenerator(),
        search_provider=search_provider,
        fetcher=fetcher,
        retriever=retriever,
        ranker=ConfigurableEvidenceRanker(
            SourceReliability(reliability_path), ScoringWeights.from_yaml(scoring_path)
        ),
        verifier=verifier,
        aggregator=ConservativeVerdictAggregator(
            settings.model_confidence_threshold,
            settings.model_margin_threshold,
            settings.evidence_quality_threshold,
        ),
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        content_length = request.headers.get("content-length")
        maximum_bytes = request.app.state.settings.max_input_chars * 4 + 1024
        if content_length and int(content_length) > maximum_bytes:
            return Response(
                content='{"detail":"request body is too large"}',
                status_code=413,
                media_type="application/json",
            )
        if request.url.path.endswith("/fact-check") and request.method == "POST":
            cache: RedisCache | None = getattr(request.app.state, "cache", None)
            if cache is not None:
                minute = int(time.time() // 60)
                client = request.client.host if request.client else "unknown"
                count = await cache.increment(f"factchecker:rate:{client}:{minute}", 61)
                if count is not None and count > request.app.state.settings.rate_limit_per_minute:
                    return Response(
                        content='{"detail":"rate limit exceeded"}',
                        status_code=429,
                        media_type="application/json",
                    )
        response = await call_next(request)
        duration = time.perf_counter() - started
        response.headers["x-request-id"] = request_id
        REQUESTS.labels(request.method, request.url.path, response.status_code).inc()
        LATENCY.labels(request.method, request.url.path).observe(duration)
        await logger.ainfo(
            "request_completed", status=response.status_code, latency_ms=duration * 1000
        )
        structlog.contextvars.clear_contextvars()
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    selected = settings or get_settings()
    configure_logging(selected.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cache = RedisCache(selected.redis_url, selected.cache_enabled)
        engine = build_engine(selected.database_url)
        app.state.settings = selected
        app.state.cache = cache
        app.state.engine = engine
        app.state.session_factory = build_session_factory(engine)
        app.state.pipeline = build_pipeline(selected, cache)
        await create_schema(engine)
        yield
        await cache.close()
        await engine.dispose()

    app = FastAPI(
        title=selected.app_name,
        version="0.1.0",
        description="Evidence-based Arabic fact checking without generative LLMs.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[selected.frontend_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-request-id"],
    )
    app.include_router(router, prefix=selected.api_prefix)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
