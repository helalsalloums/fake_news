from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from app.schemas.domain import SearchResult
from app.services.interfaces import SearchProvider


class LocalSearchProvider:
    name = "local"

    def __init__(self, fixture_path: Path) -> None:
        self.fixture_path = fixture_path

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if not self.fixture_path.exists():
            return []
        raw = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        rows = raw.get("results", raw) if isinstance(raw, dict) else raw
        query_terms = set(query.replace('"', "").split())
        ranked: list[tuple[int, SearchResult]] = []
        for index, row in enumerate(rows):
            item = SearchResult.model_validate({**row, "provider": self.name, "rank": index + 1})
            text_terms = set(f"{item.title} {item.snippet}".split())
            ranked.append((len(query_terms & text_terms), item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].rank))
        return [item for _, item in ranked[:limit]]


class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params: dict[str, str | int] = {
            "q": query,
            "count": min(limit, 20),
            "search_lang": "ar",
            "safesearch": "moderate",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search", headers=headers, params=params
            )
            response.raise_for_status()
        rows = response.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                title=row.get("title", ""),
                url=row["url"],
                snippet=row.get("description", ""),
                source=row.get("profile", {}).get("long_name", ""),
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(rows[:limit], start=1)
            if row.get("url")
        ]


class SearxngSearchProvider:
    name = "searxng"

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                urljoin(self.base_url, "search"),
                params={"q": query, "format": "json", "language": "ar"},
            )
            response.raise_for_status()
        return [
            SearchResult(
                title=row.get("title", ""),
                url=row["url"],
                snippet=row.get("content", ""),
                source=row.get("engine", ""),
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(response.json().get("results", [])[:limit], start=1)
            if row.get("url")
        ]


class GoogleCustomSearchProvider:
    """Legacy adapter for existing Google Programmable Search customers."""

    name = "google"

    def __init__(self, api_key: str, engine_id: str, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.engine_id = engine_id
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": min(limit, 10),
            "lr": "lang_ar",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                "https://customsearch.googleapis.com/customsearch/v1", params=params
            )
            response.raise_for_status()
        return [
            SearchResult(
                title=row.get("title", ""),
                url=row["link"],
                snippet=row.get("snippet", ""),
                source=row.get("displayLink", ""),
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(response.json().get("items", [])[:limit], start=1)
            if row.get("link")
        ]


class DuckDuckGoSearchProvider:
    """Best-effort public provider; not recommended for production guarantees."""

    name = "duckduckgo"

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        def execute() -> list[dict[str, str]]:
            try:
                from ddgs import DDGS
            except ImportError as error:
                raise RuntimeError("install backend[search] to use DuckDuckGo") from error
            return list(
                DDGS().text(query, region="xa-ar", safesearch="moderate", max_results=limit)
            )

        rows = await asyncio.to_thread(execute)
        return [
            SearchResult(
                title=row.get("title", ""),
                url=row["href"],
                snippet=row.get("body", ""),
                source="DuckDuckGo",
                provider=self.name,
                rank=index,
            )
            for index, row in enumerate(rows, start=1)
            if row.get("href")
        ]


def build_search_provider(
    settings: Any,
) -> SearchProvider:
    name = getattr(settings, "search_provider", "local").lower()
    if name == "brave":
        key = getattr(settings, "brave_search_api_key", None) or getattr(
            settings, "search_api_key", None
        )
        if not key:
            raise ValueError("BRAVE_SEARCH_API_KEY is required for SEARCH_PROVIDER=brave")
        return BraveSearchProvider(key, getattr(settings, "request_timeout_seconds", 15.0))
    if name == "searxng":
        base_url = getattr(settings, "searxng_base_url", None)
        if not base_url:
            raise ValueError("SEARXNG_BASE_URL is required for SEARCH_PROVIDER=searxng")
        return SearxngSearchProvider(base_url, getattr(settings, "request_timeout_seconds", 15.0))
    if name == "google":
        key = getattr(settings, "search_api_key", None)
        engine_id = getattr(settings, "google_cse_id", None)
        if not key or not engine_id:
            raise ValueError("SEARCH_API_KEY and GOOGLE_CSE_ID are required")
        return GoogleCustomSearchProvider(
            key, engine_id, getattr(settings, "request_timeout_seconds", 15.0)
        )
    if name in {"ddg", "duckduckgo"}:
        return DuckDuckGoSearchProvider()
    return LocalSearchProvider(settings.local_search_fixture)
