from __future__ import annotations

import hashlib
import json
from typing import Any

from app.schemas.domain import Document, SearchResult


class RedisCache:
    def __init__(self, url: str | None, enabled: bool = True) -> None:
        self.url = url
        self.enabled = enabled and bool(url)
        self._client: Any = None

    async def _get_client(self) -> Any:
        if not self.enabled:
            return None
        if self._client is None:
            from redis.asyncio import from_url

            self._client = from_url(self.url, encoding="utf-8", decode_responses=True)
        return self._client

    @staticmethod
    def key(namespace: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"factchecker:{namespace}:{digest}"

    async def get_json(self, key: str) -> Any | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            value = await client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
        except Exception:
            return

    async def increment(self, key: str, ttl: int) -> int | None:
        client = await self._get_client()
        if client is None:
            return None
        try:
            value = await client.incr(key)
            if value == 1:
                await client.expire(key, ttl)
            return int(value)
        except Exception:
            return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class CachedSearchProvider:
    def __init__(self, provider: Any, cache: RedisCache, ttl: int) -> None:
        self.provider = provider
        self.cache = cache
        self.ttl = ttl
        self.name = provider.name

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        key = self.cache.key("search", f"{self.name}:{limit}:{query}")
        cached = await self.cache.get_json(key)
        if cached is not None:
            return [SearchResult.model_validate(item) for item in cached]
        result = await self.provider.search(query, limit)
        await self.cache.set_json(key, [item.model_dump(mode="json") for item in result], self.ttl)
        return result


class CachedDocumentFetcher:
    def __init__(self, fetcher: Any, cache: RedisCache, ttl: int) -> None:
        self.fetcher = fetcher
        self.cache = cache
        self.ttl = ttl

    async def fetch(self, result: SearchResult) -> Document | None:
        key = self.cache.key("document", result.url)
        cached = await self.cache.get_json(key)
        if cached is not None:
            return Document.model_validate(cached)
        document = await self.fetcher.fetch(result)
        if document is not None:
            await self.cache.set_json(key, document.model_dump(mode="json"), self.ttl)
        return document
