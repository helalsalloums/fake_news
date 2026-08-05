from __future__ import annotations

import hashlib
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
import trafilatura
from dateutil import parser as date_parser

from app.schemas.domain import Document, SearchResult

ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}


class UnsafeUrlError(ValueError):
    pass

import asyncio

async def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("only public HTTP(S) URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("credentialed URLs are not allowed")
    if parsed.port not in {None, 80, 443}:
        raise UnsafeUrlError("non-standard ports are not allowed")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in BLOCKED_HOSTS or hostname.endswith(".localhost"):
        raise UnsafeUrlError("local hosts are not allowed")
    try:
        loop = asyncio.get_running_loop()
        addresses = await loop.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise UnsafeUrlError("host could not be resolved") from error
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise UnsafeUrlError("non-public address is not allowed")

class SecureDocumentFetcher:
    def __init__(
        self,
        timeout: float = 15.0,
        max_bytes: int = 5_000_000,
        cleaner: TrafilaturaDocumentCleaner | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.cleaner = cleaner or TrafilaturaDocumentCleaner()

    async def fetch(self, result: SearchResult) -> Document | None:
        current_url = result.url
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            headers={"User-Agent": "ArabicFactChecker/0.1 (+https://github.com/)"},
        ) as client:
            for _ in range(5):
                await validate_public_url(current_url)
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return None
                        current_url = str(httpx.URL(current_url).join(location))
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if not content_type.startswith(ALLOWED_CONTENT_TYPES):
                        return None
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            return None
                        chunks.append(chunk)
                    html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                    return self.cleaner.clean(html, current_url, result)
        return None


class TrafilaturaDocumentCleaner:
    """Convert untrusted HTML into bounded article data; never interprets instructions."""

    def clean(self, html: str, url: str, result: SearchResult) -> Document | None:
        extracted = trafilatura.bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            with_metadata=True,
        )
        if not extracted:
            return None
        data = extracted.as_dict() if hasattr(extracted, "as_dict") else extracted
        text = str(data.get("text") or "").strip()
        if len(text) < 80:
            return None
        published_at = result.published_at
        if not published_at and data.get("date"):
            try:
                published_at = date_parser.parse(str(data["date"]))
            except (ValueError, TypeError, OverflowError):
                published_at = None
        return Document(
            url=url,
            title=str(data.get("title") or result.title),
            source=str(data.get("sitename") or result.source or urlparse(url).hostname or ""),
            text=text,
            published_at=published_at,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


def clean_document(html: str, url: str, result: SearchResult) -> Document | None:
    """Backward-compatible function for callers that do not inject a cleaner."""
    return TrafilaturaDocumentCleaner().clean(html, url, result)
