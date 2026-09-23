"""Search results and fetched pages are separate types by design."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from hypertrace.models import Source


class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""  # Discovery only; never stored as evidence.


class Retrieval(Protocol):
    async def search(self, query: str, count: int = 5) -> list[SearchResult]: ...
    async def fetch(self, url: str) -> Source: ...
