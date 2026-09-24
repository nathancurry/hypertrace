"""Search results and fetched pages are separate types by design."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from hypertrace.models import Source


class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""  # Discovery only; never stored as evidence.


class FetchFailure(Exception):
    def __init__(
        self,
        reason: str,
        original_url: str,
        redirect_chain: list[str],
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(reason)
        self.original_url = original_url
        self.redirect_chain = redirect_chain
        self.final_url = redirect_chain[-1]
        self.status_code = status_code
        self.retryable = retryable


class Retrieval(Protocol):
    async def search(self, query: str, count: int = 5) -> list[SearchResult]: ...
    async def fetch(self, url: str) -> Source: ...
