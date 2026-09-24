"""Bounded Internet Archive CDX lookup and snapshot retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlsplit

import httpx

from hypertrace.models import Source
from hypertrace.retrieval.base import FetchFailure
from hypertrace.retrieval.web import BraveWeb, canonicalize_url

CDX_URL = "https://web.archive.org/cdx/search/cdx"
MAX_SNAPSHOTS = 3


@dataclass(frozen=True)
class Snapshot:
    timestamp: str
    original_url: str
    url: str
    digest: str


class ArchiveGap(Exception):
    def __init__(
        self,
        reason: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        redirect_chain: list[str] | None = None,
        final_url: str | None = None,
    ):
        super().__init__(reason)
        self.retryable = retryable
        self.status_code = status_code
        self.redirect_chain = redirect_chain
        self.final_url = final_url


class Wayback:
    def __init__(self, web: BraveWeb):
        self.web = web

    async def lookup(
        self,
        original_url: str,
        start_year: int | None = None,
        end_year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Snapshot]:
        original_url = canonicalize_url(original_url)
        first = date.fromisoformat(start_date) if start_date else None
        if first is None and start_year:
            first = date(start_year, 1, 1)
        last = date.fromisoformat(end_date) if end_date else None
        if last is None and end_year:
            last = date(end_year, 12, 31)
        if last is None:
            last = first if start_date else date(start_year, 12, 31) if start_year else None
        if first is None:
            first = last
        if first and last and last < first:
            raise ValueError("Archive target date range is reversed")
        params: dict[str, str | int] = {
            "url": original_url,
            "output": "json",
            "fl": "timestamp,original,mimetype,statuscode,digest",
            "limit": 100,
        }
        if first:
            params["from"] = max(1996, first.year - 1)
        try:
            response = await self.web.client.get(CDX_URL, params=params)
            response.raise_for_status()
            rows = response.json() if response.content.strip() else []
        except (httpx.HTTPError, ValueError) as exc:
            reason = (
                f"HTTP {exc.response.status_code}"
                if isinstance(exc, httpx.HTTPStatusError)
                else type(exc).__name__
            )
            raise ArchiveGap(f"CDX lookup failed: {reason}", retryable=True) from exc
        if not isinstance(rows, list):
            raise ArchiveGap("CDX returned malformed data", retryable=True)
        if len(rows) <= 1:
            raise ArchiveGap("No Wayback snapshots")
        fields = rows[0]
        if not isinstance(fields, list) or not set(params["fl"].split(",")).issubset(fields):
            raise ArchiveGap("CDX returned malformed fields", retryable=True)
        candidates: list[Snapshot] = []
        failures: set[str] = set()
        for row in rows[1:]:
            if not isinstance(row, list) or len(row) != len(fields):
                continue
            item = dict(zip(fields, row, strict=True))
            stamp = item["timestamp"]
            if not re.fullmatch(r"\d{14}", stamp):
                continue
            try:
                date.fromisoformat(stamp[:8])
            except ValueError:
                continue
            if item["statuscode"] != "200":
                failures.add(
                    "redirect-only capture"
                    if item["statuscode"].startswith("3")
                    else "blocked snapshot"
                )
                continue
            if item["mimetype"] != "text/html":
                failures.add("unsupported content")
                continue
            try:
                resolved = canonicalize_url(item["original"])
            except ValueError:
                continue
            candidates.append(
                Snapshot(
                    stamp,
                    resolved,
                    f"https://web.archive.org/web/{stamp}id_/{resolved}",
                    item["digest"],
                )
            )
        if not candidates:
            raise ArchiveGap(", ".join(sorted(failures)) or "No useful Wayback snapshots")

        def distance(snapshot: Snapshot) -> tuple[int, str]:
            captured = date.fromisoformat(snapshot.timestamp[:8])
            if not first:
                return (0, snapshot.timestamp)
            return (
                max((first - captured).days, (captured - (last or first)).days, 0),
                snapshot.timestamp,
            )

        selected: list[Snapshot] = []
        digests: set[str] = set()
        for snapshot in sorted(candidates, key=distance):
            if snapshot.digest in digests:
                continue
            digests.add(snapshot.digest)
            selected.append(snapshot)
            if len(selected) == MAX_SNAPSHOTS:
                break
        return selected

    async def fetch(self, requested_url: str, snapshot: Snapshot) -> Source:
        try:
            source = await self.web.fetch(snapshot.url)
        except FetchFailure as exc:
            raise ArchiveGap(
                f"Snapshot fetch failed: {exc}",
                retryable=exc.retryable,
                status_code=exc.status_code,
                redirect_chain=exc.redirect_chain,
                final_url=exc.final_url,
            ) from exc
        final = urlsplit(source.retrieved_url)
        replay = re.match(r"/web/(\d{14})[^/]*/(https?://.+)", final.path)
        if final.hostname != "web.archive.org" or replay is None:
            raise ArchiveGap(
                "Redirect-only capture",
                status_code=source.http_status,
                redirect_chain=source.redirect_history,
                final_url=source.retrieved_url,
            )
        if source.source_type != "web_page":
            raise ArchiveGap(
                "Unsupported archived content",
                status_code=source.http_status,
                redirect_chain=source.redirect_history,
                final_url=source.retrieved_url,
            )
        try:
            replayed_original = replay.group(2) + (f"?{final.query}" if final.query else "")
            resolved = canonicalize_url(replayed_original)
        except ValueError as exc:
            raise ArchiveGap("Redirect-only capture") from exc
        return source.model_copy(
            update={
                "canonical_url": source.retrieved_url,
                "original_url": canonicalize_url(requested_url),
                "resolved_original_url": resolved,
                "archive_url": source.retrieved_url,
                "archive_timestamp": replay.group(1),
                "dating_notes": (
                    source.dating_notes
                    + f" Wayback capture {replay.group(1)} proves content existed by capture date only."
                ),
            }
        )
