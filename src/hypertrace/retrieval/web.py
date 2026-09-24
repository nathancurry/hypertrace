"""Brave web search and ordinary HTTP page retrieval."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from datetime import date
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from hypertrace.models import ContentRegion, QuoteRegion, Source
from hypertrace.retrieval.base import FetchFailure, SearchResult

TRACKING = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Unsupported URL: {url}")
    host = parts.hostname.lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Non-public URL")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Non-public URL")
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in TRACKING
        ]
    )
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), host + port, path, query, ""))


def _published_date(value: str) -> str | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:T.*)?$", value.strip())
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1)).isoformat()
    except ValueError:
        return None


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.stack: list[tuple[str, QuoteRegion, dict[str, str | None] | None]] = []
        self.block_ids: list[int] = []
        self.next_block_id = 0
        self.title: list[str] = []
        self.parts: list[tuple[str, QuoteRegion, dict[str, str | None] | None, int]] = []
        self.canonical: str | None = None
        self.author: str | None = None
        self.publication_date: str | None = None
        self.search_aggregator = False
        self.article_count = 0
        self.article_tail_region: QuoteRegion | None = None

    def _region(self, tag: str, attrs: dict[str, str | None]) -> QuoteRegion:
        marker = " ".join(str(attrs.get(key) or "") for key in ("class", "id", "role", "itemtype"))
        marker = marker.lower().replace("_", "-")
        if (
            any(
                word in marker
                for word in (
                    "searchresultspage",
                    "search-results",
                    "search-result",
                    "search-hit",
                    "results-list",
                    "result-card",
                    "result-snippet",
                )
            )
            or "result" in marker.split()
        ):
            self.search_aggregator = True
        if self.stack and self.stack[-1][1] in {
            QuoteRegion.NAVIGATION,
            QuoteRegion.TAG,
            QuoteRegion.RELATED_CONTENT,
        }:
            return self.stack[-1][1]
        if tag in {"nav", "footer", "header"} or "navigation" in marker:
            return QuoteRegion.NAVIGATION
        if tag == "aside" or any(
            word in marker
            for word in (
                "related",
                "recommended",
                "search-result",
                "search-hit",
                "results-list",
                "result",
                "snippet",
                "recirculation",
                "comment",
                "sidebar",
                "promo",
                "share-widget",
            )
        ):
            return QuoteRegion.RELATED_CONTENT
        if "tag" in (attrs.get("rel") or "").lower().split() or any(
            word in marker
            for word in ("tag-list", "tags", "post-tag", "tag-cloud", "taxonomy", "topics")
        ):
            return QuoteRegion.TAG
        if tag == "title":
            return QuoteRegion.TITLE
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return QuoteRegion.HEADING
        if self.article_tail_region is not None and any(
            ancestor == "article" for ancestor, _, _ in self.stack
        ):
            return self.article_tail_region
        if tag == "article" or any(
            word in marker
            for word in ("article-body", "post-content", "entry-content", "story-body")
        ):
            return QuoteRegion.ARTICLE_BODY
        return self.stack[-1][1] if self.stack else QuoteRegion.UNKNOWN

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "article":
            self.article_count += 1
            if self.article_count > 1:
                self.search_aggregator = True
        if (
            tag == "a"
            and attrs_dict.get("href")
            and any(ancestor in {"h2", "h3"} for ancestor, _, _ in self.stack)
            and any(ancestor == "article" for ancestor, _, _ in self.stack)
        ):
            self.search_aggregator = True
        if tag == "link" and "canonical" in (attrs_dict.get("rel") or "").lower():
            self.canonical = attrs_dict.get("href")
        if tag == "meta":
            name = (attrs_dict.get("property") or attrs_dict.get("name") or "").lower()
            content = attrs_dict.get("content") or ""
            if name in {"article:published_time", "datepublished", "pubdate"}:
                self.publication_date = _published_date(content) or self.publication_date
            if name in {"author", "article:author"}:
                self.author = content[:300] or None
        if tag not in {"meta", "link", "br", "hr", "img", "input", "source", "wbr"}:
            region = self._region(tag, attrs_dict)
            marker = " ".join(str(attrs_dict.get(key) or "") for key in ("class", "id"))
            marker_tokens = marker.lower().replace("_", "-").split()
            card = (
                {"url": None}
                if any(
                    word in marker_tokens
                    for word in ("search-result", "search-hit", "result-card", "result")
                )
                else (self.stack[-1][2] if self.stack else None)
            )
            if tag == "a" and card is not None and card["url"] is None and attrs_dict.get("href"):
                try:
                    card["url"] = canonicalize_url(urljoin(self.base_url, attrs_dict["href"]))
                except ValueError:
                    pass
            self.stack.append((tag, region, card))
            if tag in {
                "article",
                "p",
                "div",
                "section",
                "li",
                "blockquote",
                "title",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "nav",
                "aside",
            }:
                self.next_block_id += 1
                self.block_ids.append(self.next_block_id)
            else:
                self.block_ids.append(self.block_ids[-1] if self.block_ids else 0)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                del self.block_ids[index:]
                break
        if tag == "article":
            self.article_tail_region = None

    def handle_data(self, data: str) -> None:
        if any(tag in {"script", "style", "noscript", "svg"} for tag, _, _ in self.stack):
            return
        if any(tag == "title" for tag, _, _ in self.stack):
            self.title.append(data)
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            if self.stack and self.stack[-1][0] in {"h2", "h3", "h4", "h5", "h6"}:
                if re.match(
                    r"^(related|recommended|more (?:stories|articles)|read more)\b",
                    text,
                    re.IGNORECASE,
                ):
                    self.article_tail_region = QuoteRegion.RELATED_CONTENT
                elif re.match(r"^(tags|topics)\b", text, re.IGNORECASE):
                    self.article_tail_region = QuoteRegion.TAG
            self.parts.append(
                (
                    text,
                    (
                        self.article_tail_region
                        if self.article_tail_region is not None
                        and any(ancestor == "article" for ancestor, _, _ in self.stack)
                        else (self.stack[-1][1] if self.stack else QuoteRegion.UNKNOWN)
                    ),
                    self.stack[-1][2] if self.stack else None,
                    self.block_ids[-1] if self.block_ids else 0,
                )
            )

    def rendered(self) -> tuple[str, list[ContentRegion]]:
        content_parts: list[str] = []
        regions: list[ContentRegion] = []
        position = 0
        for part, region, card, block_id in self.parts:
            if content_parts:
                content_parts.append(" ")
                position += 1
            start = position
            content_parts.append(part)
            position += len(part)
            regions.append(
                ContentRegion(
                    start=start,
                    end=position,
                    region=region,
                    block_id=block_id,
                    target_url=card["url"] if card else None,
                )
            )
        return "".join(content_parts), regions


class BraveWeb:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=30, follow_redirects=False)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        response = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(count, 20)},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        )
        response.raise_for_status()
        return [
            SearchResult(
                url=item["url"], title=item.get("title", ""), snippet=item.get("description", "")
            )
            for item in response.json().get("web", {}).get("results", [])
            if item.get("url")
        ]

    async def fetch(self, url: str) -> Source:
        redirects = [canonicalize_url(url)]
        try:
            return await self._fetch(redirects)
        except (httpx.HTTPStatusError, httpx.TransportError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if status is not None:
                reason = f"HTTP {status}"
            elif isinstance(exc, httpx.TransportError):
                reason = type(exc).__name__
            else:
                reason = str(exc)
            raise FetchFailure(
                reason,
                url,
                redirects,
                status_code=status,
                retryable=(status == 429 or status is not None and 500 <= status < 600)
                or isinstance(exc, httpx.TransportError),
            ) from exc

    async def _fetch(self, redirects: list[str]) -> Source:
        current = redirects[-1]
        for _ in range(4):
            async with self.client.stream(
                "GET", current, headers={"User-Agent": "hypertrace/0.1"}
            ) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Redirect without location")
                    current = canonicalize_url(urljoin(current, location))
                    redirects.append(current)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if not ("text/html" in content_type or "text/plain" in content_type):
                    raise ValueError(f"Unsupported content type: {content_type}")
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 2_000_000:
                        raise ValueError("Page exceeds 2 MB retrieval limit")
                    chunks.append(chunk)
                raw = b"".join(chunks)
                try:
                    page_text = raw.decode(response.encoding or "utf-8", errors="replace")
                except LookupError:
                    page_text = raw.decode("utf-8", errors="replace")
                break
        else:
            raise ValueError("Too many redirects")
        if "text/html" in content_type:
            parser = _PageParser(current)
            parser.feed(page_text)
            content, content_regions = parser.rendered()
            title = re.sub(r"\s+", " ", "".join(parser.title)).strip()[:500]
            canonical = current
            if parser.canonical:
                try:
                    proposed = canonicalize_url(urljoin(current, parser.canonical))
                except ValueError:
                    pass
                else:
                    if urlsplit(proposed).hostname == urlsplit(current).hostname:
                        canonical = proposed
            author = parser.author
            publication_date = parser.publication_date
            source_type = (
                "search_aggregator"
                if parser.search_aggregator
                or re.search(
                    r"/(?:search|results|find)(?:/|$)|[?&](?:q|query|search)=",
                    urlsplit(current).path + "?" + urlsplit(current).query,
                    re.IGNORECASE,
                )
                else "web_page"
            )
        else:
            content = page_text.strip()
            content_regions = []
            title = ""
            canonical = current
            author = None
            publication_date = None
            source_type = "plain_text"
        if not content:
            raise ValueError("Fetched page contains no readable text")
        content = content[:200_000]
        content_regions = [
            region.model_copy(update={"end": min(region.end, len(content))})
            for region in content_regions
            if region.start < len(content)
        ]
        return Source(
            canonical_url=canonical,
            retrieved_url=current,
            title=title,
            author=author,
            page_publication_date=publication_date,
            source_type=source_type,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            document_hash=hashlib.sha256(raw).hexdigest(),
            content=content,
            content_regions=content_regions,
            dating_notes=(
                "Date from page metadata; publication timing unverified."
                if publication_date
                else "No reliable publication date found."
            ),
        )
