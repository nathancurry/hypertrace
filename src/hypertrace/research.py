"""Bounded, inspectable research loop."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass

import httpx

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.base import StructuredLLM
from hypertrace.llm.openai_compatible import OpenAICompatibleLLM, StructuredOutputError
from hypertrace.models import (
    AdversarialReview,
    Evidence,
    Hypothesis,
    HypothesisScreen,
    Interpretation,
    PageAssessment,
    QueryPlan,
    QuoteRegion,
    ResearchLead,
    ResearchRun,
    SearchQuery,
    Source,
)
from hypertrace.planner import (
    ASSESS_SYSTEM,
    INTERPRET_SYSTEM,
    PLAN_SYSTEM,
    REVIEW_SYSTEM,
    SCREEN_SYSTEM,
)
from hypertrace.retrieval.base import FetchFailure, Retrieval
from hypertrace.retrieval.wayback import ArchiveGap, Snapshot, Wayback
from hypertrace.retrieval.web import BraveWeb


class BudgetStop(Exception):
    pass


ASSESSMENT_LIMIT = 16_000
REVIEW_CONTEXT_LIMIT = 24_000  # About 6,000 tokens at the provider's four-character estimate.
WINDOW_CONTEXT = 400
SEARCH_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "any",
    "are",
    "before",
    "between",
    "could",
    "dated",
    "did",
    "does",
    "earliest",
    "from",
    "have",
    "into",
    "its",
    "modern",
    "name",
    "origin",
    "other",
    "relationship",
    "search",
    "source",
    "term",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

_NON_HYPOTHESIS = re.compile(
    r"\b(?:find|locate|locating|retrieve|search|seek|verify|verification|"
    r"check|access|trace|tracing|prioritize|"
    r"should be sought|should be accessed|should be verified|remains unchecked|"
    r"would test|would adjudicate|source reliability|primary sources|"
    r"targeted search|unresolved|unverified source)\b",
    re.IGNORECASE,
)


def _hypothesis_key(statement: str) -> str:
    return " ".join(re.findall(r"[\w]+", statement.casefold()))


def _relevance_windows(
    source: Source, question: str, hypotheses: list[dict], query: str
) -> list[dict]:
    words = " ".join(
        [
            question,
            query,
            *(item["statement"] for item in hypotheses if item["status"] != "rejected"),
        ]
    )
    terms = {
        word.lower()
        for word in re.findall(r"[^\W_]+", words)
        if len(word) >= 3 and word.lower() not in SEARCH_STOPWORDS
    }
    if not terms:
        return []

    # Join adjacent body fragments only across whitespace. Furniture breaks a body run.
    body_spans: list[list[int]] = []
    for region in sorted(source.content_regions, key=lambda item: item.start):
        if region.region != QuoteRegion.ARTICLE_BODY:
            continue
        if body_spans and not source.content[body_spans[-1][1] : region.start].strip():
            body_spans[-1][1] = max(body_spans[-1][1], region.end)
        else:
            body_spans.append([region.start, region.end])

    hits: list[tuple[int, int, int, int, int]] = []
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        matches = [
            (span_start + match.start(), span_start + match.end(), span_start, span_end)
            for span_start, span_end in body_spans
            for match in pattern.finditer(source.content[span_start:span_end])
        ]
        hits.extend((len(matches), -len(term), *match) for match in matches)

    windows: list[tuple[int, int]] = []
    for _, _, hit_start, hit_end, span_start, span_end in sorted(hits):
        start = max(span_start, hit_start - WINDOW_CONTEXT)
        end = min(span_end, hit_end + WINDOW_CONTEXT)
        merged: list[tuple[int, int]] = []
        for old_start, old_end in sorted([*windows, (start, end)]):
            if merged and old_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], old_end))
            else:
                merged.append((old_start, old_end))
        if sum(end - start for start, end in merged) <= ASSESSMENT_LIMIT:
            windows = merged
    return [
        {"start": start, "end": end, "text": source.content[start:end]} for start, end in windows
    ]


class _AttemptLogger:
    def __init__(self, researcher: Researcher, action: str):
        self.researcher = researcher
        self.action = action
        self.rates: dict[int, tuple[float, float]] = {}

    def started(
        self,
        model: str,
        local_input_tokens: int,
        provider: str = "unknown",
        role: str = "primary",
        attempt_order: int = 1,
        retry_reason: str = "initial",
    ) -> int:
        researcher = self.researcher
        attempt_id = researcher.db.start_provider_attempt(
            researcher.run_id,
            self.action,
            model,
            local_input_tokens,
            provider,
            role,
            attempt_order,
            retry_reason,
        )
        config = researcher.config
        rates = (config.input_cost_per_million, config.output_cost_per_million)
        if self.action == "review":
            if role == "fallback":
                rates = (
                    config.review_fallback_input_cost_per_million,
                    config.review_fallback_output_cost_per_million,
                )
            else:
                rates = (
                    config.input_cost_per_million
                    if config.review_primary_input_cost_per_million is None
                    else config.review_primary_input_cost_per_million,
                    config.output_cost_per_million
                    if config.review_primary_output_cost_per_million is None
                    else config.review_primary_output_cost_per_million,
                )
        self.rates[attempt_id] = rates
        return attempt_id

    def finished(
        self,
        attempt_id: int,
        outcome: str,
        input_tokens: int | None,
        output_tokens: int | None,
        local_output_tokens: int,
        diagnostics: dict,
    ) -> None:
        researcher = self.researcher
        input_rate, output_rate = self.rates.pop(attempt_id)
        cost = researcher.db.finish_provider_attempt(
            attempt_id,
            outcome,
            input_tokens,
            output_tokens,
            input_rate,
            output_rate,
            local_output_tokens,
            researcher.config.local_usage_multiplier,
            diagnostics,
        )
        researcher.cost += cost
        if researcher.limits.max_cost is not None and researcher.cost >= researcher.limits.max_cost:
            raise BudgetStop("max_cost")


@dataclass
class Limits:
    max_actions: int = 30
    max_cost: float | None = None
    max_minutes: float = 20
    min_yield: float = 0.05

    def __post_init__(self) -> None:
        if self.max_actions < 1 or self.max_minutes <= 0:
            raise ValueError("max-actions and max-minutes must be positive")
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max-cost must be positive")
        if not 0 <= self.min_yield <= 1:
            raise ValueError("min-yield must be between 0 and 1")


class Researcher:
    def __init__(
        self,
        db: Database,
        llm: StructuredLLM,
        retrieval: Retrieval,
        config: Config,
        question_id: int,
        limits: Limits,
        model: str | None = None,
        archive: Wayback | None = None,
        review_primary: OpenAICompatibleLLM | None = None,
        review_fallback: OpenAICompatibleLLM | None = None,
    ):
        self.db = db
        self.llm = llm
        self.review_primary = review_primary
        self.review_fallback = review_fallback
        self.retrieval = retrieval
        self.archive = (
            archive
            if archive is not None
            else (Wayback(retrieval) if isinstance(retrieval, BraveWeb) else None)
        )
        self.config = config
        self.question_id = question_id
        self.limits = limits
        self.model = model or config.research_model
        self.deadline = 0.0
        self.run_id = 0
        self.actions = 0
        self.cost = 0.0
        self.fetched = 0
        self.evidence_added = 0
        self.recent_yields: deque[int] = deque(maxlen=3)

    def _check(self) -> None:
        if self.actions >= self.limits.max_actions:
            raise BudgetStop("max_actions")
        if time.monotonic() >= self.deadline:
            raise BudgetStop("max_minutes")
        if self.limits.max_cost is not None and self.cost >= self.limits.max_cost:
            raise BudgetStop("max_cost")

    def _safe_completion_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        if isinstance(exc, StructuredOutputError):
            message = str(exc)
            for secret in (
                self.config.llm_api_key,
                self.config.review_primary_api_key,
                self.config.review_fallback_api_key,
            ):
                if secret:
                    message = message.replace(secret, "[redacted]")
            return f"{type(exc).__name__}: {message[:300]}"
        return type(exc).__name__

    @staticmethod
    def _safe_fetch_error(exc: Exception) -> str:
        if isinstance(exc, FetchFailure) and exc.status_code is not None:
            return f"HTTP {exc.status_code}"
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        if isinstance(exc, (FetchFailure, ValueError)):
            message = str(exc)
            if isinstance(exc, FetchFailure) and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9]{1,40}", message
            ):
                return message
            if message.startswith("Unsupported content type:"):
                match = re.match(r"Unsupported content type: ([\w.+/-]+)", message)
                return match.group(0)[:200] if match else "Unsupported content type"
            if message in {
                "Redirect without location",
                "Too many redirects",
                "Page exceeds 2 MB retrieval limit",
                "Fetched page contains no readable text",
            }:
                return message
        return type(exc).__name__

    def _record(
        self,
        action: str,
        detail: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        provider_tracked: bool = False,
    ) -> None:
        cost = (
            input_tokens * self.config.input_cost_per_million
            + output_tokens * self.config.output_cost_per_million
        ) / 1_000_000
        self.db.record_action(
            self.run_id,
            action,
            detail,
            input_tokens,
            output_tokens,
            cost,
            aggregate_usage=not provider_tracked,
        )
        self.actions += 1
        if not provider_tracked:
            self.cost += cost

    async def _complete(self, action: str, model: str, system: str, user: str, schema: type):
        self._check()
        llm = (self.review_primary or self.llm) if action == "review" else self.llm
        fallback = self.review_fallback if action == "review" else None
        provider_tracked = isinstance(llm, OpenAICompatibleLLM)
        if self.limits.max_cost is not None:
            prompt_bytes = (
                len(system.encode())
                + len(user.encode())
                + len(json.dumps(schema.model_json_schema()).encode())
                + 1024
            )
            attempts = getattr(llm, "max_retries", 0) + 1
            if provider_tracked:
                attempts += 1  # One possible schema correction request.
            max_output_tokens = (
                llm.structured_call_policy(model).max_completion_tokens
                if provider_tracked
                else 1500
            )
            primary_input_rate = self.config.input_cost_per_million
            primary_output_rate = self.config.output_cost_per_million
            if action == "review":
                if self.config.review_primary_input_cost_per_million is not None:
                    primary_input_rate = self.config.review_primary_input_cost_per_million
                if self.config.review_primary_output_cost_per_million is not None:
                    primary_output_rate = self.config.review_primary_output_cost_per_million
            reserve = (
                attempts
                * (prompt_bytes * primary_input_rate + max_output_tokens * primary_output_rate)
                / 1_000_000
            )
            if fallback:
                reserve += (
                    (fallback.max_retries + 1)
                    * (
                        prompt_bytes * self.config.review_fallback_input_cost_per_million
                        + max_output_tokens * self.config.review_fallback_output_cost_per_million
                    )
                    / 1_000_000
                )
            if self.cost + reserve > self.limits.max_cost:
                raise BudgetStop("max_cost")
        if provider_tracked:
            llm.attempt_observer = _AttemptLogger(self, action)
        try:
            call = (
                llm.complete(
                    model,
                    system,
                    user,
                    schema,
                    fallback=fallback,
                    fallback_model=self.config.review_fallback_model,
                )
                if fallback
                else llm.complete(model, system, user, schema)
            )
            result = await asyncio.wait_for(
                call,
                timeout=max(0.001, self.deadline - time.monotonic()),
            )
        except TimeoutError as exc:
            self._record(action, "failed: elapsed time limit", provider_tracked=provider_tracked)
            raise BudgetStop("max_minutes") from exc
        except Exception as exc:
            self._record(
                action,
                f"failed: {self._safe_completion_error(exc)}",
                getattr(exc, "input_tokens", 0),
                getattr(exc, "output_tokens", 0),
                provider_tracked=provider_tracked,
            )
            raise
        finally:
            if provider_tracked:
                llm.attempt_observer = None
        self._record(
            action,
            json.dumps(
                {"model": result.model or model, "output": result.value.model_dump(mode="json")},
                ensure_ascii=False,
            ),
            result.input_tokens,
            result.output_tokens,
            provider_tracked=provider_tracked,
        )
        if (
            self.limits.max_cost is not None
            and not provider_tracked
            and not (result.input_tokens or result.output_tokens)
        ):
            raise BudgetStop("missing_usage_for_cost_limit")
        return result.value

    def _context(self) -> dict:
        q = self.db.rows("SELECT * FROM questions WHERE id=?", (self.question_id,))
        if not q:
            raise ValueError(f"Unknown question ID {self.question_id}")
        hypotheses = [
            dict(row)
            for row in self.db.rows(
                "SELECT id,statement,status,rationale FROM hypotheses "
                "WHERE research_question_id=? AND archived_at IS NULL ORDER BY id",
                (self.question_id,),
            )
        ]
        evidence = [
            dict(row)
            for row in self.db.rows(
                "SELECT e.id,e.exact_quote,e.normalized_claim,e.evidence_type,e.confidence,"
                "e.quote_verified_date,e.quote_region,e.quote_context,e.term_sense,"
                "s.retrieved_url,s.page_publication_date,s.dating_notes FROM evidence e "
                "JOIN sources s ON s.id=e.source_id WHERE e.research_question_id=? "
                "ORDER BY e.id DESC",
                (self.question_id,),
            )
        ]
        relationships = [
            dict(row)
            for row in self.db.rows(
                "SELECT r.evidence_id,r.hypothesis_id,r.kind,r.rationale FROM relationships r "
                "JOIN evidence e ON e.id=r.evidence_id WHERE e.research_question_id=? "
                "ORDER BY r.id DESC",
                (self.question_id,),
            )
        ]
        pending = [
            dict(row)
            for row in self.db.rows(
                "SELECT q.id,q.query,q.rationale,q.gap,q.information_value,q.avenue,"
                "q.target_purpose,"
                "t.key AS source_target FROM queries q LEFT JOIN source_targets t "
                "ON t.id=q.source_target_id WHERE q.research_question_id=? "
                "AND q.status='active' AND q.executed_at IS NULL "
                "ORDER BY q.priority DESC,q.id LIMIT 10",
                (self.question_id,),
            )
        ]
        source_targets = [
            dict(row)
            for row in self.db.rows(
                "SELECT key,description,status,evidence_status,dating_status FROM source_targets "
                "WHERE research_question_id=? ORDER BY id",
                (self.question_id,),
            )
        ]
        leads = [
            dict(row)
            for row in self.db.rows(
                "SELECT l.kind,l.value,l.rationale,s.retrieved_url FROM leads l "
                "JOIN sources s ON s.id=l.source_id WHERE l.research_question_id=? "
                "ORDER BY l.id DESC LIMIT 30",
                (self.question_id,),
            )
        ]
        failed_sources = [
            dict(row)
            for row in self.db.rows(
                "SELECT c.url,c.fetch_error,c.failure_at FROM candidates c "
                "JOIN queries q ON q.id=c.query_id "
                "WHERE q.research_question_id=? AND c.fetch_error IS NOT NULL",
                (self.question_id,),
            )
        ]
        avenues = [
            dict(row)
            for row in self.db.rows(
                "SELECT avenue AS avenue_id,MIN(query) AS example_query,COUNT(*) searches "
                "FROM queries WHERE research_question_id=? "
                "AND status IN ('active','deferred') AND avenue!='' GROUP BY avenue "
                "ORDER BY MAX(id) DESC LIMIT 30",
                (self.question_id,),
            )
        ]
        return {
            "question": q[0]["question"],
            "hypotheses": hypotheses,
            "evidence": evidence,
            "relationships": relationships,
            "pending_queries": pending,
            "source_targets": source_targets,
            "leads": leads,
            "failed_sources": failed_sources,
            "avenues": avenues,
        }

    def _review_context(self, query_id: int) -> tuple[dict, dict]:
        def short(value: str | None, limit: int = 300) -> str:
            return (value or "")[:limit]

        qid = self.question_id
        question = self.db.rows("SELECT question FROM questions WHERE id=?", (qid,))[0][0]
        hypotheses = [
            dict(row)
            for row in self.db.rows(
                "SELECT id,statement,status FROM hypotheses WHERE research_question_id=? "
                "AND archived_at IS NULL ORDER BY id",
                (qid,),
            )
        ]
        targets = [
            dict(row)
            for row in self.db.rows(
                "SELECT key,status,evidence_status,dating_status,description FROM source_targets "
                "WHERE research_question_id=? AND (status='unresolved' OR dating_status='unresolved') "
                "ORDER BY id",
                (qid,),
            )
        ]
        for target in targets:
            target["description"] = short(target["description"], 250)
        reviewed = dict(
            self.db.rows(
                "SELECT id,query,avenue AS avenue_id FROM queries "
                "WHERE id=? AND research_question_id=?",
                (query_id, qid),
            )[0]
        )
        reviewed["query"] = short(reviewed["query"], 250)
        previous = self.db.rows(
            "SELECT created_at,content_json FROM reviews WHERE research_question_id=? "
            "ORDER BY id DESC LIMIT 1",
            (qid,),
        )
        last_review_at = previous[0]["created_at"] if previous else ""
        earlier = json.loads(previous[0]["content_json"]) if previous else {}
        conclusions = {
            key: [short(item, 220) for item in earlier.get(key, [])[:2]]
            for key in ("overclaims", "dating_concerns", "transmission_gaps")
        }
        evidence = [
            dict(row)
            for row in self.db.rows(
                "SELECT e.id,e.exact_quote,e.evidence_type,e.confidence,e.term_sense,"
                "e.quote_verified_date,e.primary_source_verified,e.created_at,"
                "s.retrieved_url,s.page_publication_date,t.key source_target "
                "FROM evidence e JOIN sources s ON s.id=e.source_id "
                "LEFT JOIN queries q ON q.id=e.discovered_by_query_id "
                "LEFT JOIN source_targets t ON t.id=q.source_target_id "
                "WHERE e.research_question_id=? ORDER BY e.id DESC",
                (qid,),
            )
        ]
        links = [
            dict(row)
            for row in self.db.rows(
                "SELECT r.evidence_id,r.hypothesis_id,r.kind FROM relationships r "
                "JOIN evidence e ON e.id=r.evidence_id WHERE e.research_question_id=? "
                "ORDER BY r.id DESC",
                (qid,),
            )
        ]
        by_evidence: dict[int, list[dict]] = {}
        for link in links:
            by_evidence.setdefault(link["evidence_id"], []).append(link)
        selected: dict[int, dict] = {}

        def add(item: dict) -> None:
            if len(selected) < 20:
                selected.setdefault(item["id"], item)

        def rank(item: dict) -> tuple:
            return (
                item["primary_source_verified"],
                item["quote_verified_date"] is not None,
                item["evidence_type"] in {"observed_usage", "demonstrated_transmission"},
                item["id"],
            )

        for item in evidence:
            if item["created_at"] > last_review_at:
                add(item)
                if len(selected) >= 8:
                    break
        for hypothesis in hypotheses:
            for kind in ("contradicts", "supports", "contextualizes"):
                matches = [
                    item
                    for item in evidence
                    if any(
                        link["hypothesis_id"] == hypothesis["id"] and link["kind"] == kind
                        for link in by_evidence.get(item["id"], [])
                    )
                ]
                for item in sorted(matches, key=rank, reverse=True)[:2]:
                    add(item)
        for item in sorted(evidence, key=rank, reverse=True):
            if item["primary_source_verified"]:
                add(item)
            if len(selected) >= 20:
                break
        for evidence_type in ("attributed_origin_claim", "observed_usage", "attributed_usage"):
            for item in sorted(
                (item for item in evidence if item["evidence_type"] == evidence_type),
                key=rank,
                reverse=True,
            )[:2]:
                add(item)
        chosen = []
        for item in selected.values():
            chosen.append(
                {
                    "id": item["id"],
                    "quote": short(item["exact_quote"], 700),
                    "type": item["evidence_type"],
                    "confidence": item["confidence"],
                    "term_sense": item["term_sense"],
                    "quote_verified_date": item["quote_verified_date"],
                    "page_publication_date": item["page_publication_date"],
                    "primary_source_verified": bool(item["primary_source_verified"]),
                    "source_target": item["source_target"],
                    "url": short(item["retrieved_url"], 220),
                    "relationships": by_evidence.get(item["id"], [])[:6],
                }
            )
        pending = [
            dict(row)
            for row in self.db.rows(
                "SELECT q.id,q.query,q.avenue,q.gap,q.target_purpose,"
                "t.key source_target FROM queries q "
                "LEFT JOIN source_targets t ON t.id=q.source_target_id "
                "WHERE q.research_question_id=? AND q.status='active' AND q.executed_at IS NULL "
                "ORDER BY q.priority DESC,q.id LIMIT 8",
                (qid,),
            )
        ]
        for item in pending:
            item["gap"] = short(item["gap"], 160)
        avenues = [
            dict(row)
            for row in self.db.rows(
                "SELECT avenue AS avenue_id,COUNT(*) searches FROM queries "
                "WHERE research_question_id=? AND status='active' AND avenue!='' "
                "GROUP BY avenue ORDER BY MAX(id) DESC LIMIT 12",
                (qid,),
            )
        ]
        context = {
            "question": question,
            "hypotheses": hypotheses,
            "source_targets": targets,
            "reviewed_query": reviewed,
            "evidence": chosen,
            "pending_queries": pending,
            "avenues": avenues,
            "previous_review_conclusions": conclusions,
        }

        def size() -> int:
            return len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))

        for key in ("avenues", "pending_queries", "evidence"):
            while size() > REVIEW_CONTEXT_LIMIT and context[key]:
                context[key].pop()
        if size() > REVIEW_CONTEXT_LIMIT:
            raise ValueError("Mandatory review context exceeds character budget")
        counts = {
            "evidence": {
                "included": len(context["evidence"]),
                "omitted": len(evidence) - len(context["evidence"]),
            },
            "relationships": {
                "included": sum(len(item["relationships"]) for item in context["evidence"]),
                "omitted": len(links)
                - sum(len(item["relationships"]) for item in context["evidence"]),
            },
            "hypotheses": {"included": len(hypotheses), "omitted": 0},
            "source_targets": {"included": len(targets), "omitted": 0},
            "pending_queries": {
                "included": len(context["pending_queries"]),
                "omitted": self.db.rows(
                    "SELECT COUNT(*) FROM queries WHERE research_question_id=? "
                    "AND status='active' AND executed_at IS NULL",
                    (qid,),
                )[0][0]
                - len(context["pending_queries"]),
            },
            "avenues": {
                "included": len(context["avenues"]),
                "omitted": self.db.rows(
                    "SELECT COUNT(DISTINCT avenue) FROM queries "
                    "WHERE research_question_id=? AND status='active' AND avenue!=''",
                    (qid,),
                )[0][0]
                - len(context["avenues"]),
            },
            "failed_sources": {
                "included": 0,
                "omitted": self.db.rows(
                    "SELECT COUNT(*) FROM candidates c JOIN queries q ON q.id=c.query_id "
                    "WHERE q.research_question_id=? AND c.fetch_error IS NOT NULL",
                    (qid,),
                )[0][0],
            },
            "leads": {
                "included": 0,
                "omitted": self.db.rows(
                    "SELECT COUNT(*) FROM leads WHERE research_question_id=?", (qid,)
                )[0][0],
            },
            "rejected_or_exhausted_queries": {
                "included": 0,
                "omitted": self.db.rows(
                    "SELECT COUNT(*) FROM queries WHERE research_question_id=? "
                    "AND status IN ('rejected_duplicate','exhausted')",
                    (qid,),
                )[0][0],
            },
        }
        return context, counts

    async def _plan(self) -> None:
        context = self._context()
        plan = await self._complete(
            "plan",
            self.config.router_model,
            PLAN_SYSTEM,
            json.dumps(context, ensure_ascii=False),
            QueryPlan,
        )
        self.db.persist_plan(
            self.question_id,
            [
                SearchQuery(
                    research_question_id=self.question_id,
                    query=item.query,
                    rationale=item.rationale,
                    gap=item.gap,
                    information_value=item.information_value,
                    novelty=item.novelty,
                    admission_basis=item.admission_basis,
                    source_target=item.source_target,
                    target_purpose=item.target_purpose,
                    generated_by=self.config.router_model,
                )
                for item in plan.queries
            ],
        )

    async def _search(self, query: dict) -> None:
        self._check()
        try:
            results = await asyncio.wait_for(
                self.retrieval.search(query["query"], count=5),
                timeout=max(0.001, self.deadline - time.monotonic()),
            )
        except TimeoutError as exc:
            self._record("search", f"query_id={query['id']} timed out")
            raise BudgetStop("max_minutes") from exc
        except Exception as exc:
            self._record("search", f"query_id={query['id']} failed: {type(exc).__name__}: {exc}")
            raise
        self._record("search", f"query_id={query['id']} results={len(results)}")
        self.db.persist_search_results(
            self.question_id,
            query["id"],
            [(item.url, item.title) for item in results[:5]],
        )

    async def _archive_lookup(self, target: dict) -> None:
        if self.archive is None:
            return
        self._check()
        try:
            snapshots = await asyncio.wait_for(
                self.archive.lookup(
                    target["original_url"],
                    target["start_year"],
                    target["end_year"],
                    target["start_date"],
                    target["end_date"],
                ),
                timeout=max(0.001, self.deadline - time.monotonic()),
            )
            detail = f"{len(snapshots)} useful snapshots"
        except ArchiveGap as exc:
            snapshots = []
            detail = str(exc)
        except TimeoutError:
            snapshots = []
            detail = "CDX lookup timed out"
        self.db.record_archive_lookup(
            self.question_id, target["query_id"], target["original_url"], snapshots, detail
        )
        self._record("archive_lookup", f"url={target['original_url']} {detail}")

    async def _one_candidate(self, candidate: dict) -> bool:
        source_id = candidate["source_id"]
        if source_id is None:
            self._check()
            try:
                if candidate["archive_timestamp"]:
                    if self.archive is None:
                        raise ArchiveGap("Wayback retrieval unavailable")
                    fetch = self.archive.fetch(
                        candidate["original_url"],
                        Snapshot(
                            candidate["archive_timestamp"],
                            candidate["resolved_original_url"],
                            candidate["url"],
                            "",
                        ),
                    )
                else:
                    fetch = self.retrieval.fetch(candidate["url"])
                source = await asyncio.wait_for(
                    fetch,
                    timeout=max(0.001, self.deadline - time.monotonic()),
                )
            except TimeoutError as exc:
                self._record("fetch", f"candidate_id={candidate['id']} timed out")
                self.db.record_fetch_failure(
                    candidate["id"], "Fetch timed out", retryable=True, final_url=candidate["url"]
                )
                raise BudgetStop("max_minutes") from exc
            except (ArchiveGap, FetchFailure, httpx.HTTPError, ValueError, OSError) as exc:
                if isinstance(exc, ArchiveGap):
                    status = exc.status_code
                    final_url = exc.final_url or candidate["url"]
                    redirects = exc.redirect_chain
                    retryable = exc.retryable
                elif isinstance(exc, FetchFailure):
                    status = exc.status_code
                    final_url = exc.final_url
                    redirects = exc.redirect_chain
                    retryable = exc.retryable
                elif isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    final_url = str(exc.response.url)
                    redirects = None
                    retryable = status == 429 or 500 <= status < 600
                else:
                    status = None
                    final_url = candidate["url"]
                    redirects = None
                    retryable = isinstance(exc, httpx.TransportError)
                reason = str(exc) if isinstance(exc, ArchiveGap) else self._safe_fetch_error(exc)
                self._record("fetch", f"candidate_id={candidate['id']} failed: {reason}")
                self.db.record_fetch_failure(
                    candidate["id"],
                    reason,
                    retryable=retryable,
                    status_code=status,
                    final_url=final_url,
                    redirect_chain=redirects,
                )
                return False
            source_id = self.db.attach_source(candidate["id"], source)
            self.fetched += 1
            self._record("fetch", f"source_id={source_id} url={source.retrieved_url}")
        else:
            row = self.db.rows("SELECT * FROM sources WHERE id=?", (source_id,))[0]
            source = Source.model_validate(
                dict(row)
                | {
                    "content_regions": json.loads(row["regions_json"]),
                    "redirect_history": json.loads(row["redirect_history_json"]),
                }
            )
        if self.db.document_already_assessed(source.content_hash, self.question_id):
            self.db.finish_candidate(candidate["id"])
            return False
        if source.source_type != "search_aggregator" and not any(
            region.region == QuoteRegion.ARTICLE_BODY for region in source.content_regions
        ):
            self.db.record_fetch_failure(
                candidate["id"],
                "No attributable article-body text was extracted",
                retryable=False,
                final_url=source.retrieved_url,
            )
            return False
        context = self._context()
        mode = "full_document"
        source_text: dict = {"text": source.content}
        if len(source.content) > ASSESSMENT_LIMIT:
            mode = "relevance_windows"
            query = self.db.rows("SELECT query FROM queries WHERE id=?", (candidate["query_id"],))[
                0
            ]
            windows = _relevance_windows(
                source, context["question"], context["hypotheses"], query[0]
            )
            if not windows:
                self.db.persist_candidate_assessment(candidate["id"], [], [], [], mode)
                self._record("assess", f"source_id={source_id} mode={mode} no relevant body terms")
                return True
            source_text = {"windows": windows}
        source_metadata = {
            "id": source_id,
            "url": source.retrieved_url,
            "title": source.title,
            "page_publication_date": source.page_publication_date,
            "source_type": source.source_type,
            "assessment_mode": mode,
        }
        if mode == "full_document":
            source_metadata["canonical_alias"] = source.canonical_url
            source_metadata["dating_notes"] = source.dating_notes
        try:
            assessment: PageAssessment = await self._complete(
                "assess",
                self.model,
                ASSESS_SYSTEM,
                json.dumps(
                    {
                        "question": context["question"],
                        "hypotheses": context["hypotheses"],
                        "source_targets": context["source_targets"],
                        "source": {**source_metadata, **source_text},
                    },
                    ensure_ascii=False,
                ),
                PageAssessment,
            )
        except BudgetStop:
            raise
        except Exception as exc:
            self.db.record_assessment_failure(candidate["id"], self._safe_completion_error(exc))
            raise BudgetStop("candidate_assessment_failed") from exc
        evidence_records: list[tuple[Evidence, list[tuple[int, str, str]]]] = []
        lead_records: list[ResearchLead] = []
        valid_ids = {h["id"] for h in context["hypotheses"]}
        if assessment.relevant:
            for item_evidence in assessment.evidence:
                if item_evidence.exact_quote not in source.content:
                    continue
                region = self.db.quote_region(source_id, item_evidence.exact_quote)
                if (
                    source.source_type == "search_aggregator"
                    or region != QuoteRegion.ARTICLE_BODY
                    or item_evidence.evidence_type.value
                    not in {
                        "observed_usage",
                        "attributed_usage",
                        "attributed_origin_claim",
                        "attributed_intent",
                        "interpretive_context",
                        "cultural_proximity",
                    }
                ):
                    target = self.db.quote_target(source_id, item_evidence.exact_quote)
                    lead_records.append(
                        ResearchLead(
                            research_question_id=self.question_id,
                            source_id=source_id,
                            kind="other",
                            value=item_evidence.exact_quote[:500],
                            rationale=(
                                f"Unverified {region.value if region else 'unknown'} excerpt; "
                                "not accepted as dated usage or transmission evidence."
                                + (f" Underlying source lead: {target}" if target else "")
                            ),
                        )
                    )
                    continue
                evidence_records.append(
                    (
                        Evidence(
                            source_id=source_id,
                            research_question_id=self.question_id,
                            exact_quote=item_evidence.exact_quote,
                            normalized_claim=item_evidence.exact_quote,
                            relevance=item_evidence.relevance,
                            evidence_type=item_evidence.evidence_type,
                            # Page metadata alone cannot establish a contemporary artifact.
                            contemporaneous=None,
                            confidence=item_evidence.confidence,
                            interpretation_notes=(
                                "Model interpretation, not verified source evidence: "
                                f"{item_evidence.interpretation_notes}"
                            ),
                            term_sense=item_evidence.term_sense,
                            discovered_by_query_id=candidate["query_id"],
                        ),
                        [],
                    )
                )
                for link in item_evidence.hypothesis_links:
                    hypothesis_id, kind, rationale = link.hypothesis_id, link.kind, link.rationale
                    if hypothesis_id not in valid_ids:
                        continue
                    # Model-proposed support cannot establish a lineage relationship.
                    if kind.value == "contextualizes":
                        evidence_records[-1][1].append((hypothesis_id, kind.value, rationale))
        query_records = [
            SearchQuery(
                research_question_id=self.question_id,
                query=lead.query,
                rationale=lead.rationale,
                gap=lead.gap,
                information_value=lead.information_value,
                novelty=lead.novelty,
                admission_basis=lead.admission_basis,
                source_target=lead.source_target,
                target_purpose=lead.target_purpose,
                generated_by=self.model,
            )
            for lead in assessment.new_queries
        ]
        for lead in assessment.leads:
            if lead.value.strip() and lead.value.strip() in source.content:
                lead_records.append(
                    ResearchLead(
                        research_question_id=self.question_id,
                        source_id=source_id,
                        kind=lead.kind,
                        value=lead.value.strip(),
                        rationale=lead.rationale,
                    )
                )
        added = self.db.persist_candidate_assessment(
            candidate["id"], evidence_records, lead_records, query_records, mode
        )
        self.evidence_added += added
        return True

    async def _interpret(self) -> None:
        context = self._context()
        if not context["evidence"]:
            return
        interpretation: Interpretation = await self._complete(
            "interpret",
            self.model,
            INTERPRET_SYSTEM,
            json.dumps(context, ensure_ascii=False),
            Interpretation,
        )
        valid_ids = {h["id"] for h in context["hypotheses"]}
        for update in interpretation.updates:
            if update.hypothesis_id in valid_ids:
                self.db.add_hypothesis_suggestion(
                    self.run_id,
                    update.hypothesis_id,
                    update.status.value,
                    update.rationale,
                    update.evidence_ids,
                    self.question_id,
                )
        for statement in interpretation.new_hypotheses:
            statement = statement.strip()
            if not statement or _NON_HYPOTHESIS.search(statement):
                continue
            existing = self.db.rows(
                "SELECT id,statement FROM hypotheses WHERE research_question_id=?",
                (self.question_id,),
            )
            if len(context["hypotheses"]) >= 8 or _hypothesis_key(statement) in {
                _hypothesis_key(row["statement"]) for row in existing
            }:
                continue
            screen: HypothesisScreen = await self._complete(
                "screen_hypothesis",
                self.model,
                SCREEN_SYSTEM,
                json.dumps(
                    {
                        "question": context["question"],
                        "existing_hypotheses": context["hypotheses"],
                        "proposal": statement,
                    },
                    ensure_ascii=False,
                ),
                HypothesisScreen,
            )
            if not screen.explanatory or not screen.relevant or screen.overlapping_ids:
                continue
            hypothesis_id = self.db.add_hypothesis(
                Hypothesis(
                    research_question_id=self.question_id,
                    statement=statement,
                    rationale="Proposed during interpretation; passed explanatory relevance and overlap screen.",
                )
            )
            context["hypotheses"].append(
                {"id": hypothesis_id, "statement": statement, "status": "open"}
            )

    async def _review(self, query_id: int, *, allow_repeat: bool = False) -> None:
        try:
            context, counts = self._review_context(query_id)
            prompt = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            self.db.record_action(
                self.run_id,
                "review_context",
                json.dumps(
                    {
                        "characters": len(prompt),
                        "estimated_tokens": (len(prompt) + 3) // 4,
                        "counts": counts,
                    }
                ),
            )
            review: AdversarialReview = await self._complete(
                "review",
                self.config.review_model,
                REVIEW_SYSTEM,
                prompt,
                AdversarialReview,
            )
        except BudgetStop as exc:
            if self.db.rows(
                "SELECT id FROM provider_attempts WHERE run_id=? AND logical_action='review' LIMIT 1",
                (self.run_id,),
            ):
                self.db.record_failed_review(self.run_id, self.question_id, query_id, str(exc))
            raise
        except Exception as exc:
            self.db.record_failed_review(
                self.run_id, self.question_id, query_id, self._safe_completion_error(exc)
            )
            raise BudgetStop("review_failed") from exc
        self.db.persist_review(
            self.run_id,
            self.question_id,
            query_id,
            review.model_dump_json(),
            [
                SearchQuery(
                    research_question_id=self.question_id,
                    query=lead.query,
                    rationale=lead.rationale,
                    gap=lead.gap,
                    information_value=lead.information_value,
                    novelty=lead.novelty,
                    admission_basis=lead.admission_basis,
                    source_target=lead.source_target,
                    target_purpose=lead.target_purpose,
                    generated_by=self.config.review_model,
                )
                for lead in review.next_queries
                if lead.information_value == "high"
            ][: self.config.review_query_limit],
            [(item.avenue_id, item.reason) for item in review.exhausted_avenues],
            allow_repeat=allow_repeat,
        )

    async def _maybe_review(self, *, final: bool = False) -> bool:
        state = self.db.review_cadence_state(self.question_id)
        query_id = self.db.review_due(self.question_id)
        material = bool(state["evidence_ids"] or state["target_keys"] or state["hypothesis_ids"])
        if query_id is None and (
            material or state["meaningful_actions"] >= self.config.review_action_threshold
        ):
            query_id = self.db.review_due(self.question_id, allow_reviewed=True)
        if query_id is None:
            return False
        if final:
            if not material:
                reason = "no_material_change"
            elif (
                self.actions >= self.limits.max_actions
                or time.monotonic() + 120 >= self.deadline
                or (self.limits.max_cost is not None and self.cost >= self.limits.max_cost)
            ):
                reason = "insufficient_budget"
            else:
                reason = "final_material_evidence"
        elif state["target_keys"]:
            reason = "source_target_changed"
        elif state["evidence_ids"]:
            reason = "important_evidence"
        elif state["hypothesis_ids"] and state["last_review_at"] is not None:
            reason = "hypothesis_changed"
        elif state["meaningful_actions"] >= self.config.review_action_threshold:
            reason = "action_threshold"
        else:
            reason = "below_action_threshold"
        if (
            reason not in {"no_material_change", "insufficient_budget", "below_action_threshold"}
            and state["last_review_at"] is not None
            and time.monotonic() + 120 >= self.deadline
        ):
            reason = "insufficient_budget"
        if reason in {"no_material_change", "insufficient_budget", "below_action_threshold"}:
            self.db.record_review_cadence(self.run_id, self.question_id, "skipped", reason, state)
            return False
        self.db.record_review_cadence(self.run_id, self.question_id, "triggered", reason, state)
        await self._review(
            query_id,
            allow_repeat=bool(
                self.db.rows("SELECT reviewed_at FROM queries WHERE id=?", (query_id,))[0][0]
            ),
        )
        return True

    async def run(self) -> int:
        self.deadline = time.monotonic() + self.limits.max_minutes * 60
        self._context()  # Check question before creating a run.
        self.run_id = self.db.start_run(
            ResearchRun(
                research_question_id=self.question_id,
                model=self.model,
                provider=self.config.llm_base_url,
            )
        )
        try:
            self.db.requeue_retryable_fetches(self.question_id)
            while True:
                self._check()
                candidate = self.db.pending_candidate(self.question_id)
                if candidate:
                    before = self.evidence_added
                    distinct = await self._one_candidate(dict(candidate))
                    if distinct:
                        self.recent_yields.append(min(1, self.evidence_added - before))
                    if self.evidence_added > before:
                        await self._interpret()
                    continue
                if await self._maybe_review():
                    continue
                archive_target = (
                    self.db.next_archive_lookup(self.question_id) if self.archive else None
                )
                if archive_target:
                    await self._archive_lookup(archive_target)
                    continue
                context = self._context()
                if (
                    len(self.recent_yields) == 3
                    and sum(self.recent_yields) / 3 < self.limits.min_yield
                    and not context["pending_queries"]
                ):
                    raise BudgetStop("low_marginal_yield")
                if not context["pending_queries"]:
                    await self._plan()
                    context = self._context()
                    if not context["pending_queries"]:
                        raise BudgetStop("no_new_queries")
                query = context["pending_queries"][0]
                await self._search(query)
        except BudgetStop as exc:
            if str(exc) not in {"review_failed", "candidate_assessment_failed"}:
                try:
                    await self._maybe_review(final=True)
                except BudgetStop:
                    pass
            self.db.finish_run(self.run_id, "stopped", str(exc))
        except KeyboardInterrupt:
            self.db.finish_run(self.run_id, "stopped", "user_interrupt")
            raise
        except Exception as exc:
            self.db.finish_run(self.run_id, "failed", f"{type(exc).__name__}: {exc}")
            raise
        return self.run_id
