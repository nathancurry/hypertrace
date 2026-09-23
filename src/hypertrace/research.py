"""Bounded, inspectable research loop."""

from __future__ import annotations

import asyncio
import json
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
    Interpretation,
    PageAssessment,
    QueryPlan,
    QuoteRegion,
    ResearchLead,
    ResearchRun,
    SearchQuery,
    Source,
)
from hypertrace.planner import ASSESS_SYSTEM, INTERPRET_SYSTEM, PLAN_SYSTEM, REVIEW_SYSTEM
from hypertrace.retrieval.base import Retrieval


class BudgetStop(Exception):
    pass


class _AttemptLogger:
    def __init__(self, researcher: Researcher, action: str):
        self.researcher = researcher
        self.action = action

    def started(self, model: str) -> int:
        return self.researcher.db.start_provider_attempt(self.researcher.run_id, self.action, model)

    def finished(
        self,
        attempt_id: int,
        outcome: str,
        input_tokens: int | None,
        output_tokens: int | None,
        diagnostics: dict,
    ) -> None:
        researcher = self.researcher
        cost = researcher.db.finish_provider_attempt(
            attempt_id,
            outcome,
            input_tokens,
            output_tokens,
            researcher.config.input_cost_per_million,
            researcher.config.output_cost_per_million,
            diagnostics,
        )
        researcher.cost += cost
        if input_tokens is None or output_tokens is None:
            researcher.unknown_spend = True
            if researcher.limits.max_cost is not None:
                raise BudgetStop("unknown_spend_for_cost_limit")
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
    ):
        self.db = db
        self.llm = llm
        self.retrieval = retrieval
        self.config = config
        self.question_id = question_id
        self.limits = limits
        self.model = model or config.research_model
        self.deadline = 0.0
        self.run_id = 0
        self.actions = 0
        self.cost = 0.0
        self.unknown_spend = False
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
        if isinstance(self.llm, OpenAICompatibleLLM) and isinstance(exc, StructuredOutputError):
            message = str(exc)
            if self.config.llm_api_key:
                message = message.replace(self.config.llm_api_key, "[redacted]")
            return f"{type(exc).__name__}: {message[:300]}"
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
        provider_tracked = isinstance(self.llm, OpenAICompatibleLLM)
        if self.limits.max_cost is not None:
            prompt_bytes = (
                len(system.encode())
                + len(user.encode())
                + len(json.dumps(schema.model_json_schema()).encode())
                + 1024
            )
            attempts = getattr(self.llm, "max_retries", 0) + 1
            if provider_tracked and schema is AdversarialReview:
                attempts += 1  # One possible schema correction request.
            max_output_tokens = self.llm.max_tokens_for(schema) if provider_tracked else 1500
            reserve = (
                attempts
                * (
                    prompt_bytes * self.config.input_cost_per_million
                    + max_output_tokens * self.config.output_cost_per_million
                )
                / 1_000_000
            )
            if self.cost + reserve > self.limits.max_cost:
                raise BudgetStop("max_cost")
        if provider_tracked:
            self.llm.attempt_observer = _AttemptLogger(self, action)
        try:
            result = await asyncio.wait_for(
                self.llm.complete(model, system, user, schema),
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
                self.llm.attempt_observer = None
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
        if self.limits.max_cost is not None and self.unknown_spend:
            raise BudgetStop("unknown_spend_for_cost_limit")
        if self.limits.max_cost is not None and not (result.input_tokens or result.output_tokens):
            raise BudgetStop("missing_usage_for_cost_limit")
        return result.value

    def _context(self) -> dict:
        q = self.db.rows("SELECT * FROM questions WHERE id=?", (self.question_id,))
        if not q:
            raise ValueError(f"Unknown question ID {self.question_id}")
        hypotheses = [
            dict(row)
            for row in self.db.rows(
                "SELECT id,statement,status,rationale FROM hypotheses WHERE research_question_id=?",
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
                "SELECT id,query,rationale FROM queries WHERE research_question_id=? "
                "AND executed_at IS NULL ORDER BY id LIMIT 10",
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
        return {
            "question": q[0]["question"],
            "hypotheses": hypotheses,
            "evidence": evidence,
            "relationships": relationships,
            "pending_queries": pending,
            "leads": leads,
            "failed_sources": failed_sources,
        }

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

    async def _one_candidate(self, candidate: dict) -> bool:
        source_id = candidate["source_id"]
        if source_id is None:
            self._check()
            try:
                source = await asyncio.wait_for(
                    self.retrieval.fetch(candidate["url"]),
                    timeout=max(0.001, self.deadline - time.monotonic()),
                )
            except TimeoutError as exc:
                self._record("fetch", f"url={candidate['url']} timed out")
                self.db.record_fetch_failure(candidate["id"], "Fetch timed out", retryable=True)
                raise BudgetStop("max_minutes") from exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (429, 500, 502, 503, 504):
                    self._record("fetch", f"url={candidate['url']} retry later: {exc}")
                    self.db.record_fetch_failure(candidate["id"], str(exc), retryable=True)
                    raise
                self._record("fetch", f"url={candidate['url']} skipped: {exc}")
                self.db.record_fetch_failure(candidate["id"], str(exc), retryable=False)
                return False
            except httpx.TransportError as exc:
                self._record("fetch", f"url={candidate['url']} retry later: {exc}")
                self.db.record_fetch_failure(candidate["id"], str(exc), retryable=True)
                raise
            except (ValueError, OSError) as exc:
                self._record(
                    "fetch", f"url={candidate['url']} skipped: {type(exc).__name__}: {exc}"
                )
                self.db.record_fetch_failure(candidate["id"], str(exc), retryable=False)
                return False
            except Exception as exc:
                self._record("fetch", f"url={candidate['url']} failed: {type(exc).__name__}: {exc}")
                self.db.record_fetch_failure(
                    candidate["id"], f"{type(exc).__name__}: {exc}", retryable=True
                )
                raise
            source_id = self.db.attach_source(candidate["id"], source)
            self.fetched += 1
            self._record("fetch", f"source_id={source_id} url={source.retrieved_url}")
        else:
            row = self.db.rows("SELECT * FROM sources WHERE id=?", (source_id,))[0]
            source = Source.model_validate(
                dict(row) | {"content_regions": json.loads(row["regions_json"])}
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
            )
            return False
        if len(source.content) > 16_000:
            self.db.record_fetch_failure(
                candidate["id"],
                "Fetched text exceeds the 16,000-character assessment window",
                retryable=False,
            )
            return False
        context = self._context()
        assessment: PageAssessment = await self._complete(
            "assess",
            self.model,
            ASSESS_SYSTEM,
            json.dumps(
                {
                    "question": context["question"],
                    "hypotheses": context["hypotheses"],
                    "source": {
                        "id": source_id,
                        "url": source.retrieved_url,
                        "canonical_alias": source.canonical_url,
                        "title": source.title,
                        "page_publication_date": source.page_publication_date,
                        "dating_notes": source.dating_notes,
                        "source_type": source.source_type,
                        "text": source.content[:16000],
                    },
                },
                ensure_ascii=False,
            ),
            PageAssessment,
        )
        evidence_records: list[tuple[Evidence, list[tuple[int, str, str]]]] = []
        lead_records: list[ResearchLead] = []
        valid_ids = {h["id"] for h in context["hypotheses"]}
        if assessment.relevant:
            for item_evidence in assessment.evidence:
                if item_evidence.exact_quote not in source.content[:16000]:
                    continue
                region = self.db.quote_region(source_id, item_evidence.exact_quote)
                if (
                    source.source_type == "search_aggregator"
                    or region != QuoteRegion.ARTICLE_BODY
                    or item_evidence.evidence_type.value
                    not in {"observed_usage", "cultural_proximity"}
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
                                "Unverified model claim: "
                                f"{item_evidence.normalized_claim}. "
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
            candidate["id"], evidence_records, lead_records, query_records
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
            if statement.strip():
                self.db.add_hypothesis(
                    Hypothesis(
                        research_question_id=self.question_id,
                        statement=statement.strip(),
                        rationale="Proposed during interpretation.",
                    )
                )

    async def _review(self, query_id: int) -> None:
        try:
            review: AdversarialReview = await self._complete(
                "review",
                self.config.review_model,
                REVIEW_SYSTEM,
                json.dumps(self._context(), ensure_ascii=False),
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
                    generated_by=self.config.review_model,
                )
                for lead in review.next_queries
            ],
        )

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
                review_query_id = self.db.review_due(self.question_id)
                if review_query_id is not None:
                    await self._review(review_query_id)
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
            self.db.finish_run(self.run_id, "stopped", str(exc))
        except KeyboardInterrupt:
            self.db.finish_run(self.run_id, "stopped", "user_interrupt")
            raise
        except Exception as exc:
            self.db.finish_run(self.run_id, "failed", f"{type(exc).__name__}: {exc}")
            raise
        return self.run_id
