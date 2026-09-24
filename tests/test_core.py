from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import replace

import httpx
import pytest

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.base import LLMResult
from hypertrace.llm.openai_compatible import OpenAICompatibleLLM, StructuredOutputError
from hypertrace.models import (
    AdversarialReview,
    ContentRegion,
    Evidence,
    Hypothesis,
    Interpretation,
    PageAssessment,
    QuoteRegion,
    ResearchLead,
    ResearchQuestion,
    ResearchRun,
    SearchQuery,
    Source,
    TermSense,
)
from hypertrace.reports import markdown_report
from hypertrace.research import Limits, Researcher
from hypertrace.retrieval.base import SearchResult
from hypertrace.retrieval.web import BraveWeb, canonicalize_url


def _source(url: str, content: str) -> Source:
    return Source(
        canonical_url=canonicalize_url(url),
        retrieved_url=url,
        content=content,
        content_regions=[ContentRegion(start=0, end=len(content), region=QuoteRegion.ARTICLE_BODY)],
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def test_persistence_provenance_and_revision_integrity(tmp_path):
    with Database(tmp_path / "research.db") as db:
        q1 = db.add_question(ResearchQuestion(question="Where did this name originate?"))
        q2 = db.add_question(ResearchQuestion(question="Was the other name borrowed?"))
        h2 = db.add_hypothesis(Hypothesis(research_question_id=q2, statement="It was borrowed."))
        h1 = db.add_hypothesis(Hypothesis(research_question_id=q1, statement="The term was used."))
        query = db.add_query(SearchQuery(research_question_id=q1, query="name origin"))
        wrong_query = db.add_query(SearchQuery(research_question_id=q2, query="other origin"))
        first = _source("https://example.org/a?utm_source=x", "This names the style hyperpop.")
        first_id = db.add_source(first)
        assert db.add_source(first) == first_id
        assert db.add_source(_source("https://example.org/a", first.content)) == first_id
        metadata_revision = first.model_copy(
            update={"document_hash": "a" * 64, "page_publication_date": "2001-01-01"}
        )
        assert db.add_source(metadata_revision) != first_id
        with pytest.raises(ValueError, match="hash"):
            db.add_source(first.model_copy(update={"content_hash": "wrong"}))
        changed = _source("https://example.org/a", "This names the style hyper-pop.")
        assert db.add_source(changed) != first_id
        with pytest.raises(ValueError, match="verbatim"):
            db.add_evidence(
                Evidence(
                    source_id=first_id,
                    research_question_id=q1,
                    exact_quote="Search snippet only",
                    normalized_claim="Claim",
                    evidence_type="observed_usage",
                )
            )
        with pytest.raises(ValueError, match="Discovery query"):
            db.add_evidence(
                Evidence(
                    source_id=first_id,
                    research_question_id=q1,
                    exact_quote="names the style hyperpop",
                    normalized_claim="names the style hyperpop",
                    evidence_type="observed_usage",
                    discovered_by_query_id=wrong_query,
                )
            )
        evidence_id = db.add_evidence(
            Evidence(
                source_id=first_id,
                research_question_id=q1,
                exact_quote="names the style hyperpop",
                normalized_claim="names the style hyperpop",
                evidence_type="observed_usage",
                discovered_by_query_id=query,
            )
        )
        with pytest.raises(ValueError, match="Normalized claim"):
            db.add_evidence(
                Evidence(
                    source_id=first_id,
                    research_question_id=q1,
                    exact_quote="names the style hyperpop",
                    normalized_claim="This was the first use of hyperpop.",
                    evidence_type="observed_usage",
                )
            )
        with pytest.raises(ValueError, match="same research question"):
            db.add_relationship(evidence_id, h2, "supports", "Cross-question link")
        with pytest.raises(ValueError, match="cannot promote"):
            db.add_relationship(evidence_id, h1, "demonstrated_transmission", "Overclaim")
        assert db.rows("SELECT quote_start FROM evidence WHERE id=?", (evidence_id,))[0][0] == 5
        report = markdown_report(db, q1)
        assert "Inconclusive" in report
        assert "names the style hyperpop" in report
        assert first.content_hash in report
        assert (
            db.rows("SELECT discovered_by_query_id FROM evidence WHERE id=?", (evidence_id,))[0][0]
            == query
        )


def test_structured_llm_retries_invalid_json_and_stops():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 8000
        assert "max_tokens" not in body
        assert "reasoning_effort" not in body
        content = '{"relevant": true, "reason": "ok"}' if calls == 2 else "not json"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=1)
            result = await llm.complete("model", "system", "user", PageAssessment)
            assert result.value.relevant is True
            assert result.input_tokens == 20  # Both attempts count toward cost.
        assert calls == 2
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "bad"}}]})
            )
        ) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=1)
            with pytest.raises(StructuredOutputError, match="bounded retries"):
                await llm.complete("model", "system", "user", PageAssessment)

    asyncio.run(exercise())


def test_page_assessment_valid_json_schema_error_gets_one_repair_call():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = (
            '{"relevant": "unknown", "reason": "unclear"}'
            if len(requests) == 1
            else '{"relevant": false, "reason": "unclear"}'
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=0)
            result = await llm.complete("model", "system", "page", PageAssessment)
            assert result.value.relevant is False

    asyncio.run(exercise())
    assert len(requests) == 2
    assert all(body["max_completion_tokens"] == 8000 for body in requests)
    assert '"relevant": "unknown"' in requests[1]["messages"][1]["content"]


def test_web_fetch_preserves_metadata_caveat_and_canonical_url():
    html = """<html><head><title>Early usage</title>
    <meta property="article:published_time" content="2001-03-04T11:00:00Z">
    <link rel="canonical" href="https://example.org/story">
    </head><body><article><p>The term hyperpop appears here.</p></article></body></html>"""

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, text=html, headers={"content-type": "text/html; charset=utf-8"}
                )
            )
        ) as client:
            page = await BraveWeb("key", client).fetch("https://example.org/story?utm_source=x")
        assert page.canonical_url == "https://example.org/story"
        assert page.page_publication_date == "2001-03-04"
        assert "unverified" in page.dating_notes
        assert "The term hyperpop appears here." in page.content
        assert page.content_hash == hashlib.sha256(page.content.encode()).hexdigest()
        assert page.document_hash == hashlib.sha256(html.encode()).hexdigest()

    asyncio.run(exercise())


class FakeRetrieval:
    def __init__(self):
        self.fetches = 0

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                url="https://example.org/page",
                title="Page",
                snippet="Invented influence in snippet",
            )
        ]

    async def fetch(self, url: str) -> Source:
        self.fetches += 1
        return _source(url, "A 2001 page uses the word hyperpop.")


class FakeLLM:
    async def complete(self, model, system, user, schema):
        if schema is PageAssessment:
            payload = {
                "relevant": True,
                "reason": "Direct usage",
                "evidence": [
                    {
                        "exact_quote": "uses the word hyperpop",
                        "normalized_claim": "The fetched page uses hyperpop.",
                        "evidence_type": "observed_usage",
                        "hypothesis_links": [],
                    }
                ],
                "leads": [
                    {"kind": "term", "value": "hyperpop", "rationale": "Search for an earlier use."}
                ],
            }
        elif schema is Interpretation:
            payload = {"updates": [], "new_hypotheses": []}
        elif schema is AdversarialReview:
            payload = {"transmission_gaps": ["No borrowing statement found."], "next_queries": []}
        else:
            payload = {"queries": []}
        return LLMResult(
            value=schema.model_validate(payload), input_tokens=10, output_tokens=5, model=model
        )


def _config(tmp_path) -> Config:
    return Config(
        db_path=tmp_path / "research.db",
        llm_base_url="https://llm.example/v1",
        llm_api_key="key",
        router_model="cheap",
        research_model="cheap",
        review_model="review",
        brave_api_key="key",
        input_cost_per_million=1,
        output_cost_per_million=1,
        min_yield=0,
        json_mode=True,
    )


class WindowLLM(FakeLLM):
    def __init__(self, quote: str | None = None):
        self.prompts = []
        self.quote = quote

    async def complete(self, model, system, user, schema):
        if schema is PageAssessment:
            self.prompts.append(json.loads(user))
            evidence = (
                [
                    {
                        "exact_quote": self.quote,
                        "normalized_claim": self.quote,
                        "evidence_type": "observed_usage",
                    }
                ]
                if self.quote
                else []
            )
            return LLMResult(
                value=PageAssessment(relevant=bool(evidence), reason="checked", evidence=evidence),
                input_tokens=10,
                output_tokens=5,
            )
        return await super().complete(model, system, user, schema)


def _assess_stored_page(
    db,
    config,
    source,
    llm,
    *,
    question="Where was zephyrcore used?",
    query="zephyrcore history",
    hypothesis=None,
):
    qid = db.add_question(ResearchQuestion(question=question))
    if hypothesis:
        db.add_hypothesis(Hypothesis(research_question_id=qid, statement=hypothesis))
    query_id = db.add_query(SearchQuery(research_question_id=qid, query=query))
    db.mark_query_executed(query_id)
    db.add_candidate(query_id, source.retrieved_url, source.title)
    candidate_id = db.pending_candidate(qid)["id"]
    db.set_candidate_source(candidate_id, db.add_source(source))
    retrieval = FakeRetrieval()
    researcher = Researcher(db, llm, retrieval, config, qid, Limits(max_actions=1))
    asyncio.run(researcher.run())
    assert retrieval.fetches == 0
    return qid, candidate_id


def test_long_page_one_relevant_body_occurrence_is_assessed(tmp_path):
    config = _config(tmp_path)
    navigation = "Zephyrcore appears in navigation. "
    body = "filler " * 2600 + "An article discusses zephyrcore here."
    content = navigation + body
    source = _source("https://example.org/long-one", content).model_copy(
        update={
            "content_regions": [
                ContentRegion(start=0, end=len(navigation), region=QuoteRegion.NAVIGATION),
                ContentRegion(
                    start=len(navigation), end=len(content), region=QuoteRegion.ARTICLE_BODY
                ),
            ],
        }
    )
    llm = WindowLLM()
    with Database(config.db_path) as db:
        _, candidate_id = _assess_stored_page(db, config, source, llm)
        row = db.rows(
            "SELECT assessed_at,assessment_mode FROM candidates WHERE id=?", (candidate_id,)
        )[0]
        assert row["assessed_at"] and row["assessment_mode"] == "relevance_windows"
        windows = llm.prompts[0]["source"]["windows"]
        assert len(windows) == 1
        assert "zephyrcore here" in windows[0]["text"]
        assert "navigation" not in windows[0]["text"]
        assert windows[0]["text"] == content[windows[0]["start"] : windows[0]["end"]]
        assert sum(len(window["text"]) for window in windows) <= 16_000
        assert "text" not in llm.prompts[0]["source"]


def test_long_page_nearby_occurrences_merge_windows(tmp_path):
    config = _config(tmp_path)
    content = "filler " * 2500 + "zephyrcore " + "context " * 20 + "zephyrcore" + " tail " * 100
    llm = WindowLLM()
    with Database(config.db_path) as db:
        source = _source("https://example.org/long-nearby", content)
        _assess_stored_page(db, config, source, llm, question="Where was the style discussed?")
        windows = llm.prompts[0]["source"]["windows"]
        assert len(windows) == 1
        assert windows[0]["text"].count("zephyrcore") == 2
        assert windows[0]["start"] <= content.index("zephyrcore")
        assert windows[0]["end"] >= content.rindex("zephyrcore") + len("zephyrcore")


def test_long_page_without_relevant_body_term_skips_model(tmp_path):
    config = _config(tmp_path)
    navigation = "zephyrcore "
    content = navigation + "unrelated article body " * 1100
    source = _source("https://example.org/long-unrelated", content).model_copy(
        update={
            "content_regions": [
                ContentRegion(start=0, end=len(navigation), region=QuoteRegion.NAVIGATION),
                ContentRegion(
                    start=len(navigation), end=len(content), region=QuoteRegion.ARTICLE_BODY
                ),
            ],
        }
    )
    llm = WindowLLM()
    with Database(config.db_path) as db:
        _, candidate_id = _assess_stored_page(db, config, source, llm)
        assert llm.prompts == []
        row = db.rows(
            "SELECT assessed_at,assessment_mode FROM candidates WHERE id=?", (candidate_id,)
        )[0]
        assert row["assessed_at"] and row["assessment_mode"] == "relevance_windows"
        assert not db.rows("SELECT id FROM evidence")


def test_long_page_excerpt_validates_against_full_stored_source(tmp_path):
    config = _config(tmp_path)
    quote = "This article names zephyrcore as a style."
    content = "earlier text " * 1500 + quote + " following context " * 40
    llm = WindowLLM(quote)
    with Database(config.db_path) as db:
        source = _source("https://example.org/long-evidence", content)
        qid, candidate_id = _assess_stored_page(
            db,
            config,
            source,
            llm,
            question="What musical style was discussed?",
            query="historical discussion",
            hypothesis="Zephyrcore was named as a style.",
        )
        evidence = db.rows("SELECT * FROM evidence WHERE research_question_id=?", (qid,))[0]
        assert evidence["quote_start"] == content.index(quote) > 16_000
        assert evidence["quote_region"] == "article_body"
        assert quote in evidence["quote_context"]
        assert evidence["quote_verified_date"] is None
        assert (
            db.rows("SELECT assessment_mode FROM candidates WHERE id=?", (candidate_id,))[0][0]
            == "relevance_windows"
        )


def test_bounded_run_resumes_candidate_and_never_uses_snippet_as_evidence(tmp_path):
    config = _config(tmp_path)
    retrieval = FakeRetrieval()
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="hyperpop date"))
        first = asyncio.run(
            Researcher(db, FakeLLM(), retrieval, config, qid, Limits(max_actions=2)).run()
        )
        row = db.rows("SELECT * FROM runs WHERE id=?", (first,))[0]
        assert (row["actions_taken"], row["stop_reason"]) == (2, "max_actions")
        assert db.pending_candidate(qid)["source_id"] is not None
        second = asyncio.run(
            Researcher(db, FakeLLM(), retrieval, config, qid, Limits(max_actions=3)).run()
        )
        assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (second,))[0][0] == "max_actions"
        evidence = db.rows("SELECT * FROM evidence WHERE research_question_id=?", (qid,))
        assert len(evidence) == 1
        assert evidence[0]["exact_quote"] == "uses the word hyperpop"
        assert "snippet" not in evidence[0]["exact_quote"]
        assert evidence[0]["contemporaneous"] is None
        assert retrieval.fetches == 1
        assert db.pending_candidate(qid) is None
        assert (
            db.rows("SELECT kind,value FROM leads WHERE research_question_id=?", (qid,))[0]["value"]
            == "hyperpop"
        )
        assert json.loads(db.rows("SELECT content_json FROM reviews")[0][0])["transmission_gaps"]


def test_action_limit_stops_before_fetch(tmp_path):
    config = _config(tmp_path)
    retrieval = FakeRetrieval()
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="hyperpop date"))
        run_id = asyncio.run(
            Researcher(db, FakeLLM(), retrieval, config, qid, Limits(max_actions=1)).run()
        )
        assert retrieval.fetches == 0
        row = db.rows("SELECT actions_taken,stop_reason FROM runs WHERE id=?", (run_id,))[0]
        assert (row["actions_taken"], row["stop_reason"]) == (1, "max_actions")
        assert db.pending_candidate(qid) is not None


def test_cost_limited_run_stops_if_provider_omits_usage(tmp_path):
    class NoUsageLLM(FakeLLM):
        async def complete(self, model, system, user, schema):
            result = await super().complete(model, system, user, schema)
            result.input_tokens = 0
            result.output_tokens = 0
            return result

    config = _config(tmp_path)
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="hyperpop date"))
        run_id = asyncio.run(
            Researcher(
                db, NoUsageLLM(), FakeRetrieval(), config, qid, Limits(max_actions=10, max_cost=1)
            ).run()
        )
        row = db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (run_id,))[0]
        assert (row["status"], row["stop_reason"]) == ("stopped", "missing_usage_for_cost_limit")
        assert db.pending_candidate(qid) is not None
        assert not db.rows("SELECT * FROM evidence")


def test_page_date_regions_context_and_retrieved_citation(tmp_path):
    html = """<html><head><title>Hyperpop pioneer</title>
    <meta property="article:published_time" content="1995-01-02T00:00:00Z">
    <link rel="canonical" href="https://example.org/other"></head>
    <body><article><p>The term hyperpop appears in this later page.</p>
    <p>I named hyperpop</p><p>after Hyperballad.</p>
    <aside class="related-content">Hyperpop was named after Hyperballad.</aside>
    <section><h2>Related stories</h2><p>A retrospective hyperpop label.</p></section>
    </article></body></html>"""

    async def fetch():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=html, headers={"content-type": "text/html"})
            )
        ) as client:
            return await BraveWeb("key", client).fetch("https://example.org/retrieved")

    page = asyncio.run(fetch())
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        sid = db.add_source(page)
        assert page.page_publication_date == "1995-01-02"
        assert db.quote_region(sid, "Hyperpop pioneer") == QuoteRegion.TITLE
        assert (
            db.quote_region(sid, "Hyperpop was named after Hyperballad")
            == QuoteRegion.RELATED_CONTENT
        )
        assert (
            db.quote_region(sid, "A retrospective hyperpop label.") == QuoteRegion.RELATED_CONTENT
        )
        assert db.quote_region(sid, "I named hyperpop after Hyperballad.") == QuoteRegion.UNKNOWN
        with pytest.raises(ValueError, match="article-body"):
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote="Hyperpop pioneer",
                    normalized_claim="1995 genre usage",
                    evidence_type="observed_usage",
                )
            )
        quote = "The term hyperpop appears in this later page."
        db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                term_sense=TermSense.RETROSPECTIVE_LABEL,
            )
        )
        mirror = page.model_copy(
            update={
                "canonical_url": "https://mirror.example.org/copy",
                "retrieved_url": "https://mirror.example.org/copy",
            }
        )
        mirror_id = db.add_source(mirror)
        db.add_evidence(
            Evidence(
                source_id=mirror_id,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                term_sense=TermSense.RETROSPECTIVE_LABEL,
            )
        )
        report = markdown_report(db, qid)
        assert "No independently dated quote" in report
        assert "1995-01-02 (unverified for this quote)" in report
        assert "[Hyperpop pioneer](https://example.org/retrieved)" in report
        assert "[Hyperpop pioneer](https://example.org/other)" not in report
        assert "Context:" in report
        undated = report.split("## Evidence outside verified chronology", 1)[1].split(
            "## Evidence in relation", 1
        )[0]
        assert undated.count("- E") == 1
        assert "independence remains unresolved" in report


def test_verified_chronology_ignores_conflicting_page_dates(tmp_path):
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="When was the term used?"))
        later = _source("https://example.org/old-page", "Later verified use of hyperpop.")
        later = later.model_copy(update={"page_publication_date": "1990-01-01"})
        earlier = _source("https://example.org/new-page", "Earlier verified use of hyperpop.")
        earlier = earlier.model_copy(update={"page_publication_date": "2020-01-01"})
        for source, verified_date in ((later, "2005-01-01"), (earlier, "1995-01-01")):
            sid = db.add_source(source)
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=source.content,
                    normalized_claim=source.content,
                    evidence_type="observed_usage",
                    quote_verified_date=verified_date,
                    date_verification_note="Manually checked dated print issue.",
                    primary_source_verified=True,
                )
            )
        chronology = (
            markdown_report(db, qid)
            .split("## Verified chronology", 1)[1]
            .split("## Evidence outside verified chronology", 1)[0]
        )
        assert chronology.index("1995-01-01") < chronology.index("2005-01-01")
        assert "2020-01-01" not in chronology
        assert "1990-01-01" not in chronology


@pytest.mark.parametrize(
    ("quote", "category", "has_lead"),
    [
        (
            'The term "hyperpop" first appeared in print in October 1988, when Don Shewey used it.',
            "attributed_origin_claim",
            True,
        ),
        (
            "Philip Sherburne was describing PC Music as hyper-pop in Pitchfork as early as 2014.",
            "attributed_usage",
            True,
        ),
        (
            "Glenn McDonald coined the term hyperpop for the playlist in 2019.",
            "attributed_origin_claim",
            True,
        ),
        (
            "Björk intended Hyperballad as an exaggeration of emotion.",
            "attributed_intent",
            False,
        ),
    ],
)
def test_secondary_claims_keep_attribution_and_source_leads(tmp_path, quote, category, has_lead):
    path = tmp_path / "research.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Who first used hyperpop?"))
        sid = db.add_source(_source("https://example.org/later-article", quote))
        eid = db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
            )
        )
        assert db.rows("SELECT evidence_type FROM evidence WHERE id=?", (eid,))[0][0] == category
        assert bool(db.rows("SELECT id FROM leads WHERE source_id=?", (sid,))) == has_lead
        # Reopening also corrects older rows in place without changing their identity.
        db.conn.execute("UPDATE evidence SET evidence_type='observed_usage' WHERE id=?", (eid,))
        db.conn.commit()
    with Database(path) as db:
        row = db.rows("SELECT id,source_id,evidence_type FROM evidence WHERE id=?", (eid,))[0]
        assert (row["id"], row["source_id"], row["evidence_type"]) == (eid, sid, category)


def test_direct_dated_usage_only_enters_verified_chronology(tmp_path):
    with Database(tmp_path / "research.db") as db:
        qid = db.add_question(ResearchQuestion(question="When was hyperpop directly used?"))
        secondary = 'The term "hyperpop" first appeared in print in October 1988.'
        direct = "The hyperpop scene is growing."
        for url, quote, verified_date, primary in (
            ("https://example.org/2021-history", secondary, "2021-05-01", False),
            ("https://example.org/2014-article", direct, "2014-06-01", True),
        ):
            sid = db.add_source(_source(url, quote))
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    quote_verified_date=verified_date,
                    date_verification_note="Checked the dated source artifact.",
                    primary_source_verified=primary,
                )
            )
        report = markdown_report(db, qid)
        chronology = report.split("## Verified chronology", 1)[1].split(
            "## Evidence outside verified chronology", 1
        )[0]
        assert direct in chronology
        assert secondary not in chronology
        assert "1988" not in chronology
        assert "2014-06-01" in chronology
        assert "### Origin and coinage claims" in report
        assert secondary in report


def test_assessment_preserves_reported_usage_without_promoting_model_claim(tmp_path):
    quote = "Philip Sherburne was describing PC Music as hyper-pop in Pitchfork as early as 2014."

    class SecondaryLLM(WindowLLM):
        async def complete(self, model, system, user, schema):
            if schema is PageAssessment:
                return LLMResult(
                    value=PageAssessment.model_validate(
                        {
                            "relevant": True,
                            "reason": "historical lead",
                            "evidence": [
                                {
                                    "exact_quote": quote,
                                    "normalized_claim": "Sherburne directly used hyper-pop in 2014.",
                                    "evidence_type": "observed_usage",
                                }
                            ],
                        }
                    ),
                    input_tokens=10,
                    output_tokens=5,
                )
            return await super().complete(model, system, user, schema)

    config = _config(tmp_path)
    with Database(config.db_path) as db:
        source = _source("https://example.org/later-article", quote)
        qid, _ = _assess_stored_page(db, config, source, SecondaryLLM())
        row = db.rows(
            "SELECT evidence_type,normalized_claim,quote_verified_date,source_id FROM evidence "
            "WHERE research_question_id=?",
            (qid,),
        )[0]
        assert tuple(row)[:3] == ("attributed_usage", quote, None)
        assert db.rows(
            "SELECT id FROM leads WHERE kind='citation' AND source_id=?", (row["source_id"],)
        )


def test_search_aggregator_snippet_is_only_a_lead(tmp_path):
    html = """<html><body><main class="search-results"><article class="search-result">
    <a href="https://archive.example.org/old">Archived result</a>
    <div class="result-snippet">A 1995 source supposedly coined hyperpop.</div>
    </article></main></body></html>"""

    async def fetch():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=html, headers={"content-type": "text/html"})
            )
        ) as client:
            return await BraveWeb("key", client).fetch("https://example.org/results")

    page = asyncio.run(fetch())
    assert page.source_type == "search_aggregator"
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        sid = db.add_source(page)
        assert (
            db.quote_target(sid, "A 1995 source supposedly coined hyperpop.")
            == "https://archive.example.org/old"
        )
        with pytest.raises(ValueError, match="Aggregator"):
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote="A 1995 source supposedly coined hyperpop.",
                    normalized_claim="1995 origin",
                    evidence_type="observed_usage",
                )
            )

    unmarked_card = """<html><body><article><h2>
    <a href="https://archive.example.org/old">Old article</a></h2>
    <p>Hyperpop was allegedly coined in 1995.</p></article></body></html>"""

    async def fetch_unmarked():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, text=unmarked_card, headers={"content-type": "text/html"}
                )
            )
        ) as client:
            return await BraveWeb("key", client).fetch("https://example.org/archive")

    assert asyncio.run(fetch_unmarked()).source_type == "search_aggregator"


def test_candidate_and_review_commits_roll_back_on_interruption(tmp_path, monkeypatch):
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        hid = db.add_hypothesis(Hypothesis(research_question_id=qid, statement="A borrowed name."))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="hyperpop origin"))
        db.mark_query_executed(query_id)
        url = "https://example.org/article"
        db.add_candidate(query_id, url, "Article")
        candidate_id = db.pending_candidate(qid)["id"]
        sid = db.add_source(_source(url, "The article uses hyperpop."))
        db.set_candidate_source(candidate_id, sid)
        item = Evidence(
            source_id=sid,
            research_question_id=qid,
            exact_quote="uses hyperpop",
            normalized_claim="uses hyperpop",
            evidence_type="observed_usage",
        )
        lead = ResearchLead(research_question_id=qid, source_id=sid, kind="term", value="hyperpop")
        with monkeypatch.context() as patch:
            patch.setattr(
                db, "_add_lead_tx", lambda _: (_ for _ in ()).throw(RuntimeError("crash"))
            )
            with pytest.raises(RuntimeError, match="crash"):
                db.persist_candidate_assessment(
                    candidate_id, [(item, [(hid, "contextualizes", "context")])], [lead], []
                )
        assert not db.rows("SELECT * FROM evidence")
        assert not db.rows("SELECT * FROM relationships")
        assert db.pending_candidate(qid)["id"] == candidate_id
        db.persist_candidate_assessment(
            candidate_id, [(item, [(hid, "contextualizes", "context")])], [lead], []
        )
        assert db.pending_candidate(qid) is None
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="model", provider="test"))
        next_query = SearchQuery(research_question_id=qid, query="older hyperpop use")
        with monkeypatch.context() as patch:
            patch.setattr(
                db, "_add_query_tx", lambda _: (_ for _ in ()).throw(RuntimeError("crash"))
            )
            with pytest.raises(RuntimeError, match="crash"):
                db.persist_review(run_id, qid, query_id, "{}", [next_query])
        assert not db.rows("SELECT * FROM reviews")
        assert db.review_due(qid) == query_id
        db.persist_review(run_id, qid, query_id, "{}", [next_query])
        assert len(db.rows("SELECT * FROM reviews")) == 1
        assert db.review_due(qid) is None


def test_unverified_transmission_and_status_remain_provisional(tmp_path):
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="Did modern hyperpop borrow a name?"))
        hid = db.add_hypothesis(
            Hypothesis(research_question_id=qid, statement="It borrowed the name.")
        )
        sid = db.add_source(
            _source("https://example.org/post", "I named hyperpop after Hyperballad.")
        )
        with pytest.raises(ValueError, match="verified primary"):
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote="I named hyperpop after Hyperballad.",
                    normalized_claim="Direct borrowing",
                    evidence_type="demonstrated_transmission",
                    confidence="high",
                )
            )
        eid = db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote="hyperpop",
                normalized_claim="hyperpop",
                evidence_type="observed_usage",
                term_sense=TermSense.EARLIER_UNRELATED_USAGE,
            )
        )
        with pytest.raises(ValueError, match="Unrelated usage"):
            db.add_relationship(eid, hid, "supports", "Same word")
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="model", provider="test"))
        db.add_hypothesis_suggestion(run_id, hid, "supported", "Model inferred origin", [eid], qid)
        assert db.rows("SELECT status FROM hypotheses WHERE id=?", (hid,))[0][0] == "open"
        assert "not accepted" in markdown_report(db, qid)


def test_duplicate_zero_yield_candidates_do_not_block_pending_seed(tmp_path):
    class DuplicateRetrieval(FakeRetrieval):
        async def search(self, query: str, count: int = 5):
            return [SearchResult(url="https://example.org/useful", title="Useful")]

        async def fetch(self, url: str):
            self.fetches += 1
            content = (
                "The modern genre is called hyperpop."
                if url.endswith("useful")
                else "No useful claim."
            )
            return _source(url, content)

    class YieldLLM(FakeLLM):
        async def complete(self, model, system, user, schema):
            if schema is PageAssessment:
                payload = (
                    {
                        "relevant": True,
                        "reason": "usage",
                        "evidence": [
                            {
                                "exact_quote": "modern genre is called hyperpop",
                                "normalized_claim": "The modern genre is called hyperpop.",
                                "term_sense": "modern_genre",
                                "evidence_type": "observed_usage",
                            }
                        ],
                    }
                    if "modern genre" in json.loads(user)["source"]["text"]
                    else {"relevant": False, "reason": "no usage"}
                )
                return LLMResult(
                    value=schema.model_validate(payload), input_tokens=10, output_tokens=5
                )
            return await super().complete(model, system, user, schema)

    config = _config(tmp_path)
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        first = db.add_query(
            SearchQuery(research_question_id=qid, query="duplicate avenue", generated_by="seed")
        )
        second = db.add_query(
            SearchQuery(research_question_id=qid, query="useful avenue", generated_by="seed")
        )
        db.mark_query_executed(first)
        for index in range(3):
            db.add_candidate(first, f"https://example.org/duplicate-{index}", "Duplicate")
        run_id = asyncio.run(
            Researcher(
                db,
                YieldLLM(),
                DuplicateRetrieval(),
                config,
                qid,
                Limits(max_actions=20, min_yield=0.5),
            ).run()
        )
        assert db.rows("SELECT executed_at FROM queries WHERE id=?", (second,))[0][0]
        assert (
            db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0]
            != "low_marginal_yield"
        )
        assert len(db.rows("SELECT * FROM evidence")) == 1
        assert "3 capture(s); identical extracted text" in markdown_report(db, qid)


def test_provider_attempts_logged_separately_from_logical_action(tmp_path):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = '{"queries": []}' if calls == 2 else "not json"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=1)
            config = _config(tmp_path)
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                run_id = await Researcher(
                    db, llm, FakeRetrieval(), config, qid, Limits(max_actions=3)
                ).run()
                run = db.rows("SELECT * FROM runs WHERE id=?", (run_id,))[0]
                assert (run["actions_taken"], run["provider_requests"]) == (1, 2)
                assert (run["input_tokens"], run["output_tokens"]) == (20, 8)
                assert run["unknown_spend_requests"] == 0
                assert len(db.rows("SELECT * FROM provider_attempts")) == 2

    asyncio.run(exercise())


def test_unsupported_pdf_remains_an_explicit_gap(tmp_path):
    class PdfRetrieval(FakeRetrieval):
        async def fetch(self, url: str):
            raise ValueError("Unsupported content type: application/pdf")

    config = _config(tmp_path)
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="historical scan"))
        asyncio.run(
            Researcher(db, FakeLLM(), PdfRetrieval(), config, qid, Limits(max_actions=5)).run()
        )
        failure = db.rows("SELECT assessed_at,failure_at,fetch_error FROM candidates")[0]
        assert failure["assessed_at"] is None
        assert failure["failure_at"] is not None
        assert "application/pdf" in failure["fetch_error"]
        assert "application/pdf" in markdown_report(db, qid)


def test_unreported_provider_spend_is_not_counted_as_zero(tmp_path):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": '{"queries": []}'}}],
                    },
                )
            )
        ) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client)
            config = _config(tmp_path)
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                run_id = await Researcher(
                    db,
                    llm,
                    FakeRetrieval(),
                    config,
                    qid,
                    Limits(max_actions=3, max_cost=1),
                ).run()
                run = db.rows("SELECT * FROM runs WHERE id=?", (run_id,))[0]
                assert run["stop_reason"] == "unknown_spend_for_cost_limit"
                assert (run["provider_requests"], run["unknown_spend_requests"]) == (1, 1)
                assert db.rows("SELECT usage_reported FROM provider_attempts")[0][0] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("payload", "state", "reason"),
    [
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": '{"transmission_gaps": []}',
                        },
                        "finish_reason": "length",
                    }
                ]
            },
            "empty_string",
            "empty_string",
        ),
        ({"choices": [{"message": {"content": None}}]}, "null", "null"),
        ({"choices": []}, "missing", "no completion choices"),
        ({}, "missing", "no completion choices"),
    ],
)
def test_provider_empty_output_is_classified_before_validation(tmp_path, payload, state, reason):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
        ) as client:
            llm = OpenAICompatibleLLM(
                "https://llm.example/v1", "secret-key", client=client, max_retries=0
            )
            config = _config(tmp_path)
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                query_id = db.add_query(
                    SearchQuery(research_question_id=qid, query="hyperpop origin")
                )
                db.mark_query_executed(query_id)
                run_id = await Researcher(
                    db, llm, FakeRetrieval(), config, qid, Limits(max_actions=2)
                ).run()
                run = db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (run_id,))[0]
                assert (run["status"], run["stop_reason"]) == ("stopped", "review_failed")
                attempt = db.rows("SELECT outcome,diagnostics_json FROM provider_attempts")[0]
                assert attempt["outcome"] == "provider_output_failure"
                diagnostics = json.loads(attempt["diagnostics_json"])
                assert diagnostics["http_status"] == 200
                assert diagnostics["content_state"] == state
                assert diagnostics["choices_present"] is ("choices" in payload)
                assert diagnostics["choice_count"] == (
                    len(payload["choices"]) if "choices" in payload else None
                )
                assert diagnostics["response_keys"] == sorted(payload)
                assert diagnostics["message_keys"] == (
                    sorted(payload["choices"][0]["message"]) if payload.get("choices") else []
                )
                if state == "empty_string":
                    assert diagnostics["finish_reason"] == "length"
                    assert diagnostics["reasoning_content_chars"] > 0
                failed = db.rows("SELECT error,diagnostics_json FROM review_attempts")[0]
                assert reason in failed["error"]
                assert json.loads(failed["diagnostics_json"]) == [diagnostics]
                assert "secret-key" not in str(failed)
                assert db.review_due(qid) == query_id

    asyncio.run(exercise())


def test_failed_review_resume_retries_without_duplicating_evidence(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 8000
        assert "max_tokens" not in body
        content = "" if calls == 1 else '{"transmission_gaps": ["Unverified"], "next_queries": []}'
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM(
                "https://llm.example/v1", "secret-key", client=client, max_retries=0
            )
            config = _config(tmp_path)
            retrieval = FakeRetrieval()
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                query_id = db.add_query(
                    SearchQuery(research_question_id=qid, query="hyperpop origin")
                )
                db.mark_query_executed(query_id)
                source_id = db.add_source(_source("https://example.org/page", "Uses hyperpop."))
                evidence_id = db.add_evidence(
                    Evidence(
                        source_id=source_id,
                        research_question_id=qid,
                        exact_quote="Uses hyperpop.",
                        normalized_claim="Uses hyperpop.",
                        evidence_type="observed_usage",
                    )
                )
                first = await Researcher(
                    db, llm, retrieval, config, qid, Limits(max_actions=1)
                ).run()
                assert tuple(
                    db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (first,))[0]
                ) == (
                    "stopped",
                    "review_failed",
                )
                assert db.review_due(qid) == query_id
                second = await Researcher(
                    db, llm, retrieval, config, qid, Limits(max_actions=1)
                ).run()
                assert (
                    db.rows("SELECT stop_reason FROM runs WHERE id=?", (second,))[0][0]
                    == "max_actions"
                )
                assert db.review_due(qid) is None
                assert len(db.rows("SELECT * FROM reviews")) == 1
                assert [row["id"] for row in db.rows("SELECT id FROM evidence")] == [evidence_id]
                assert retrieval.fetches == 0
                assert calls == 2

    asyncio.run(exercise())


def test_reasoning_only_page_assessment_stops_and_resumes_without_duplicate_work(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["max_completion_tokens"] == 8000
        assert "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        if len(requests) <= 2:
            content = ""
            reasoning = "thinking through the page"
            finish_reason = "length"
            completion_tokens = 8000
        else:
            content = '{"relevant": false, "reason": "No usable evidence"}'
            reasoning = ""
            finish_reason = "stop"
            completion_tokens = 40
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": content, "reasoning_content": reasoning},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": completion_tokens},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM(
                "https://api.cheaperinference.com/v1", "key", client=client, max_retries=1
            )
            config = replace(_config(tmp_path), research_model="glm-5.3-flash")
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                query_id = db.add_query(
                    SearchQuery(research_question_id=qid, query="hyperpop origin")
                )
                db.add_candidate(query_id, "https://example.org/prior", "Prior")
                db.add_candidate(query_id, "https://example.org/page", "Page")
                db.mark_query_executed(query_id)
                prior = db.pending_candidate(qid)
                prior_source_id = db.attach_source(
                    prior["id"], _source(prior["url"], "Prior use of hyperpop.")
                )
                prior_evidence_id = db.add_evidence(
                    Evidence(
                        source_id=prior_source_id,
                        research_question_id=qid,
                        exact_quote="Prior use of hyperpop.",
                        normalized_claim="Prior use of hyperpop.",
                        evidence_type="observed_usage",
                    )
                )
                db.finish_candidate(prior["id"])
                candidate = db.pending_candidate(qid)
                source_id = db.attach_source(
                    candidate["id"], _source(candidate["url"], "A later discussion of hyperpop.")
                )
                retrieval = FakeRetrieval()
                first = await Researcher(
                    db, llm, retrieval, config, qid, Limits(max_actions=1)
                ).run()
                assert tuple(
                    db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (first,))[0]
                ) == ("stopped", "candidate_assessment_failed")
                pending = db.pending_candidate(qid)
                assert pending["id"] == candidate["id"]
                assert pending["source_id"] == source_id
                assert pending["assessed_at"] is None
                assert "empty_string" in pending["assessment_error"]
                assert len(requests) == 2
                assert "immediately with concise JSON matching the schema" in str(
                    requests[1]["messages"]
                )
                attempts = db.rows(
                    "SELECT outcome,diagnostics_json FROM provider_attempts WHERE run_id=?",
                    (first,),
                )
                assert len(attempts) == 2
                for attempt in attempts:
                    assert attempt["outcome"] == "provider_output_failure"
                    diagnostics = json.loads(attempt["diagnostics_json"])
                    assert diagnostics["http_status"] == 200
                    assert diagnostics["finish_reason"] == "length"
                    assert diagnostics["content_chars"] == 0
                    assert diagnostics["reasoning_content_chars"] > 0
                    assert diagnostics["reported_completion_tokens"] == 8000
                    assert diagnostics["max_completion_tokens"] == 8000
                    assert diagnostics["reasoning_effort_requested"] == "low"
                second = await Researcher(
                    db, llm, retrieval, config, qid, Limits(max_actions=1)
                ).run()
                assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (second,))[0][0] == (
                    "max_actions"
                )
                assert db.pending_candidate(qid) is None
                assert (
                    db.rows(
                        "SELECT assessment_error FROM candidates WHERE id=?", (candidate["id"],)
                    )[0][0]
                    is None
                )
                assert [row["id"] for row in db.rows("SELECT id FROM evidence")] == [
                    prior_evidence_id
                ]
                assert len(db.rows("SELECT id FROM queries")) == 1
                assert len(db.rows("SELECT id FROM sources")) == 2
                assert retrieval.fetches == 0
                assert len(requests) == 3

    asyncio.run(exercise())


def test_review_exception_stops_cleanly_and_remains_due(tmp_path):
    class BrokenReviewLLM(FakeLLM):
        async def complete(self, model, system, user, schema):
            if schema is AdversarialReview:
                raise RuntimeError("provider failed")
            return await super().complete(model, system, user, schema)

    config = _config(tmp_path)
    with Database(config.db_path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="hyperpop origin"))
        db.mark_query_executed(query_id)
        run_id = asyncio.run(
            Researcher(db, BrokenReviewLLM(), FakeRetrieval(), config, qid, Limits()).run()
        )
        row = db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (run_id,))[0]
        assert tuple(row) == ("stopped", "review_failed")
        failed = db.rows("SELECT error,diagnostics_json FROM review_attempts")[0]
        assert failed["error"] == "RuntimeError"
        assert json.loads(failed["diagnostics_json"]) == []
        assert db.review_due(qid) == query_id


def test_live_provider_review_settings_and_truncation():
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        assert body["max_completion_tokens"] == 8000
        assert "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": '{"transmission_gaps": []}',
                            "reasoning": "thinking",
                        },
                    }
                ],
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM(
                "https://api.cheaperinference.com/v1", "secret-key", client=client, max_retries=0
            )
            with pytest.raises(StructuredOutputError, match="completion token limit"):
                await llm.complete("glm-5.3-flash", "system", "user", AdversarialReview)

    asyncio.run(exercise())
    assert requests == 1


def test_review_schema_failure_gets_one_correction_call():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        content = (
            '{"overclaims": "unsupported"}'
            if len(requests) == 1
            else '{"overclaims": ["unsupported"]}'
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": content,
                            "reasoning_content": '{"overclaims": ["wrong channel"]}',
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=2)
            result = await llm.complete(
                "review-model", "review instructions", "research context", AdversarialReview
            )
            assert result.value.overclaims == ["unsupported"]
            assert (result.input_tokens, result.output_tokens) == (20, 10)

    asyncio.run(exercise())
    assert len(requests) == 2
    assert all(body["max_completion_tokens"] == 8000 for body in requests)
    assert all("reasoning_effort" not in body for body in requests)
    repair = requests[1]["messages"]
    assert len(repair) == 2
    assert "research context" not in str(repair)
    assert "Adversarially review" not in str(repair)
    assert '"overclaims": "unsupported"' in repair[1]["content"]
    assert "list_type" in repair[1]["content"]
    assert "do not redo the review" in repair[1]["content"]


def test_review_failed_after_invalid_repair_remains_due(tmp_path):
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"overclaims": "still invalid"}'}}]},
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "key", client=client, max_retries=2)
            config = _config(tmp_path)
            with Database(config.db_path) as db:
                qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
                query_id = db.add_query(
                    SearchQuery(research_question_id=qid, query="hyperpop origin")
                )
                db.mark_query_executed(query_id)
                run_id = await Researcher(
                    db, llm, FakeRetrieval(), config, qid, Limits(max_actions=2)
                ).run()
                run = db.rows("SELECT status,stop_reason FROM runs WHERE id=?", (run_id,))[0]
                assert tuple(run) == ("stopped", "review_failed")
                assert db.review_due(qid) == query_id
                assert len(db.rows("SELECT * FROM review_attempts")) == 1
                assert len(db.rows("SELECT * FROM provider_attempts")) == 2

    asyncio.run(exercise())
    assert requests == 2


def test_provider_error_diagnostics_redact_credentials_and_response_text(caplog):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "code": "secret-key",
                            "message": "full prompt: private research context",
                        }
                    },
                )
            )
        ) as client:
            llm = OpenAICompatibleLLM("https://llm.example/v1", "secret-key", client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await llm.complete("model", "system", "private research context", AdversarialReview)

    caplog.set_level(logging.INFO, logger="hypertrace.llm.openai_compatible")
    asyncio.run(exercise())
    logged = "\n".join(record.message for record in caplog.records)
    assert '"http_status": 400' in logged
    assert '"provider_error":' in logged
    assert '"code": "<redacted>"' in logged
    assert "secret-key" not in logged
    assert "private research context" not in logged
