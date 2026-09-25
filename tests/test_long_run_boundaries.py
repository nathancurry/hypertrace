"""Long-run work selection, retry, and target provenance boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from hypertrace.config import Config
from hypertrace.db import ARCHIVE_MAX_ATTEMPTS, FETCH_MAX_ATTEMPTS, Database
from hypertrace.llm.base import LLMResult
from hypertrace.models import (
    AdversarialReview,
    ContentRegion,
    Evidence,
    ExtractedEvidence,
    Hypothesis,
    Interpretation,
    PageAssessment,
    PlannedQuery,
    QueryPlan,
    QuoteRegion,
    ResearchQuestion,
    ResearchRun,
    SearchQuery,
    Source,
)
from hypertrace.reports import latest_run_summary, markdown_report
from hypertrace.research import RESEARCH_CONTEXT_LIMIT, Limits, Researcher
from hypertrace.retrieval.base import FetchFailure, SearchResult
from hypertrace.retrieval.wayback import ArchiveGap
from hypertrace.retrieval.web import BraveWeb


def config(tmp_path, *, consecutive=2):
    return Config(
        db_path=tmp_path / "case.db",
        llm_base_url="https://research.example/v1",
        llm_api_key="key",
        router_model="research",
        research_model="research",
        review_model="review",
        brave_api_key="key",
        input_cost_per_million=0,
        output_cost_per_million=0,
        min_yield=0,
        json_mode=True,
        max_consecutive_target_searches=consecutive,
    )


def source(url, content, **fields):
    return Source(
        canonical_url=url,
        retrieved_url=url,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_regions=[ContentRegion(start=0, end=len(content), region=QuoteRegion.ARTICLE_BODY)],
        **fields,
    )


def two_targets(db):
    qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
    db.install_source_targets(
        qid,
        [("bjork-transmission", "Direct link"), ("spotify-naming", "Naming account")],
        [],
    )
    for key, word in (("bjork-transmission", "Björk"), ("spotify-naming", "Spotify")):
        for number in range(4):
            db.add_query(
                SearchQuery(
                    research_question_id=qid,
                    query=f"{word} primary source variant {number}",
                    source_target=key,
                    information_value="high",
                )
            )
    return qid


class NoResults:
    def __init__(self):
        self.searches = []

    async def search(self, query, count=5):
        self.searches.append(query)
        return []


def test_zero_yield_target_rotates_and_pauses_auditably(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = two_targets(db)
        retrieval = NoResults()
        runner = Researcher(db, object(), retrieval, config(tmp_path), qid, Limits(max_actions=4))
        run_id = asyncio.run(runner.run())
        assert retrieval.searches[:2] == [
            "Björk primary source variant 0",
            "Björk primary source variant 1",
        ]
        assert retrieval.searches[2:] == [
            "Spotify primary source variant 0",
            "Spotify primary source variant 1",
        ]
        assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == "max_actions"
        progress = db.rows(
            "SELECT t.key,p.outcome,p.paused_at,p.pause_reason,p.pause_threshold "
            "FROM target_search_progress p "
            "JOIN source_targets t ON t.id=p.source_target_id "
            "WHERE p.run_id=? ORDER BY p.id",
            (run_id,),
        )
        assert [row["outcome"] for row in progress] == ["zero_yield"] * 4
        assert [row["key"] for row in progress] == [
            "bjork-transmission",
            "bjork-transmission",
            "spotify-naming",
            "spotify-naming",
        ]
        assert progress[1]["paused_at"] and progress[3]["paused_at"]
        assert progress[1]["pause_reason"] == "consecutive_zero_yield"
        assert progress[1]["pause_threshold"] == 2


def test_new_run_reconciles_unfinished_predecessors_without_losing_billing(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="first source"))
        first = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.record_action(first, "search", "query_id=1 results=1")
        attempt = db.start_provider_attempt(first, "assess", "test", 10)
        second = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        successor = asyncio.run(
            Researcher(
                db, object(), NoResults(), config(tmp_path), qid, Limits(max_actions=1)
            ).run()
        )
        rows = db.rows(
            "SELECT id,status,actions_taken,unknown_spend_requests,stop_reason FROM runs "
            "WHERE id IN (?,?) ORDER BY id",
            (first, second),
        )
        assert [
            (row["status"], row["actions_taken"], row["unknown_spend_requests"]) for row in rows
        ] == [
            ("interrupted", 1, 1),
            ("interrupted", 0, 0),
        ]
        assert all("actual stop time unknown" in row["stop_reason"] for row in rows)
        assert (
            db.rows("SELECT outcome FROM provider_attempts WHERE id=?", (attempt,))[0][0]
            == "in_flight"
        )
        audits = db.rows(
            "SELECT run_id,detail FROM actions WHERE action='run_state_recovery' ORDER BY run_id"
        )
        assert [row["run_id"] for row in audits] == [first, second]
        assert all(json.loads(row["detail"])["superseding_run_id"] == successor for row in audits)


def test_legacy_review_context_does_not_inflate_persisted_action_count(tmp_path):
    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.record_action(run_id, "review_context", '{"characters":100}')
        assert db.rows("SELECT actions_taken FROM runs WHERE id=?", (run_id,))[0][0] == 0
        db.record_action(run_id, "review", "completed")
        db.conn.execute("UPDATE runs SET actions_taken=2 WHERE id=?", (run_id,))
        assert db.rows("SELECT actions_taken FROM runs WHERE id=?", (run_id,))[0][0] == 2
    with Database(path) as db:
        assert db.rows("SELECT actions_taken FROM runs WHERE id=?", (run_id,))[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM actions WHERE run_id=?", (run_id,))[0][0] == 2


def test_active_concurrent_run_cannot_be_reconciled(tmp_path):
    class BlockingSearch:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def search(self, query, count=5):
            self.started.set()
            await self.release.wait()
            return []

    async def exercise():
        with Database(tmp_path / "case.db") as db:
            qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
            db.add_query(SearchQuery(research_question_id=qid, query="first source"))
            retrieval = BlockingSearch()
            first = asyncio.create_task(
                Researcher(
                    db, object(), retrieval, config(tmp_path), qid, Limits(max_actions=1)
                ).run()
            )
            await retrieval.started.wait()
            with pytest.raises(RuntimeError, match="already active"):
                await Researcher(
                    db, object(), NoResults(), config(tmp_path), qid, Limits(max_actions=1)
                ).run()
            assert db.rows("SELECT COUNT(*) FROM runs")[0][0] == 1
            assert db.rows("SELECT status FROM runs")[0][0] == "running"
            retrieval.release.set()
            await first

    asyncio.run(exercise())


def test_retryable_fetches_keep_history_and_back_off_across_runs(tmp_path):
    class FailingFetch:
        async def fetch(self, url):
            raise FetchFailure("ConnectError", url, [url], retryable=True)

    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="first source"))
        db.add_candidate(query_id, "https://example.org/failing", "Failing source")
    for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):
        with Database(path) as db:
            if attempt > 1:
                db.conn.execute(
                    "UPDATE candidates SET fetch_next_eligible_at='2000-01-01T00:00:00+00:00'"
                )
            candidate = db.pending_candidate(qid)
            assert candidate is not None
            runner = Researcher(db, object(), FailingFetch(), config(tmp_path), qid, Limits())
            runner.run_id = db.start_run(
                ResearchRun(research_question_id=qid, model="test", provider="test")
            )
            runner.deadline = time.monotonic() + 600
            assert not asyncio.run(runner._one_candidate(dict(candidate)))
            failed = db.rows("SELECT * FROM candidates")[0]
            assert failed["fetch_attempts"] == attempt
            assert failed["failure_at"] and failed["fetch_error"] == "ConnectError"
            assert failed["fetch_retryable"] == int(attempt < FETCH_MAX_ATTEMPTS)
            if attempt < FETCH_MAX_ATTEMPTS:
                assert failed["fetch_next_eligible_at"] and db.pending_candidate(qid) is None
                assert (
                    datetime.fromisoformat(failed["fetch_next_eligible_at"])
                    - datetime.fromisoformat(failed["failure_at"])
                ).total_seconds() == 60 * 2 ** (attempt - 1)
            else:
                assert failed["fetch_next_eligible_at"] is None
        with Database(path) as db:
            assert db.rows("SELECT fetch_attempts FROM candidates")[0][0] == attempt
    with Database(path) as db:
        assert db.pending_candidate(qid) is None
        assert (
            db.rows("SELECT COUNT(*) FROM actions WHERE action='fetch'")[0][0] == FETCH_MAX_ATTEMPTS
        )


def test_latest_run_summary_uses_persisted_outcomes(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.record_action(run_id, "search", "query_id=1 results=2")
        db.record_action(run_id, "fetch", "candidate_id=1 failed: HTTP 429")
        db.record_review_cadence(
            run_id,
            qid,
            "triggered",
            "action_threshold",
            {"meaningful_actions": 2, "evidence_ids": [], "target_keys": [], "hypothesis_ids": []},
        )
        db.finish_run(run_id, "stopped", "max_actions")
        summary = latest_run_summary(db, qid)
        assert summary is not None
        assert "2 actions; 1 productive research actions" in summary
        assert "retrieval throttle 1" in summary
        assert "triggers: action_threshold 1" in summary
        assert summary in markdown_report(db, qid)


def test_productive_target_regains_priority_after_rotation(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = two_targets(db)
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        high = db.rows("SELECT id FROM source_targets WHERE key='bjork-transmission'")[0][0]
        for number in range(2):
            query = db.next_search_query(qid, run_id, 2)
            assert query["source_target_id"] == high
            db.persist_search_results(qid, query["id"], [], run_id)
            quote = f"Direct source evidence {number}."
            sid = db.add_source(source(f"https://example.org/{number}", quote))
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    discovered_by_query_id=query["id"],
                )
            )
            db.finalize_target_searches(run_id, 2)
        lower = db.next_search_query(qid, run_id, 2)
        assert lower["source_target_id"] != high
        db.persist_search_results(qid, lower["id"], [], run_id)
        db.finalize_target_searches(run_id, 2)
        assert db.next_search_query(qid, run_id, 2)["source_target_id"] == high
        assert [
            row[0]
            for row in db.rows(
                "SELECT outcome FROM target_search_progress WHERE source_target_id=? ORDER BY id",
                (high,),
            )
        ] == ["productive", "productive"]


def test_all_paused_targets_resume_best_work_without_clearing_history(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = two_targets(db)
        retrieval = NoResults()
        runner = Researcher(db, object(), retrieval, config(tmp_path), qid, Limits(max_actions=5))
        run_id = asyncio.run(runner.run())
        assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == "max_actions"
        assert retrieval.searches[-1] == "Björk primary source variant 2"
        assert db.rows("SELECT COUNT(*) FROM queries WHERE status='active'")[0][0] > 0
        pauses = db.rows(
            "SELECT p.paused_at,p.pause_reason FROM target_search_progress p "
            "WHERE p.run_id=? AND p.paused_at IS NOT NULL ORDER BY p.id",
            (run_id,),
        )
        assert len(pauses) >= 2
        assert all(row["pause_reason"] == "consecutive_zero_yield" for row in pauses)


def test_unpaused_target_beats_higher_priority_paused_target(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = two_targets(db)
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        for _ in range(2):
            query = db.next_search_query(qid, run_id, 2)
            assert "Björk" in query["query"]
            db.persist_search_results(qid, query["id"], [], run_id)
            db.finalize_target_searches(run_id, 2)
        assert "Spotify" in db.next_search_query(qid, run_id, 2)["query"]


def test_retry_blocked_target_executes_deferred_work_then_restores_retries(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        ids = [
            db.add_query(
                SearchQuery(
                    research_question_id=qid,
                    query=f"Björk original interview variant {number}",
                    source_target="bjork-transmission",
                    information_value="high",
                )
            )
            for number in range(4)
        ]
        assert db.rows("SELECT status FROM queries WHERE id=?", (ids[3],))[0][0] == "deferred"
        for query_id in ids[:3]:
            db.record_search_failure(qid, query_id, "HTTP 429", 8000)
        retrieval = NoResults()
        runner = Researcher(db, object(), retrieval, config(tmp_path), qid, Limits(max_actions=3))
        run_id = asyncio.run(runner.run())
        assert retrieval.searches == ["Björk original interview variant 3"]
        assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == (
            "retry_beyond_deadline"
        )
        assert [
            tuple(row)
            for row in db.rows(
                "SELECT search_attempts,search_error FROM queries WHERE id IN (?,?,?) ORDER BY id",
                tuple(ids[:3]),
            )
        ] == [(1, "HTTP 429")] * 3
        assert all(
            row[0]
            for row in db.rows(
                "SELECT search_next_eligible_at FROM queries WHERE id IN (?,?,?)",
                tuple(ids[:3]),
            )
        )
        db.conn.execute(
            "UPDATE queries SET search_next_eligible_at=? WHERE id IN (?,?,?)",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), *ids[:3]),
        )
        assert db.next_search_query(qid, run_id, 2)["id"] in ids[:3]
        assert db.rows("SELECT COUNT(*) FROM queries WHERE status='active'")[0][0] <= 3


def test_eligible_retry_reenters_bounded_target_frontier(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        ids = [
            db.add_query(
                SearchQuery(
                    research_question_id=qid,
                    query=f"Björk independent interview {number}",
                    source_target="bjork-transmission",
                    information_value="high",
                )
            )
            for number in range(6)
        ]
        for query_id in ids[:3]:
            db.record_search_failure(qid, query_id, "HTTP 503", 8000)
        assert {row[0] for row in db.rows("SELECT id FROM queries WHERE status='active'")} == set(
            ids[3:]
        )
        db.conn.execute(
            "UPDATE queries SET search_next_eligible_at=? WHERE id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), ids[0]),
        )
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        assert db.next_search_query(qid, run_id, 3)["id"] == ids[0]
        assert {row[0] for row in db.rows("SELECT id FROM queries WHERE status='active'")} == {
            ids[0],
            ids[3],
            ids[4],
        }
        assert db.rows("SELECT search_attempts,search_error FROM queries WHERE id=?", (ids[0],))[0][
            :
        ] == (1, "HTTP 503")


class ContextLLM:
    async def complete(self, model, system, user, schema):
        if schema is QueryPlan:
            return LLMResult(value=QueryPlan(queries=[]))
        if schema is Interpretation:
            return LLMResult(value=Interpretation())
        raise AssertionError(schema)


def test_planning_and_interpretation_context_bounds_preserve_material_records(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        hid = db.add_hypothesis(Hypothesis(research_question_id=qid, statement="A linked B."))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        target_query = db.add_query(
            SearchQuery(
                research_question_id=qid,
                query="Björk exact source",
                source_target="bjork-transmission",
            )
        )
        other_query = db.add_query(SearchQuery(research_question_id=qid, query="unrelated archive"))
        ids = []
        for number in range(90):
            quote = f"Historical excerpt {number}: " + "evidence text " * 20
            sid = db.add_source(source(f"https://example.org/{number}", quote))
            evidence_id = db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage" if number == 2 else "interpretive_context",
                    discovered_by_query_id=target_query if number == 1 else other_query,
                )
            )
            ids.append(evidence_id)
        db.add_relationship(ids[0], hid, "contradicts", "Contrary primary context")
        db.record_archive_lookup(
            qid, target_query, "https://example.org/missing-original", [], "No Wayback snapshots"
        )
        runner = Researcher(db, ContextLLM(), object(), config(tmp_path), qid, Limits())
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        context, counts = runner._context(target_query)
        assert (
            len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
            <= RESEARCH_CONTEXT_LIMIT
        )
        assert {ids[0], ids[1], ids[2], ids[-1]} <= {item["id"] for item in context["evidence"]}
        assert counts["evidence"]["omitted"] > 0
        assert any("missing-original" in gap["original_url"] for gap in context["gaps"])
        asyncio.run(runner._plan())
        asyncio.run(runner._interpret(target_query))
        diagnostics = db.rows(
            "SELECT logical_action,characters,counts_json FROM context_diagnostics ORDER BY id"
        )
        assert [row["logical_action"] for row in diagnostics] == ["plan", "interpret"]
        assert all(row["characters"] <= RESEARCH_CONTEXT_LIMIT for row in diagnostics)
        assert all(json.loads(row["counts_json"])["evidence"]["omitted"] > 0 for row in diagnostics)


class TimeoutArchive:
    async def lookup(self, *args):
        raise ArchiveGap("CDX lookup failed: ReadTimeout", retryable=True)


def test_transient_archive_retry_and_dating_only_lookup(tmp_path):
    original = "https://example.org/old-article"
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="When was the article published?"))
        db.install_source_targets(qid, [("pc-music-2014", "Original dated text")], [])
        query_id = db.add_query(
            SearchQuery(
                research_question_id=qid,
                query="find original article",
                source_target="pc-music-2014",
            )
        )
        db.add_archive_target(qid, "pc-music-2014", original, start_year=2014, end_year=2014)
        db.conn.execute(
            "UPDATE source_targets SET status='resolved',evidence_status='found' "
            "WHERE key='pc-music-2014'"
        )
        target = db.next_archive_lookup(qid)
        assert target["original_url"] == original
        runner = Researcher(
            db, object(), object(), config(tmp_path), qid, Limits(), archive=TimeoutArchive()
        )
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        asyncio.run(runner._archive_lookup(target))
        first = db.rows(
            "SELECT status,retryable,attempt_count,next_eligible_at FROM archive_lookups"
        )[0]
        assert (first["status"], first["retryable"], first["attempt_count"]) == ("retryable", 1, 1)
        assert first["next_eligible_at"] and db.next_archive_lookup(qid) is None
        for attempt in (2, 3):
            db.conn.execute("UPDATE archive_lookups SET next_eligible_at=?", ("2000-01-01",))
            db.record_archive_lookup(
                qid, query_id, original, [], "CDX lookup failed: ReadTimeout", retryable=True
            )
            row = db.rows("SELECT status,attempt_count FROM archive_lookups")[0]
            assert row["attempt_count"] == attempt
        assert row["status"] == "gap" and db.next_archive_lookup(qid) is None
        assert ARCHIVE_MAX_ATTEMPTS == 3


def test_legacy_cdx_timeout_migrates_to_retryable(tmp_path):
    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="When was the article published?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="find archive"))
        db.record_archive_lookup(
            qid, query_id, "https://example.org/old", [], "CDX lookup failed: ReadTimeout"
        )
    with Database(path) as db:
        row = db.rows(
            "SELECT status,retryable,attempt_count,next_eligible_at FROM archive_lookups"
        )[0]
        assert (row["status"], row["retryable"], row["attempt_count"]) == ("retryable", 1, 1)
        assert row["next_eligible_at"]


async def run_search_case(tmp_path, first_response):
    calls = []

    def handler(request):
        query = request.url.params["q"]
        calls.append(query)
        if query == "first source":
            if isinstance(first_response, Exception):
                raise first_response
            return first_response
        return httpx.Response(200, json={"web": {"results": []}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with Database(tmp_path / "case.db") as db:
            qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
            first = db.add_query(SearchQuery(research_question_id=qid, query="first source"))
            db.add_query(SearchQuery(research_question_id=qid, query="second source"))
            runner = Researcher(
                db, object(), BraveWeb("key", client), config(tmp_path), qid, Limits(max_actions=2)
            )
            run_id = await runner.run()
            row = db.rows(
                "SELECT search_error,search_attempts,search_next_eligible_at,status FROM queries WHERE id=?",
                (first,),
            )[0]
            assert row["search_attempts"] == 1 and row["status"] == "active"
            assert row["search_next_eligible_at"]
            assert calls == ["first source", "second source"]
            assert (
                db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == "max_actions"
            )
            return row["search_next_eligible_at"]


def test_brave_429_honors_retry_after_and_continues(tmp_path):
    next_time = asyncio.run(
        run_search_case(
            tmp_path, httpx.Response(429, headers={"Retry-After": "120"}, json={"error": "limited"})
        )
    )
    assert 100 <= (datetime.fromisoformat(next_time) - datetime.now(UTC)).total_seconds() <= 121


def test_brave_5xx_and_timeout_continue_other_searches(tmp_path):
    for number, response in enumerate(
        [httpx.Response(503), httpx.ReadTimeout("slow"), httpx.ConnectError("offline")]
    ):
        case_dir = tmp_path / str(number)
        case_dir.mkdir()
        asyncio.run(run_search_case(case_dir, response))


def test_retry_beyond_deadline_stops_cleanly(tmp_path):
    async def case():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(503))
        ) as client:
            with Database(tmp_path / "case.db") as db:
                qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
                query_id = db.add_query(SearchQuery(research_question_id=qid, query="sole source"))
                runner = Researcher(
                    db,
                    object(),
                    BraveWeb("key", client),
                    config(tmp_path),
                    qid,
                    Limits(max_minutes=0.01),
                )
                run_id = await runner.run()
                assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == (
                    "retry_beyond_deadline"
                )
                assert (
                    db.rows("SELECT search_attempts FROM queries WHERE id=?", (query_id,))[0][0]
                    == 1
                )

    asyncio.run(case())


def test_search_retry_limit_is_a_persisted_gap(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="first source"))
        for number in range(3):
            assert db.record_search_failure(qid, query_id, "HTTP 503", 0) == (number < 2)
        row = db.rows(
            "SELECT status,search_attempts,search_error FROM queries WHERE id=?", (query_id,)
        )[0]
        assert tuple(row) == ("search_failed", 3, "HTTP 503")
        assert db.next_search_query(qid, 0, 2) is None


def test_sole_search_retry_waits_once_then_resumes(tmp_path, monkeypatch):
    class FlakySearch:
        def __init__(self):
            self.calls = 0

        async def search(self, query, count=5):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("temporary")
            return []

    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did this term originate?"))
        db.add_query(SearchQuery(research_question_id=qid, query="sole source"))
        sleeps = []

        async def advance(delay):
            sleeps.append(delay)
            db.conn.execute(
                "UPDATE queries SET search_next_eligible_at='2000-01-01T00:00:00+00:00'"
            )

        monkeypatch.setattr("hypertrace.research.asyncio.sleep", advance)
        retrieval = FlakySearch()
        run_id = asyncio.run(
            Researcher(db, object(), retrieval, config(tmp_path), qid, Limits(max_actions=2)).run()
        )
        assert retrieval.calls == 2
        assert len(sleeps) == 1 and sleeps[0] > 0
        assert tuple(
            db.rows("SELECT stop_reason,actions_taken FROM runs WHERE id=?", (run_id,))[0]
        ) == ("max_actions", 2)


def test_sole_archive_retry_waits_once_then_resumes(tmp_path, monkeypatch):
    class FlakyArchive:
        def __init__(self):
            self.calls = 0

        async def lookup(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise ArchiveGap("CDX lookup failed: ReadTimeout", retryable=True)
            return []

    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="When was this article published?"))
        db.install_source_targets(qid, [("pc-music-2014", "Original text")], [])
        query_id = db.add_query(
            SearchQuery(
                research_question_id=qid, query="original article", source_target="pc-music-2014"
            )
        )
        db.mark_query_executed(query_id)
        db.mark_query_reviewed(query_id)
        db.add_archive_target(qid, "pc-music-2014", "https://example.org/old")
        sleeps = []

        async def advance(delay):
            sleeps.append(delay)
            db.conn.execute(
                "UPDATE archive_lookups SET next_eligible_at='2000-01-01T00:00:00+00:00'"
            )

        monkeypatch.setattr("hypertrace.research.asyncio.sleep", advance)
        archive = FlakyArchive()
        run_id = asyncio.run(
            Researcher(
                db,
                object(),
                object(),
                config(tmp_path),
                qid,
                Limits(max_actions=2),
                archive=archive,
            ).run()
        )
        assert archive.calls == 2
        assert len(sleeps) == 1 and sleeps[0] > 0
        assert tuple(
            db.rows("SELECT stop_reason,actions_taken FROM runs WHERE id=?", (run_id,))[0]
        ) == ("max_actions", 2)


def test_mixed_planner_proposals_commit_valid_and_audit_rejections(tmp_path):
    class Planner:
        async def complete(self, model, system, user, schema):
            assert schema is QueryPlan

            def proposed(text, target, gap="Missing original source"):
                return PlannedQuery(
                    query=text,
                    rationale="Direct document",
                    gap=gap,
                    information_value="high",
                    novelty="Distinct archive search",
                    admission_basis="primary_source",
                    source_target=target,
                )

            return LLMResult(
                value=QueryPlan(
                    queries=[
                        proposed("unknown target query", "not-a-target"),
                        proposed("resolved target query", "pc-music-2014"),
                        proposed("blank gap query", "bjork-transmission", "   "),
                        proposed("valid target query", "bjork-transmission"),
                    ]
                )
            )

    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did this term originate?"))
        db.install_source_targets(
            qid,
            [
                ("pc-music-2014", "Original text"),
                ("bjork-transmission", "Direct link"),
            ],
            [],
        )
        db.conn.execute("UPDATE source_targets SET status='resolved' WHERE key='pc-music-2014'")
        runner = Researcher(db, Planner(), object(), config(tmp_path), qid, Limits())
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        asyncio.run(runner._plan())
        assert [row[0] for row in db.rows("SELECT query FROM queries")] == ["valid target query"]
        rejected = db.rows(
            "SELECT run_id,query_text,generated_by,source_target_key,reason "
            "FROM plan_query_rejections ORDER BY id"
        )
        assert len(rejected) == 3
        assert {row["source_target_key"] for row in rejected} == {
            "not-a-target",
            "pc-music-2014",
            "bjork-transmission",
        }
        assert all(
            row["run_id"] == runner.run_id and row["generated_by"] == "research" for row in rejected
        )
        assert {row["reason"] for row in rejected} == {
            "Unknown source target",
            "Source discovery already resolved",
            "Proposed query requires an unresolved gap and novelty explanation",
        }
    with Database(path) as db:
        runner = Researcher(db, Planner(), object(), config(tmp_path), qid, Limits())
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        asyncio.run(runner._plan())
        assert db.rows("SELECT COUNT(*) FROM queries WHERE query='valid target query'")[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM plan_query_rejections")[0][0] == 6


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"query": "   "}, "Query text is empty"),
        ({"rationale": "   "}, "Query rationale is empty"),
        ({"information_value": "urgent"}, "Invalid information value"),
        ({"admission_basis": "speculation"}, "Invalid admission basis"),
    ],
)
def test_model_proposal_validation_rejects_input_before_insert(tmp_path, change, reason):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        valid = SearchQuery(
            research_question_id=qid,
            query="independent source archive",
            rationale="Find the original text",
            gap="Original source missing",
            novelty="Search another archive",
        )
        db.persist_plan(qid, [valid, valid.model_copy(update=change)], run_id)
        assert db.rows("SELECT COUNT(*) FROM queries")[0][0] == 1
        rejected = db.rows("SELECT run_id,query_text,rationale,reason FROM plan_query_rejections")[
            0
        ]
        assert (rejected["run_id"], rejected["reason"]) == (run_id, reason)
        assert rejected["query_text"] == change.get("query", valid.query)
        assert rejected["rationale"] == change.get("rationale", valid.rationale)


def test_candidate_proposals_reject_whitespace_and_preserve_parent_commit(tmp_path):
    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="initial source"))
        db.mark_query_executed(query_id)
        db.add_candidate(query_id, "https://example.org/article", "Article")
        candidate_id = db.pending_candidate(qid)["id"]
        quote = "The original article uses hyperpop."
        source_id = db.add_source(source("https://example.org/article", quote))
        db.set_candidate_source(candidate_id, source_id)
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        evidence = Evidence(
            source_id=source_id,
            research_question_id=qid,
            exact_quote=quote,
            normalized_claim=quote,
            evidence_type="observed_usage",
        )
        valid = SearchQuery(
            research_question_id=qid,
            query="another original article",
            rationale="Find an independent text",
            gap="Original source missing",
            novelty="Search another archive",
        )
        invalid = valid.model_copy(update={"query": "whitespace gap article", "gap": "   "})
        other_source_id = db.add_source(source("https://example.org/other", "Other source text."))
        with pytest.raises(ValueError, match="Evidence source does not match candidate"):
            db.persist_candidate_assessment(
                candidate_id,
                [(evidence.model_copy(update={"source_id": other_source_id}), [])],
                [],
                [valid, invalid],
                run_id=run_id,
            )
        assert not db.rows("SELECT id FROM candidate_query_rejections")
        assert not db.rows("SELECT id FROM queries WHERE query=?", (valid.query,))
        assert (
            db.persist_candidate_assessment(
                candidate_id, [(evidence, [])], [], [valid, invalid], run_id=run_id
            )
            == 1
        )
        rejected = db.rows("SELECT * FROM candidate_query_rejections")[0]
        assert (
            rejected["run_id"],
            rejected["candidate_id"],
            rejected["query_text"],
            rejected["rationale"],
            rejected["source_target_key"],
            rejected["target_purpose"],
            rejected["reason"],
        ) == (
            run_id,
            candidate_id,
            invalid.query,
            invalid.rationale,
            "",
            "source",
            "Proposed query requires an unresolved gap and novelty explanation",
        )
    with Database(path) as db:
        assert db.pending_candidate(qid) is None
        assert db.rows("SELECT COUNT(*) FROM evidence")[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM queries WHERE query=?", (valid.query,))[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM candidate_query_rejections")[0][0] == 1


def test_review_proposals_reject_whitespace_and_commit_valid_work(tmp_path):
    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        reviewed_id = db.add_query(SearchQuery(research_question_id=qid, query="initial source"))
        db.mark_query_executed(reviewed_id)
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        valid = SearchQuery(
            research_question_id=qid,
            query="independent archive search",
            rationale="Find an independent text",
            gap="Original source missing",
            novelty="Search another archive",
        )
        invalid = valid.model_copy(update={"query": "whitespace novelty search", "novelty": "   "})
        db.persist_review(run_id, qid, reviewed_id, "{}", [valid, invalid])
        rejected = db.rows("SELECT * FROM review_query_rejections")[0]
        assert (
            rejected["run_id"],
            rejected["query_text"],
            rejected["rationale"],
            rejected["source_target_key"],
            rejected["target_purpose"],
            rejected["reason"],
        ) == (
            run_id,
            invalid.query,
            invalid.rationale,
            "",
            "source",
            "Proposed query requires an unresolved gap and novelty explanation",
        )
    with Database(path) as db:
        assert db.review_due(qid) is None
        assert db.rows("SELECT COUNT(*) FROM reviews")[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM queries WHERE query=?", (valid.query,))[0][0] == 1
        assert db.rows("SELECT COUNT(*) FROM review_query_rejections")[0][0] == 1


class FlakyFetch:
    def __init__(self, fail_count=1):
        self.calls = 0
        self.fail_count = fail_count

    async def search(self, query, count=5):
        return [SearchResult(url="https://example.org/article", title="Original")]

    async def fetch(self, url):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise FetchFailure("HTTP 503", url, [url], status_code=503, retryable=True)
        return source(url, "The term hyperpop appears in this original article.")


class AssessmentLLM:
    async def complete(self, model, system, user, schema):
        if schema is PageAssessment:
            return LLMResult(
                value=PageAssessment(
                    relevant=True,
                    reason="Direct use",
                    evidence=[
                        ExtractedEvidence(
                            exact_quote="The term hyperpop appears in this original article.",
                            normalized_claim="The term hyperpop appears in this original article.",
                            evidence_type="observed_usage",
                        )
                    ],
                )
            )
        if schema is Interpretation:
            return LLMResult(value=Interpretation())
        raise AssertionError(schema)


def test_transient_fetch_retries_in_run_and_avoids_false_pause(tmp_path, monkeypatch):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        db.add_query(
            SearchQuery(
                research_question_id=qid, query="direct source", source_target="bjork-transmission"
            )
        )
        sleeps = []

        async def advance(delay):
            sleeps.append(delay)
            db.conn.execute(
                "UPDATE candidates SET fetch_next_eligible_at='2000-01-01T00:00:00+00:00'"
            )

        monkeypatch.setattr("hypertrace.research.asyncio.sleep", advance)
        retrieval = FlakyFetch()
        run_id = asyncio.run(
            Researcher(
                db,
                AssessmentLLM(),
                retrieval,
                config(tmp_path),
                qid,
                Limits(max_actions=5),
            ).run()
        )
        assert db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0] == "max_actions"
        assert retrieval.calls == 2 and len(sleeps) == 1
        assert db.rows("SELECT outcome FROM target_search_progress")[0][0] == "productive"
        assert db.rows("SELECT COUNT(*) FROM fetch_attempt_events")[0][0] == 1
        assert db.rows("SELECT fetch_attempts,assessed_at FROM candidates")[0]["assessed_at"]


def test_fetch_retry_exhaustion_yields_zero_and_preserves_attempts(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        query_id = db.add_query(
            SearchQuery(
                research_question_id=qid, query="direct source", source_target="bjork-transmission"
            )
        )
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.persist_search_results(
            qid, query_id, [("https://example.org/article", "Original")], run_id
        )
        candidate_id = db.rows("SELECT id FROM candidates")[0][0]
        for attempt in range(3):
            db.record_fetch_failure(candidate_id, "HTTP 503", retryable=True, status_code=503)
            db.finalize_target_searches(run_id, 2)
            outcome = db.rows("SELECT outcome FROM target_search_progress")[0][0]
            assert outcome == ("pending" if attempt < 2 else "zero_yield")
        assert [
            row[0] for row in db.rows("SELECT attempt_number FROM fetch_attempt_events ORDER BY id")
        ] == [1, 2, 3]
        assert db.rows("SELECT fetch_retryable FROM candidates")[0][0] == 0


def test_fetch_retry_beyond_run_deadline_remains_pending(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did the term originate?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        db.add_query(
            SearchQuery(
                research_question_id=qid, query="direct source", source_target="bjork-transmission"
            )
        )
        run_id = asyncio.run(
            Researcher(
                db,
                object(),
                FlakyFetch(fail_count=3),
                config(tmp_path),
                qid,
                Limits(max_minutes=0.01),
            ).run()
        )
        assert (
            db.rows("SELECT stop_reason FROM runs WHERE id=?", (run_id,))[0][0]
            == "retry_beyond_deadline"
        )
        candidate = db.rows("SELECT fetch_retryable,fetch_next_eligible_at FROM candidates")[0]
        assert candidate["fetch_retryable"] == 1 and candidate["fetch_next_eligible_at"]
        assert db.rows("SELECT outcome FROM target_search_progress")[0][0] == "pending"


def test_material_review_batches_survive_resume_without_duplicate_query_effects(tmp_path):
    class Reviewer:
        async def complete(self, model, system, user, schema):
            assert schema is AdversarialReview
            return LLMResult(
                value=AdversarialReview(
                    next_queries=[
                        PlannedQuery(
                            query="one follow-up archive search",
                            rationale="Original record",
                            gap="Missing original record",
                            information_value="high",
                            novelty="New archive",
                            admission_basis="primary_source",
                        )
                    ]
                )
            )

    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did this term originate?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="initial source"))
        db.mark_query_executed(query_id)
        quotes = [f"Direct hyperpop usage {number}." for number in range(25)]
        sid = db.add_source(source("https://example.org/many", "\n".join(quotes)))
        for quote in quotes:
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    primary_source_verified=True,
                    discovered_by_query_id=query_id,
                )
            )
        runner = Researcher(db, Reviewer(), object(), config(tmp_path), qid, Limits())
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        assert asyncio.run(runner._maybe_review())
        first = json.loads(db.rows("SELECT included_evidence_ids_json FROM reviews")[0][0])
        assert len(first) == 20
        assert len(db.review_cadence_state(qid)["evidence_ids"]) == 5
        assert db.rows("SELECT COUNT(*) FROM evidence WHERE review_covered_at IS NULL")[0][0] == 5
    with Database(path) as db:
        runner = Researcher(db, Reviewer(), object(), config(tmp_path), qid, Limits())
        runner.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        runner.deadline = time.monotonic() + 600
        assert asyncio.run(runner._maybe_review())
        second = json.loads(
            db.rows("SELECT included_evidence_ids_json FROM reviews ORDER BY id DESC LIMIT 1")[0][0]
        )
        assert len(set(second) - set(first)) == 5
        assert db.review_cadence_state(qid)["evidence_ids"] == []
        assert not asyncio.run(runner._maybe_review())
        assert db.rows("SELECT COUNT(*) FROM reviews")[0][0] == 2
        assert (
            db.rows("SELECT COUNT(*) FROM queries WHERE query='one follow-up archive search'")[0][0]
            == 1
        )


def test_repeated_quote_uses_first_verified_body_occurrence(tmp_path):
    quote = "The term hyperpop appears here."
    content = f"{quote}\n{quote}\n{quote}"
    heading_end = len(quote)
    first_body = heading_end + 1
    second_body = first_body + len(quote) + 1
    repeated = Source(
        canonical_url="https://example.org/repeated",
        retrieved_url="https://example.org/repeated",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_regions=[
            ContentRegion(start=0, end=heading_end, region=QuoteRegion.HEADING),
            ContentRegion(
                start=first_body, end=first_body + len(quote), region=QuoteRegion.ARTICLE_BODY
            ),
            ContentRegion(
                start=second_body, end=second_body + len(quote), region=QuoteRegion.ARTICLE_BODY
            ),
        ],
    )
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where was hyperpop used?"))
        sid = db.add_source(repeated)
        assert db.quote_region(sid, quote) == QuoteRegion.ARTICLE_BODY
        evidence_id = db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
            )
        )
        accepted = db.rows(
            "SELECT quote_start,quote_region,quote_context FROM evidence WHERE id=?", (evidence_id,)
        )[0]
        assert accepted["quote_start"] == first_body
        assert accepted["quote_region"] == "article_body"
        assert quote in accepted["quote_context"]
        furniture = Source(
            canonical_url="https://example.org/furniture",
            retrieved_url="https://example.org/furniture",
            content=quote,
            content_hash=hashlib.sha256(quote.encode()).hexdigest(),
            content_regions=[ContentRegion(start=0, end=len(quote), region=QuoteRegion.HEADING)],
        )
        furniture_id = db.add_source(furniture)
        with pytest.raises(ValueError, match="article-body"):
            db.add_evidence(
                Evidence(
                    source_id=furniture_id,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                )
            )


def test_executed_query_association_cannot_close_unrelated_target(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Was the name borrowed?"))
        db.install_source_targets(qid, [("bjork-transmission", "Direct link")], [])
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="shared exact query"))
        db.mark_query_executed(query_id)
        quote = "Unrelated primary source statement."
        sid = db.add_source(source("https://example.org/unrelated", quote))
        db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                primary_source_verified=True,
                discovered_by_query_id=query_id,
            )
        )
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.persist_review(
            run_id,
            qid,
            query_id,
            "{}",
            [
                SearchQuery(
                    research_question_id=qid,
                    query="shared exact query",
                    rationale="Check whether this search serves the target",
                    source_target="bjork-transmission",
                    gap="Find an explicit transmission source",
                    novelty="Later target proposal",
                )
            ],
        )
        db._refresh_source_targets_tx()
        query = db.rows("SELECT source_target_id,executed_at FROM queries WHERE id=?", (query_id,))[
            0
        ]
        assert query["source_target_id"] is None and query["executed_at"]
        assert db.rows("SELECT COUNT(*) FROM queries WHERE query='shared exact query'")[0][0] == 1
        assert (
            db.rows("SELECT COUNT(*) FROM query_target_associations WHERE query_id=?", (query_id,))[
                0
            ][0]
            == 1
        )
        assert (
            db.rows("SELECT status FROM source_targets WHERE key='bjork-transmission'")[0][0]
            == "unresolved"
        )
        assert "proposal only; execution provenance unchanged" in markdown_report(db, qid)


def test_later_copy_cannot_resolve_target_dating_but_artifact_can(tmp_path):
    original = "https://example.org/2014-original"
    quote = "The original article used hyper-pop."
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="When was this text published?"))
        db.install_source_targets(qid, [("pc-music-2014", "Original 2014 text")], [])
        db.add_archive_target(qid, "pc-music-2014", original, start_year=2014, end_year=2014)
        target_id = db.rows("SELECT id FROM source_targets WHERE key='pc-music-2014'")[0][0]
        sid = db.add_source(source(original, quote))
        evidence_id = db.add_evidence(
            Evidence(
                source_id=sid,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                primary_source_verified=True,
            )
        )
        assert db.rows("SELECT status FROM source_targets WHERE id=?", (target_id,))[0][0] == (
            "unresolved"
        )
        db.verify_existing_target_evidence(evidence_id, target_id, "Original target article text")
        assert db.rows("SELECT status,dating_status FROM source_targets WHERE id=?", (target_id,))[
            0
        ][:] == ("resolved", "unresolved")
        later = db.add_source(source("https://example.org/2020-copy", quote))
        db.add_evidence(
            Evidence(
                source_id=later,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                primary_source_verified=True,
                quote_verified_date="2020-01-01",
                date_verification_note="Dated 2020 copy.",
            )
        )
        assert (
            db.rows("SELECT dating_status FROM source_targets WHERE id=?", (target_id,))[0][0]
            == "unresolved"
        )
        try:
            db.add_evidence(
                Evidence(
                    source_id=later,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    primary_source_verified=True,
                    quote_verified_date="2020-01-01",
                    date_verification_note="Dated 2020 copy.",
                    verified_target_id=target_id,
                    target_verification_note="Claimed target copy",
                    target_artifact_date_verified=True,
                )
            )
        except ValueError as exc:
            assert "known artifact" in str(exc)
        else:
            raise AssertionError("Later copy was accepted as original artifact")
        try:
            db.add_evidence(
                Evidence(
                    source_id=sid,
                    research_question_id=qid,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    primary_source_verified=True,
                    quote_verified_date="2020-01-01",
                    date_verification_note="Later metadata date.",
                    verified_target_id=target_id,
                    target_verification_note="Original target article text",
                    target_artifact_date_verified=True,
                )
            )
        except ValueError as exc:
            assert "known artifact/date" in str(exc)
        else:
            raise AssertionError("Later date was accepted for the historical artifact")
        captured = db.add_source(
            source(
                "https://web.archive.org/web/20150101000000id_/https://example.org/2014-original",
                quote,
                original_url=original,
                archive_timestamp="20150101000000",
            )
        )
        db.add_evidence(
            Evidence(
                source_id=captured,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                primary_source_verified=True,
                quote_verified_date="2014-09-17",
                date_verification_note="Verified original issue scan.",
                verified_target_id=target_id,
                target_verification_note="Verified original issue scan",
                target_artifact_date_verified=True,
            )
        )
        assert (
            db.rows("SELECT dating_status FROM source_targets WHERE id=?", (target_id,))[0][0]
            == "resolved"
        )
