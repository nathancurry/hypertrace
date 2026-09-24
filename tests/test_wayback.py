from __future__ import annotations

import asyncio
import hashlib
import time

import httpx
import pytest

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.base import LLMResult
from hypertrace.models import (
    ContentRegion,
    Evidence,
    PageAssessment,
    QuoteRegion,
    ResearchQuestion,
    ResearchRun,
    SearchQuery,
    Source,
)
from hypertrace.planner import seed_source_targets
from hypertrace.reports import markdown_report
from hypertrace.research import Limits, Researcher
from hypertrace.retrieval.wayback import ArchiveGap, Snapshot, Wayback
from hypertrace.retrieval.web import BraveWeb

ORIGINAL = "https://pitchfork.com/thepitch/485-pc-musics-twisted-electronic-pop-a-users-manual/"
HTML = """<html><head><title>PC Music</title>
<meta property="article:published_time" content="2014-08-01"></head>
<body><article><p>PC Music was called hyper-pop in this text.</p></article></body></html>"""


def cdx_rows():
    return [
        ["timestamp", "original", "mimetype", "statuscode", "digest"],
        ["20130101000000", ORIGINAL, "text/html", "200", "old"],
        ["20140201000000", ORIGINAL, "text/html", "200", "feb"],
        ["20140802000000", ORIGINAL, "text/html", "200", "near"],
        ["20140803000000", ORIGINAL, "text/html", "200", "near"],
        ["20140801000000", ORIGINAL, "text/html", "302", "redirect"],
        ["20140804000000", ORIGINAL, "application/pdf", "200", "pdf"],
        ["20191201000000", ORIGINAL, "text/html", "200", "late"],
    ]


def test_known_url_selects_closest_useful_snapshot_and_retains_provenance(tmp_path):
    seen = []

    def respond(request):
        seen.append(str(request.url))
        if request.url.path == "/cdx/search/cdx":
            return httpx.Response(200, json=cdx_rows())
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    async def retrieve():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            web = BraveWeb("key", client)
            archive = Wayback(web)
            snapshots = await archive.lookup(
                ORIGINAL, start_date="2014-08-02", end_date="2014-08-02"
            )
            year_snapshots = await archive.lookup(ORIGINAL, 2014, 2014)
            source = await archive.fetch(ORIGINAL, snapshots[0])
            return snapshots, year_snapshots, source

    snapshots, year_snapshots, source = asyncio.run(retrieve())
    assert snapshots[0].timestamp == "20140802000000"
    assert year_snapshots[0].timestamp == "20140201000000"
    assert len(snapshots) == 3
    assert len({item.digest for item in snapshots}) == 3
    assert any("/cdx/search/cdx?" in url for url in seen)
    assert source.original_url == ORIGINAL
    assert source.resolved_original_url == ORIGINAL
    assert source.archive_url == source.retrieved_url == snapshots[0].url
    assert source.archive_timestamp == "20140802000000"
    assert source.http_status == 200
    assert source.page_publication_date == "2014-08-01"

    with Database(tmp_path / "archive.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where was hyper-pop used?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="PC Music 2014"))
        db.record_archive_lookup(qid, query_id, ORIGINAL, snapshots, "3 useful snapshots")
        candidate = db.pending_candidate(qid)
        source_id = db.attach_source(candidate["id"], source)
        quote = "PC Music was called hyper-pop in this text."
        db.add_evidence(
            Evidence(
                source_id=source_id,
                research_question_id=qid,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                discovered_by_query_id=query_id,
            )
        )
        report = markdown_report(db, qid)
        assert snapshots[0].url in report
        assert ORIGINAL in report
        assert "No independently dated quote" in report
        assert db.rows("SELECT quote_verified_date FROM evidence")[0][0] is None
        assert tuple(
            db.rows("SELECT original_url,archive_timestamp,http_status FROM sources")[0]
        ) == (ORIGINAL, "20140802000000", 200)


def test_no_snapshot_is_an_explicit_unresolved_gap(tmp_path):
    async def lookup():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text=""))
        ) as client:
            return await Wayback(BraveWeb("key", client)).lookup(ORIGINAL, 2014)

    try:
        asyncio.run(lookup())
    except ArchiveGap as exc:
        detail = str(exc)
    assert detail == "No Wayback snapshots"
    with Database(tmp_path / "gaps.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where was hyper-pop used?"))
        seed_source_targets(db, qid)
        target = db.next_archive_lookup(qid)
        assert target["original_url"] == ORIGINAL
        db.conn.execute("UPDATE queries SET status='exhausted' WHERE id=?", (target["query_id"],))
        db.record_archive_lookup(qid, target["query_id"], ORIGINAL, [], detail)
        assert "No Wayback snapshots" in markdown_report(db, qid)
        assert db.next_archive_lookup(qid) is None


@pytest.mark.parametrize(
    ("status", "mime", "gap"),
    [
        ("302", "text/html", "redirect-only capture"),
        ("403", "text/html", "blocked snapshot"),
        ("200", "application/pdf", "unsupported content"),
    ],
)
def test_unusable_archive_index_rows_remain_gaps(status, mime, gap):
    rows = [cdx_rows()[0], ["20140802000000", ORIGINAL, mime, status, "digest"]]

    async def lookup():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=rows))
        ) as client:
            return await Wayback(BraveWeb("key", client)).lookup(ORIGINAL, 2014)

    with pytest.raises(ArchiveGap, match=gap):
        asyncio.run(lookup())


def test_archived_candidate_uses_normal_assessment(tmp_path):
    quote = "PC Music was called hyper-pop in this text."

    def respond(request):
        if request.url.path == "/cdx/search/cdx":
            return httpx.Response(200, json=cdx_rows())
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    class Assess:
        async def complete(self, model, system, user, schema):
            assert schema is PageAssessment
            assert "Wayback capture" in user
            return LLMResult(
                value=PageAssessment.model_validate(
                    {
                        "relevant": True,
                        "reason": "Direct usage",
                        "evidence": [
                            {
                                "exact_quote": quote,
                                "normalized_claim": quote,
                                "evidence_type": "observed_usage",
                            }
                        ],
                    }
                )
            )

    async def assess(db, qid):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            web = BraveWeb("key", client)
            researcher = Researcher(
                db, Assess(), web, Config.from_env(), qid, Limits(max_actions=5)
            )
            researcher.deadline = time.monotonic() + 10
            researcher.run_id = db.start_run(
                ResearchRun(research_question_id=qid, model="test", provider="test")
            )
            await researcher._archive_lookup(db.next_archive_lookup(qid))
            await researcher._one_candidate(dict(db.pending_candidate(qid)))

    with Database(tmp_path / "assessment.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where was hyper-pop used?"))
        seed_source_targets(db, qid)
        asyncio.run(assess(db, qid))
        evidence = db.rows(
            "SELECT e.quote_verified_date,e.discovered_by_query_id,s.original_url,"
            "s.archive_timestamp,s.archive_url FROM evidence e JOIN sources s ON s.id=e.source_id"
        )[0]
        assert evidence["quote_verified_date"] is None
        assert evidence["discovered_by_query_id"] is not None
        assert evidence["original_url"] == ORIGINAL
        assert evidence["archive_timestamp"] == "20140201000000"
        assert evidence["archive_url"].startswith("https://web.archive.org/web/")


def test_archive_redirect_retains_resolved_original_and_capture():
    requested = f"https://web.archive.org/web/20140802000000id_/{ORIGINAL}"
    resolved_original = ORIGINAL.replace("pitchfork.com", "www.pitchfork.com")
    resolved = f"https://web.archive.org/web/20140803000000id_/{resolved_original}"

    def respond(request):
        if str(request.url) == requested:
            return httpx.Response(302, headers={"location": resolved})
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    async def fetch():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await Wayback(BraveWeb("key", client)).fetch(
                ORIGINAL, Snapshot("20140802000000", ORIGINAL, requested, "digest")
            )

    source = asyncio.run(fetch())
    assert source.original_url == ORIGINAL
    assert source.resolved_original_url == resolved_original
    assert source.archive_url == resolved
    assert source.archive_timestamp == "20140803000000"
    assert source.redirect_history == [requested]


def test_evidenced_pitchfork_host_variant_becomes_archive_target(tmp_path):
    with Database(tmp_path / "variants.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where was hyper-pop used?"))
        seed_source_targets(db, qid)
        query = db.rows(
            "SELECT q.id FROM queries q JOIN source_targets t ON t.id=q.source_target_id "
            "WHERE t.key='pc-music-2014' LIMIT 1"
        )[0][0]
        variant = ORIGINAL.replace("pitchfork.com", "www.pitchfork.com")
        db.add_candidate(query, variant, "PC Music article")
        seed_source_targets(db, qid)
        urls = {row[0] for row in db.rows("SELECT original_url FROM archive_targets")}
        assert urls == {ORIGINAL, variant}


def test_identical_snapshot_text_skips_second_assessment(tmp_path):
    content = "This article uses hyper-pop."
    source = Source(
        canonical_url="https://web.archive.org/web/20140802id_/https://example.org/a",
        retrieved_url="https://web.archive.org/web/20140802id_/https://example.org/a",
        original_url="https://example.org/a",
        archive_timestamp="20140802000000",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_regions=[ContentRegion(start=0, end=len(content), region=QuoteRegion.ARTICLE_BODY)],
    )

    class NoAssessment:
        async def complete(self, *args, **kwargs):
            raise AssertionError("Identical text was reassessed")

    with Database(tmp_path / "dedup.db") as db:
        qid = db.add_question(ResearchQuestion(question="Was hyper-pop used?"))
        query_id = db.add_query(SearchQuery(research_question_id=qid, query="hyper-pop 2014"))
        db.add_candidate(query_id, source.retrieved_url, "first")
        first = db.pending_candidate(qid)
        db.attach_source(first["id"], source)
        db.finish_candidate(first["id"])
        second_url = source.retrieved_url.replace("20140802", "20140803")
        db.add_candidate(query_id, second_url, "second")
        second = db.pending_candidate(qid)
        db.attach_source(
            second["id"],
            source.model_copy(
                update={
                    "canonical_url": second_url,
                    "retrieved_url": second_url,
                    "archive_timestamp": "20140803000000",
                }
            ),
        )
        researcher = Researcher(
            db, NoAssessment(), object(), Config.from_env(), qid, Limits(max_actions=2)
        )
        researcher.deadline = time.monotonic() + 10
        researcher.run_id = db.start_run(
            ResearchRun(research_question_id=qid, model="test", provider="test")
        )
        second = db.pending_candidate(qid)
        assert asyncio.run(researcher._one_candidate(dict(second))) is False
        assert db.pending_candidate(qid) is None


def test_live_fetch_is_unchanged():
    async def fetch():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
            )
        ) as client:
            return await BraveWeb("key", client).fetch("https://example.org/live")

    source = asyncio.run(fetch())
    assert source.retrieved_url == "https://example.org/live"
    assert source.archive_timestamp is None
    assert source.original_url is None
