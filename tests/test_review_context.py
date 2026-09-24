"""Regression checks for resolved source hunts and bounded review inputs."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hypertrace.db import Database
from hypertrace.models import (
    ContentRegion,
    Evidence,
    Hypothesis,
    QuoteRegion,
    ResearchQuestion,
    SearchQuery,
    Source,
    TermSense,
)
from hypertrace.planner import seed_source_targets
from hypertrace.research import REVIEW_CONTEXT_LIMIT, Limits, Researcher


def source(text: str, *, page_date: str | None = None) -> Source:
    return Source(
        canonical_url="https://example.org/article",
        retrieved_url="https://example.org/article",
        page_publication_date=page_date,
        content=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        content_regions=[ContentRegion(start=0, end=len(text), region=QuoteRegion.ARTICLE_BODY)],
    )


def test_primary_quote_closes_retrieval_but_not_dating(tmp_path):
    path = tmp_path / "case.db"
    with Database(path) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        seed_source_targets(db, qid)
        query_id = db.rows(
            "SELECT q.id FROM queries q JOIN source_targets t ON t.id=q.source_target_id "
            "WHERE t.key='pc-music-2014' LIMIT 1"
        )[0][0]
        quote = "The brand of hyper-pop was everywhere."
        source_id = db.add_source(source(quote, page_date="2014-09-17"))
        db.add_evidence(
            Evidence(
                research_question_id=qid,
                source_id=source_id,
                exact_quote=quote,
                normalized_claim=quote,
                evidence_type="observed_usage",
                primary_source_verified=True,
                discovered_by_query_id=query_id,
                verified_target_id=db.rows(
                    "SELECT source_target_id FROM queries WHERE id=?", (query_id,)
                )[0][0],
                target_verification_note="Verified target source text",
            )
        )
        target = db.rows(
            "SELECT status,evidence_status,dating_status FROM source_targets "
            "WHERE key='pc-music-2014'"
        )[0]
        assert tuple(target) == ("resolved", "found", "unresolved")
        seed_source_targets(db, qid)
        assert db.rows("SELECT quote_verified_date FROM evidence")[0][0] is None
        assert (
            db.rows(
                "SELECT COUNT(*) FROM queries WHERE source_target_id=1 "
                "AND status='active' AND executed_at IS NULL"
            )[0][0]
            == 0
        )
        with pytest.raises(ValueError, match="resolved source target"):
            db.add_query(
                SearchQuery(
                    research_question_id=qid,
                    query="find original Pitchfork article again",
                    source_target="pc-music-2014",
                )
            )
        dating_id = db.add_query(
            SearchQuery(
                research_question_id=qid,
                query="independent dated capture of Sherburne hyper-pop quotation",
                source_target="pc-music-2014",
                target_purpose="dating",
                information_value="high",
            )
        )
        assert db.rows("SELECT status,target_purpose FROM queries WHERE id=?", (dating_id,))[0][
            :
        ] == (
            "active",
            "dating",
        )
    with Database(path) as db:
        assert tuple(
            db.rows(
                "SELECT status,evidence_status,dating_status "
                "FROM source_targets WHERE key='pc-music-2014'"
            )[0]
        ) == ("resolved", "found", "unresolved")
        assert db.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_review_context_keeps_new_direct_and_contradictory_evidence_bounded(tmp_path):
    with Database(tmp_path / "case.db") as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        hypothesis_id = db.add_hypothesis(
            Hypothesis(research_question_id=qid, statement="The term came from one named source.")
        )
        seed_source_targets(db, qid)
        query_id = db.add_query(
            SearchQuery(research_question_id=qid, query="test evidence context")
        )
        text = "\n".join(f"Direct usage number {i}: hyper-pop." for i in range(90))
        source_id = db.add_source(source(text))
        first_id = None
        for i in range(90):
            quote = f"Direct usage number {i}: hyper-pop."
            evidence_id = db.add_evidence(
                Evidence(
                    research_question_id=qid,
                    source_id=source_id,
                    exact_quote=quote,
                    normalized_claim=quote,
                    evidence_type="observed_usage",
                    primary_source_verified=i in {0, 89},
                    term_sense=TermSense.MODERN_GENRE,
                    discovered_by_query_id=query_id,
                )
            )
            if i == 0:
                first_id = evidence_id
                db.add_relationship(
                    evidence_id, hypothesis_id, "contradicts", "Direct counterexample"
                )
            if i == 89:
                db.add_relationship(evidence_id, hypothesis_id, "supports", "Direct source")
        reviewer = Researcher(
            db, object(), object(), SimpleNamespace(research_model="test"), qid, Limits()
        )
        context, counts = reviewer._review_context(query_id)
        ids = {item["id"] for item in context["evidence"]}
        assert first_id in ids
        assert evidence_id in ids
        assert any(
            link["kind"] == "contradicts"
            for item in context["evidence"]
            for link in item["relationships"]
        )
        assert [item["id"] for item in context["hypotheses"]] == [hypothesis_id]
        assert len(context["source_targets"]) == 5
        assert (
            len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))
            <= REVIEW_CONTEXT_LIMIT
        )
        assert counts["evidence"]["omitted"] >= 70
        for i in range(100):
            db.add_query(
                SearchQuery(
                    research_question_id=qid, query=f"historical rejected search number {i}"
                )
            )
        later, later_counts = reviewer._review_context(query_id)
        assert (
            len(json.dumps(later, ensure_ascii=False, separators=(",", ":")))
            <= REVIEW_CONTEXT_LIMIT
        )
        assert {item["id"] for item in later["evidence"]} == ids
        assert later_counts["pending_queries"]["omitted"] > 0
