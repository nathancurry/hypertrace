"""Regression checks for durable query admission and migration."""

from __future__ import annotations

import sqlite3

from hypertrace.db import SCHEMA, Database
from hypertrace.models import ResearchQuestion, ResearchRun, SearchQuery


def proposed(question_id: int, query: str, *, value: str = "high") -> SearchQuery:
    return SearchQuery(
        research_question_id=question_id,
        query=query,
        rationale="Resolve a named source or historical distinction",
        gap="The original usage remains unverified",
        information_value=value,
        novelty="Targets a distinct source or period",
        admission_basis="primary_source" if value == "high" else "unresolved_gap",
    )


def test_paraphrase_does_not_expand_active_frontier(tmp_path):
    with Database(tmp_path / "db.sqlite") as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        for text in (
            "A.G. Cook hyperpop origin",
            "A G Cook origin of hyperpop",
            "interview A.G. Cook term hyperpop",
        ):
            db.persist_plan(qid, [proposed(qid, text)])
        assert [
            tuple(row)
            for row in db.rows(
                "SELECT status,COUNT(*) FROM queries GROUP BY status ORDER BY status"
            )
        ] == [("active", 1), ("rejected_duplicate", 2)]


def test_primary_source_query_outranks_retrospective_and_displacement_is_recoverable(tmp_path):
    with Database(tmp_path / "db.sqlite", active_query_limit=1) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        secondary = db.add_query(
            proposed(qid, "hyperpop history retrospective article", value="medium")
        )
        primary = db.add_query(
            proposed(qid, "Philip Sherburne Pitchfork 2014 hyper-pop original text")
        )
        assert db.rows("SELECT status FROM queries WHERE id=?", (primary,))[0][0] == "active"
        assert db.rows("SELECT status FROM queries WHERE id=?", (secondary,))[0][0] == "deferred"
        assert db.rows("SELECT query,generated_by FROM queries WHERE id=?", (secondary,))[0][0]
        assert not db.activate_deferred_query(qid, secondary)
        db.mark_query_executed(primary)
        assert db.activate_deferred_query(qid, secondary)
        assert db.rows("SELECT status FROM queries WHERE id=?", (secondary,))[0][0] == "active"


def test_frontier_cap_and_exhausted_avenue(tmp_path):
    with Database(tmp_path / "db.sqlite", active_query_limit=2) as db:
        qid = db.add_question(ResearchQuestion(question="Where did hyperpop originate?"))
        ids = [
            db.add_query(proposed(qid, text))
            for text in (
                "Don Shewey Cocteau Twins 1988 original scan",
                "Philip Sherburne Pitchfork 2014 hyper-pop original text",
                "Björk Hyperballad 1995 title explanation original interview",
            )
        ]
        assert db.rows("SELECT COUNT(*) FROM queries WHERE status='active'")[0][0] == 2
        assert db.rows("SELECT status FROM queries WHERE id=?", (ids[2],))[0][0] == "deferred"
        key = db.rows("SELECT avenue FROM queries WHERE id=?", (ids[0],))[0][0]
        db.mark_query_executed(ids[0])
        run_id = db.start_run(ResearchRun(research_question_id=qid, model="test", provider="test"))
        db.persist_review(
            run_id, qid, ids[0], "{}", [], [(key, "Original issue found; avenue answered")]
        )
        variant = db.add_query(proposed(qid, "1988 Don Shewey Cocteau Twins archive original"))
        assert db.rows("SELECT status FROM queries WHERE id=?", (variant,))[0][0] == "exhausted"
        assert db.rows("SELECT reason FROM exhausted_avenues WHERE avenue=?", (key,))[0][0]


def test_legacy_migration_preserves_query_provenance(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO questions VALUES (1,'Where did hyperpop originate?','2026-01-01','active')"
        )
        for query, origin in (
            ("A.G. Cook hyperpop origin", "review-model"),
            ("A G Cook origin of hyperpop", "assessment-model"),
            ("Philip Sherburne Pitchfork 2014 hyper-pop original text", "planner-model"),
        ):
            conn.execute(
                "INSERT INTO queries (research_question_id,query,rationale,generated_by,created_at) "
                "VALUES (1,?,?,?,'2026-01-01')",
                (query, "Find primary evidence", origin),
            )
        for column in (
            "duplicate_of",
            "priority",
            "avenue",
            "admission_basis",
            "novelty",
            "information_value",
            "gap",
            "status",
        ):
            conn.execute(f"ALTER TABLE queries DROP COLUMN {column}")
    with Database(path, active_query_limit=1) as db:
        rows = db.rows("SELECT query,generated_by,created_at,status FROM queries ORDER BY id")
        assert len(rows) == 3
        assert [row["generated_by"] for row in rows] == [
            "review-model",
            "assessment-model",
            "planner-model",
        ]
        assert all(row["created_at"] == "2026-01-01" for row in rows)
        assert {row["status"] for row in rows} == {"active", "deferred", "rejected_duplicate"}
        assert db.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
