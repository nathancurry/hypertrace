"""Small SQLite store. Every fetched revision remains an immutable source row."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from hypertrace.frontier import (
    TARGET_ACTIVE_LIMIT,
    TARGET_RANK,
    avenue,
    generic_query,
    legacy_value,
    overlap,
    priority,
    target_priority,
)
from hypertrace.models import (
    EpistemicType,
    Evidence,
    Hypothesis,
    QuoteRegion,
    RelationshipKind,
    ResearchLead,
    ResearchQuestion,
    ResearchRun,
    SearchQuery,
    Source,
    TermSense,
    utc_now,
)
from hypertrace.retrieval.wayback import Snapshot
from hypertrace.retrieval.web import canonicalize_url

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS questions (
  id INTEGER PRIMARY KEY, question TEXT NOT NULL, created_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('active','paused','complete'))
);
CREATE TABLE IF NOT EXISTS hypotheses (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  statement TEXT NOT NULL, status TEXT NOT NULL, rationale TEXT NOT NULL,
  created_at TEXT NOT NULL, UNIQUE(research_question_id, statement)
);
CREATE TABLE IF NOT EXISTS research_notes (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  former_hypothesis_id INTEGER NOT NULL UNIQUE REFERENCES hypotheses(id),
  kind TEXT NOT NULL CHECK(kind IN ('lead','gap','reliability','adjudication','overlap')),
  statement TEXT NOT NULL, rationale TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_targets (
  id INTEGER PRIMARY KEY,
  research_question_id INTEGER NOT NULL REFERENCES questions(id),
  key TEXT NOT NULL, description TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unresolved' CHECK(status IN ('unresolved','resolved')),
  evidence_status TEXT NOT NULL DEFAULT 'unresolved',
  dating_status TEXT NOT NULL DEFAULT 'unresolved',
  UNIQUE(research_question_id,key)
);
CREATE TABLE IF NOT EXISTS archive_targets (
  id INTEGER PRIMARY KEY, source_target_id INTEGER NOT NULL REFERENCES source_targets(id),
  original_url TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
  start_year INTEGER, end_year INTEGER, start_date TEXT, end_date TEXT,
  UNIQUE(source_target_id,original_url)
);
CREATE TABLE IF NOT EXISTS archive_lookups (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  query_id INTEGER NOT NULL REFERENCES queries(id), original_url TEXT NOT NULL,
  status TEXT NOT NULL, detail TEXT NOT NULL, attempted_at TEXT NOT NULL,
  UNIQUE(research_question_id,original_url)
);
CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  query TEXT NOT NULL, rationale TEXT NOT NULL, generated_by TEXT NOT NULL,
  created_at TEXT NOT NULL, executed_at TEXT, reviewed_at TEXT,
  status TEXT NOT NULL DEFAULT 'active', gap TEXT NOT NULL DEFAULT '',
  information_value TEXT NOT NULL DEFAULT 'medium', novelty TEXT NOT NULL DEFAULT '',
  admission_basis TEXT NOT NULL DEFAULT 'unresolved_gap',
  avenue TEXT NOT NULL DEFAULT '', priority INTEGER NOT NULL DEFAULT 0,
  duplicate_of INTEGER REFERENCES queries(id),
  source_target_id INTEGER REFERENCES source_targets(id),
  target_purpose TEXT NOT NULL DEFAULT 'source',
  UNIQUE(research_question_id, query)
);
CREATE TABLE IF NOT EXISTS exhausted_avenues (
  research_question_id INTEGER NOT NULL REFERENCES questions(id),
  avenue TEXT NOT NULL, reason TEXT NOT NULL, retired_at TEXT NOT NULL,
  PRIMARY KEY(research_question_id, avenue)
);
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY, canonical_url TEXT NOT NULL, retrieved_url TEXT NOT NULL,
  title TEXT NOT NULL, author TEXT, publication_date TEXT,
  page_publication_date TEXT, retrieval_date TEXT NOT NULL,
  source_type TEXT NOT NULL, archive_url TEXT, content_hash TEXT NOT NULL,
  original_url TEXT, resolved_original_url TEXT, archive_timestamp TEXT,
  http_status INTEGER, redirect_history_json TEXT NOT NULL DEFAULT '[]',
  document_hash TEXT NOT NULL,
  content TEXT NOT NULL, regions_json TEXT NOT NULL DEFAULT '[]', dating_notes TEXT NOT NULL,
  UNIQUE(canonical_url, document_hash, content_hash)
);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY, query_id INTEGER NOT NULL REFERENCES queries(id),
  url TEXT NOT NULL, title TEXT NOT NULL, source_id INTEGER REFERENCES sources(id),
  assessed_at TEXT, fetch_error TEXT, failure_at TEXT, assessment_error TEXT,
  fetch_status INTEGER, fetch_final_url TEXT, fetch_redirects_json TEXT,
  fetch_retryable INTEGER CHECK(fetch_retryable IN (0,1)),
  original_url TEXT, resolved_original_url TEXT, archive_timestamp TEXT,
  assessment_mode TEXT CHECK(assessment_mode IN ('full_document','relevance_windows')),
  UNIQUE(query_id, url)
);
CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES sources(id),
  research_question_id INTEGER NOT NULL REFERENCES questions(id),
  exact_quote TEXT NOT NULL, quote_start INTEGER NOT NULL,
  quote_region TEXT NOT NULL DEFAULT 'unknown', quote_context TEXT NOT NULL DEFAULT '',
  quote_verified_date TEXT, date_verification_note TEXT NOT NULL DEFAULT '',
  normalized_claim TEXT NOT NULL, relevance TEXT NOT NULL,
  evidence_type TEXT NOT NULL, contemporaneous INTEGER, confidence TEXT NOT NULL,
  interpretation_notes TEXT NOT NULL, term_sense TEXT NOT NULL DEFAULT 'unknown',
  primary_source_verified INTEGER NOT NULL DEFAULT 0,
  transmission_verified INTEGER NOT NULL DEFAULT 0,
  discovered_by_query_id INTEGER REFERENCES queries(id),
  created_at TEXT NOT NULL,
  UNIQUE(source_id, research_question_id, exact_quote, normalized_claim)
);
CREATE TABLE IF NOT EXISTS leads (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  source_id INTEGER NOT NULL REFERENCES sources(id), kind TEXT NOT NULL,
  value TEXT NOT NULL, rationale TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(research_question_id,source_id,kind,value)
);
CREATE TABLE IF NOT EXISTS relationships (
  id INTEGER PRIMARY KEY, evidence_id INTEGER NOT NULL REFERENCES evidence(id),
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id), kind TEXT NOT NULL,
  rationale TEXT NOT NULL, UNIQUE(evidence_id, hypothesis_id, kind)
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  model TEXT NOT NULL, provider TEXT NOT NULL, started_at TEXT NOT NULL,
  ended_at TEXT, status TEXT NOT NULL, actions_taken INTEGER NOT NULL,
  input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
  estimated_cost REAL NOT NULL, stop_reason TEXT NOT NULL,
  provider_requests INTEGER NOT NULL DEFAULT 0,
  unknown_spend_requests INTEGER NOT NULL DEFAULT 0,
  provider_reported_cost REAL NOT NULL DEFAULT 0,
  locally_estimated_cost REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
  occurred_at TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
  estimated_cost REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
  research_question_id INTEGER NOT NULL REFERENCES questions(id),
  created_at TEXT NOT NULL, content_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_retirement_rejections (
  id INTEGER PRIMARY KEY, review_id INTEGER NOT NULL REFERENCES reviews(id),
  avenue_id TEXT NOT NULL, reason TEXT NOT NULL, rejection_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_attempts (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
  research_question_id INTEGER NOT NULL REFERENCES questions(id),
  query_id INTEGER NOT NULL REFERENCES queries(id),
  created_at TEXT NOT NULL, status TEXT NOT NULL,
  error TEXT NOT NULL, diagnostics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypothesis_suggestions (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
  hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
  proposed_status TEXT NOT NULL, rationale TEXT NOT NULL,
  evidence_ids_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_attempts (
  id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id),
  logical_action TEXT NOT NULL, model TEXT NOT NULL, started_at TEXT NOT NULL,
  ended_at TEXT, outcome TEXT NOT NULL, input_tokens INTEGER,
  output_tokens INTEGER, estimated_cost REAL,
  usage_reported INTEGER NOT NULL DEFAULT 0,
  local_input_tokens INTEGER,
  local_output_tokens INTEGER,
  local_estimated_cost REAL,
  provider_billing_unknown INTEGER NOT NULL DEFAULT 1,
  provider TEXT,
  provider_role TEXT,
  attempt_order INTEGER,
  retry_reason TEXT,
  http_status INTEGER,
  diagnostics_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_evidence_question ON evidence(research_question_id);
CREATE INDEX IF NOT EXISTS idx_queries_pending ON queries(research_question_id, executed_at);
CREATE INDEX IF NOT EXISTS idx_candidates_pending ON candidates(assessed_at);
"""


_ORIGIN_WORDS = re.compile(
    r"\b(first|earliest|coined|originated|origin(?: point)?|derived from|influenced by|"
    r"credited as|pivotal moment)\b|\b(?:was|were) created in response to\b",
    re.IGNORECASE,
)
_INTENT_WORDS = re.compile(r"\b(?:intended|intent|meant to|wanted to)\b", re.IGNORECASE)
_INTERPRETIVE_WORDS = re.compile(
    r"\b(?:bore little resemblance|fuses|signif(?:y|ies)|evoking|reflects)\b",
    re.IGNORECASE,
)
_REPORTED_USE = re.compile(
    r"\b(?:was|were|had been)\s+(?:describing|using|used|calling|called)\b|"
    r"\b(?:used|described|called)\b.{0,100}\b(?:as early as|in (?:19|20)\d{2})\b|"
    r"\bas early as (?:19|20)\d{2}\b",
    re.IGNORECASE,
)


def _correct_evidence_type(quote: str, proposed: str) -> str:
    if proposed != EpistemicType.OBSERVED_USAGE:
        return proposed
    if _INTENT_WORDS.search(quote):
        return EpistemicType.ATTRIBUTED_INTENT
    if _ORIGIN_WORDS.search(quote):
        return EpistemicType.ATTRIBUTED_ORIGIN_CLAIM
    if _INTERPRETIVE_WORDS.search(quote):
        return EpistemicType.INTERPRETIVE_CONTEXT
    if _REPORTED_USE.search(quote):
        return EpistemicType.ATTRIBUTED_USAGE
    return proposed


class Database:
    def __init__(self, path: str | Path, active_query_limit: int | None = None):
        self.active_query_limit = (
            int(os.getenv("HYPERTRACE_ACTIVE_QUERY_LIMIT", "50"))
            if active_query_limit is None
            else active_query_limit
        )
        if self.active_query_limit < 1:
            raise ValueError("Active query limit must be positive")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        additions = {
            "source_targets": {
                "evidence_status": "TEXT NOT NULL DEFAULT 'unresolved'",
                "dating_status": "TEXT NOT NULL DEFAULT 'unresolved'",
            },
            "queries": {
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "gap": "TEXT NOT NULL DEFAULT ''",
                "information_value": "TEXT NOT NULL DEFAULT 'medium'",
                "novelty": "TEXT NOT NULL DEFAULT ''",
                "admission_basis": "TEXT NOT NULL DEFAULT 'unresolved_gap'",
                "avenue": "TEXT NOT NULL DEFAULT ''",
                "priority": "INTEGER NOT NULL DEFAULT 0",
                "duplicate_of": "INTEGER REFERENCES queries(id)",
                "source_target_id": "INTEGER REFERENCES source_targets(id)",
                "target_purpose": "TEXT NOT NULL DEFAULT 'source'",
            },
            "hypotheses": {"archived_at": "TEXT"},
            "sources": {
                "page_publication_date": "TEXT",
                "regions_json": "TEXT NOT NULL DEFAULT '[]'",
                "original_url": "TEXT",
                "resolved_original_url": "TEXT",
                "archive_timestamp": "TEXT",
                "http_status": "INTEGER",
                "redirect_history_json": "TEXT NOT NULL DEFAULT '[]'",
            },
            "candidates": {
                "failure_at": "TEXT",
                "assessment_error": "TEXT",
                "assessment_mode": "TEXT",
                "fetch_status": "INTEGER",
                "fetch_final_url": "TEXT",
                "fetch_redirects_json": "TEXT",
                "fetch_retryable": "INTEGER",
                "original_url": "TEXT",
                "resolved_original_url": "TEXT",
                "archive_timestamp": "TEXT",
            },
            "evidence": {
                "quote_region": "TEXT NOT NULL DEFAULT 'unknown'",
                "quote_context": "TEXT NOT NULL DEFAULT ''",
                "quote_verified_date": "TEXT",
                "date_verification_note": "TEXT NOT NULL DEFAULT ''",
                "term_sense": "TEXT NOT NULL DEFAULT 'unknown'",
                "primary_source_verified": "INTEGER NOT NULL DEFAULT 0",
                "transmission_verified": "INTEGER NOT NULL DEFAULT 0",
            },
            "runs": {
                "provider_requests": "INTEGER NOT NULL DEFAULT 0",
                "unknown_spend_requests": "INTEGER NOT NULL DEFAULT 0",
                "provider_reported_cost": "REAL NOT NULL DEFAULT 0",
                "locally_estimated_cost": "REAL NOT NULL DEFAULT 0",
            },
            "provider_attempts": {
                "diagnostics_json": "TEXT NOT NULL DEFAULT '{}'",
                "local_input_tokens": "INTEGER",
                "local_output_tokens": "INTEGER",
                "local_estimated_cost": "REAL",
                "provider_billing_unknown": "INTEGER NOT NULL DEFAULT 1",
                "provider": "TEXT",
                "provider_role": "TEXT",
                "attempt_order": "INTEGER",
                "retry_reason": "TEXT",
                "http_status": "INTEGER",
            },
        }
        legacy_evidence = "quote_region" not in {
            row["name"] for row in self.conn.execute("PRAGMA table_info(evidence)")
        }
        legacy_frontier = "status" not in {
            row["name"] for row in self.conn.execute("PRAGMA table_info(queries)")
        }
        with self.conn:
            needs_cost_backfill = "provider_reported_cost" not in {
                row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")
            }
            for table, columns in additions.items():
                existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                for name, declaration in columns.items():
                    if name not in existing:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
            self._refresh_source_targets_tx()
            self.conn.execute(
                "UPDATE source_targets SET description=REPLACE(description, "
                "' This URL remains unresolved.', "
                "' Independent dating of the quoted text remains unresolved.') "
                "WHERE key='pc-music-2014' AND status='resolved'"
            )
            if needs_cost_backfill:
                self.conn.execute(
                    "UPDATE provider_attempts SET provider_billing_unknown=1-usage_reported"
                )
                self.conn.execute(
                    "UPDATE runs SET provider_reported_cost=COALESCE(("
                    "SELECT SUM(estimated_cost) FROM provider_attempts "
                    "WHERE run_id=runs.id AND usage_reported=1),0)"
                )
            if legacy_frontier:
                self._migrate_frontier_tx()
            else:
                for row in self.conn.execute(
                    "SELECT DISTINCT research_question_id FROM queries WHERE status='active'"
                ).fetchall():
                    self._enforce_frontier_cap_tx(row["research_question_id"])
            self.conn.execute(
                "UPDATE sources SET page_publication_date=publication_date "
                "WHERE page_publication_date IS NULL AND publication_date IS NOT NULL"
            )
            self.conn.execute(
                "UPDATE candidates SET failure_at=NULL,fetch_error=NULL "
                "WHERE source_id IS NOT NULL AND assessed_at IS NULL "
                "AND fetch_error='Fetched text exceeds the 16,000-character assessment window'"
            )
            for row in self.conn.execute(
                "SELECT id,fetch_error,failure_at FROM candidates "
                "WHERE fetch_error IS NOT NULL AND fetch_retryable IS NULL"
            ).fetchall():
                match = re.search(r"\b([45]\d\d) [^']*' for url '([^']+)'", row["fetch_error"])
                status = int(match.group(1)) if match else None
                final_url = match.group(2) if match else None
                retryable = row["failure_at"] is None
                reason = f"HTTP {status}" if status is not None else row["fetch_error"][:200]
                self.conn.execute(
                    "UPDATE candidates SET fetch_error=?,failure_at=?,fetch_status=?,"
                    "fetch_final_url=?,fetch_retryable=? WHERE id=?",
                    (
                        reason,
                        row["failure_at"] or utc_now(),
                        status,
                        final_url,
                        int(retryable),
                        row["id"],
                    ),
                )
            if legacy_evidence:
                self.conn.execute(
                    "UPDATE hypotheses SET status='open',"
                    "rationale='Legacy model status requires renewed verification.' "
                    "WHERE status!='open'"
                )
            for row in self.conn.execute(
                "SELECT id,source_id,research_question_id,exact_quote,evidence_type "
                "FROM evidence WHERE evidence_type='observed_usage'"
            ).fetchall():
                corrected = _correct_evidence_type(row["exact_quote"], row["evidence_type"])
                if corrected == row["evidence_type"]:
                    continue
                self.conn.execute(
                    "UPDATE evidence SET evidence_type=? WHERE id=?", (corrected, row["id"])
                )
                if corrected in {
                    EpistemicType.ATTRIBUTED_USAGE,
                    EpistemicType.ATTRIBUTED_ORIGIN_CLAIM,
                }:
                    self._add_attribution_lead_tx(
                        row["research_question_id"], row["source_id"], row["exact_quote"]
                    )

    def close(self) -> None:
        self.conn.close()

    def _migrate_frontier_tx(self) -> None:
        rows = self.conn.execute("SELECT * FROM queries ORDER BY id").fetchall()
        for row in rows:
            key = avenue(row["query"])
            value = legacy_value(row["query"], row["rationale"])
            basis = "primary_source" if value == "high" else "unresolved_gap"
            self.conn.execute(
                "UPDATE queries SET status=?,gap=?,information_value=?,novelty=?,"
                "admission_basis=?,avenue=?,priority=? WHERE id=?",
                (
                    "executed" if row["executed_at"] else "deferred",
                    row["rationale"] or "Legacy search avenue requiring review",
                    value,
                    "Legacy proposal; compared with the persisted frontier",
                    basis,
                    key,
                    priority(value, row["query"], basis),
                    row["id"],
                ),
            )
        # Three completed searches with no recorded evidence retire an avenue.
        for row in self.conn.execute(
            "SELECT q.research_question_id,q.avenue,COUNT(DISTINCT q.id) searches,"
            "COUNT(DISTINCT CASE WHEN q.reviewed_at IS NOT NULL THEN q.id END) reviewed,"
            "COUNT(DISTINCT e.id) evidence_count,MAX(q.priority) top_priority "
            "FROM queries q LEFT JOIN evidence e "
            "ON e.discovered_by_query_id=q.id WHERE q.executed_at IS NOT NULL "
            "GROUP BY q.research_question_id,q.avenue"
        ).fetchall():
            if (
                row["searches"] >= 3
                and row["reviewed"] == row["searches"]
                and not row["evidence_count"]
                and row["top_priority"] < 35
            ):
                self.conn.execute(
                    "INSERT OR IGNORE INTO exhausted_avenues VALUES (?,?,?,?)",
                    (
                        row["research_question_id"],
                        row["avenue"],
                        "Three reviewed searches yielded no evidence",
                        utc_now(),
                    ),
                )
        pending = self.conn.execute(
            "SELECT * FROM queries WHERE executed_at IS NULL ORDER BY priority DESC,id"
        ).fetchall()
        kept: dict[int, list[sqlite3.Row]] = {}
        executed = self.conn.execute(
            "SELECT * FROM queries WHERE executed_at IS NOT NULL"
        ).fetchall()
        for row in pending:
            qid = row["research_question_id"]
            exhausted = self.conn.execute(
                "SELECT 1 FROM exhausted_avenues WHERE research_question_id=? AND avenue=?",
                (qid, row["avenue"]),
            ).fetchone()
            if exhausted:
                status, duplicate_of = "exhausted", None
            else:
                matching = next(
                    (old for old in kept.get(qid, []) if overlap(row["query"], old["query"])),
                    None,
                )
                if matching is None:
                    matching = next(
                        (
                            old
                            for old in executed
                            if old["research_question_id"] == qid
                            and old["priority"] >= row["priority"]
                            and overlap(row["query"], old["query"])
                            and (
                                row["admission_basis"] != "primary_source"
                                or self._query_has_primary_evidence(old["id"])
                            )
                        ),
                        None,
                    )
                if matching:
                    status, duplicate_of = "rejected_duplicate", matching["id"]
                else:
                    kept.setdefault(qid, []).append(row)
                    active_count = self.conn.execute(
                        "SELECT COUNT(*) FROM queries WHERE research_question_id=? AND status='active'",
                        (qid,),
                    ).fetchone()[0]
                    status = (
                        "active"
                        if row["information_value"] != "low"
                        and active_count < self.active_query_limit
                        else "deferred"
                    )
                    duplicate_of = None
            self.conn.execute(
                "UPDATE queries SET status=?,duplicate_of=? WHERE id=?",
                (status, duplicate_of, row["id"]),
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _insert(self, table: str, values: dict) -> int:
        keys = list(values)
        sql = f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})"
        with self.conn:
            cursor = self.conn.execute(sql, tuple(values.values()))
        return int(cursor.lastrowid)

    def add_question(self, question: ResearchQuestion) -> int:
        return self._insert("questions", question.model_dump(exclude={"id"}, mode="json"))

    def add_hypothesis(self, hypothesis: Hypothesis) -> int:
        values = hypothesis.model_dump(exclude={"id"}, mode="json")
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO hypotheses "
                "(research_question_id,statement,status,rationale,created_at) VALUES (?,?,?,?,?)",
                tuple(values.values()),
            )
        row = self.conn.execute(
            "SELECT id FROM hypotheses WHERE research_question_id=? AND statement=?",
            (hypothesis.research_question_id, hypothesis.statement),
        ).fetchone()
        return int(row["id"])

    def reclassify_hypotheses(self, question_id: int, kinds: dict[int, str]) -> None:
        """Move legacy proposals into typed notes while retaining all historical links."""
        allowed = {"lead", "gap", "reliability", "adjudication", "overlap"}
        if set(kinds.values()) - allowed:
            raise ValueError("Unknown research note kind")
        with self.conn:
            for hypothesis_id, kind in kinds.items():
                row = self.conn.execute(
                    "SELECT * FROM hypotheses WHERE id=? AND research_question_id=?",
                    (hypothesis_id, question_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"H{hypothesis_id} does not belong to question {question_id}")
                existing = self.conn.execute(
                    "SELECT kind FROM research_notes WHERE former_hypothesis_id=?",
                    (hypothesis_id,),
                ).fetchone()
                if existing:
                    if existing["kind"] != kind:
                        raise ValueError(
                            f"H{hypothesis_id} already classified as {existing['kind']}"
                        )
                    continue
                self.conn.execute(
                    "INSERT INTO research_notes "
                    "(research_question_id,former_hypothesis_id,kind,statement,rationale,created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        question_id,
                        hypothesis_id,
                        kind,
                        row["statement"],
                        row["rationale"],
                        row["created_at"],
                    ),
                )
                self.conn.execute(
                    "UPDATE hypotheses SET archived_at=? WHERE id=?",
                    (utc_now(), hypothesis_id),
                )

    def add_query(self, query: SearchQuery) -> int:
        with self.conn:
            return self._add_query_tx(query)

    def install_source_targets(
        self, question_id: int, targets: list[tuple[str, str]], queries: list[SearchQuery]
    ) -> None:
        """Install a focused source hunt without changing executed research records."""
        with self.conn:
            if not self.conn.execute(
                "SELECT 1 FROM questions WHERE id=?", (question_id,)
            ).fetchone():
                raise ValueError("Unknown research question")
            for key, description in targets:
                if key not in TARGET_RANK:
                    raise ValueError("Unknown source target key")
                self.conn.execute(
                    "INSERT OR IGNORE INTO source_targets "
                    "(research_question_id,key,description) VALUES (?,?,?)",
                    (question_id, key, description),
                )
            for query in queries:
                if query.research_question_id != question_id or query.source_target is None:
                    raise ValueError("Seed query needs a source target in this question")
                self._add_query_tx(query)
            self._enforce_frontier_cap_tx(question_id)

    def _refresh_source_targets_tx(self) -> None:
        # Direct, body-text evidence closes retrieval; page metadata never closes dating.
        self.conn.execute(
            "UPDATE source_targets SET status='resolved',evidence_status='found',"
            "dating_status=CASE WHEN EXISTS (SELECT 1 FROM evidence e "
            "JOIN queries q ON q.id=e.discovered_by_query_id "
            "WHERE q.source_target_id=source_targets.id AND e.primary_source_verified=1 "
            "AND e.quote_region='article_body' AND e.quote_verified_date IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM evidence dated "
            "JOIN queries dq ON dq.id=dated.discovered_by_query_id "
            "WHERE dq.source_target_id=source_targets.id "
            "AND dated.exact_quote=e.exact_quote AND dated.primary_source_verified=1 "
            "AND dated.quote_region='article_body' "
            "AND dated.quote_verified_date IS NOT NULL)) "
            "THEN 'unresolved' ELSE 'resolved' END "
            "WHERE EXISTS (SELECT 1 FROM evidence e "
            "JOIN queries q ON q.id=e.discovered_by_query_id "
            "WHERE q.source_target_id=source_targets.id AND e.primary_source_verified=1 "
            "AND e.quote_region='article_body')"
        )
        self.conn.execute(
            "UPDATE queries SET status='deferred' WHERE executed_at IS NULL "
            "AND status='active' AND source_target_id IN "
            "(SELECT id FROM source_targets t WHERE "
            "(t.status='resolved' AND queries.target_purpose='source') OR "
            "(t.dating_status='resolved' AND queries.target_purpose='dating'))"
        )

    def _add_query_tx(self, query: SearchQuery, *, proposed: bool = False) -> int:
        if proposed and not (query.gap.strip() and query.novelty.strip()):
            raise ValueError("Proposed query requires an unresolved gap and novelty explanation")
        if query.information_value not in {"high", "medium", "low"}:
            raise ValueError("Invalid information value")
        if query.admission_basis not in {
            "unresolved_gap",
            "hypothesis_distinction",
            "primary_source",
            "new_avenue",
        }:
            raise ValueError("Invalid admission basis")
        target_id = None
        if query.source_target is not None:
            target = self.conn.execute(
                "SELECT id FROM source_targets WHERE research_question_id=? AND key=? "
                "AND ((status='unresolved' AND ?='source') OR "
                "(status='resolved' AND dating_status='unresolved' AND ?='dating'))",
                (
                    query.research_question_id,
                    query.source_target,
                    query.target_purpose,
                    query.target_purpose,
                ),
            ).fetchone()
            if target is None:
                raise ValueError("Unknown or resolved source target for this purpose")
            target_id = int(target["id"])
        elif query.target_purpose == "dating":
            raise ValueError("Dating query requires a source target")
        key = avenue(query.query)
        score = (
            target_priority(query.source_target, query.query, query.information_value)
            if query.source_target is not None
            else priority(query.information_value, query.query, query.admission_basis)
        )
        row = self.conn.execute(
            "SELECT id,source_target_id,target_purpose,executed_at,status,priority FROM queries "
            "WHERE research_question_id=? AND query=?",
            (query.research_question_id, query.query),
        ).fetchone()
        if row:
            if target_id is not None:
                if row["source_target_id"] not in (None, target_id):
                    raise ValueError("Query already belongs to another source target")
                if (
                    row["source_target_id"] is not None
                    and row["target_purpose"] != query.target_purpose
                ):
                    raise ValueError("Query already has another target purpose")
                status = (
                    "deferred"
                    if row["executed_at"] is None
                    and row["status"] in {"rejected_duplicate", "exhausted"}
                    else row["status"]
                )
                self.conn.execute(
                    "UPDATE queries SET source_target_id=?,target_purpose=?,priority=?,"
                    "status=?,duplicate_of=NULL "
                    "WHERE id=?",
                    (
                        target_id,
                        query.target_purpose,
                        max(score, row["priority"]),
                        status,
                        row["id"],
                    ),
                )
                self._enforce_frontier_cap_tx(query.research_question_id)
            return int(row["id"])
        exhausted = self.conn.execute(
            "SELECT 1 FROM exhausted_avenues WHERE research_question_id=? AND avenue=?",
            (query.research_question_id, key),
        ).fetchone()
        existing = self.conn.execute(
            "SELECT id,query,status,priority FROM queries WHERE research_question_id=? "
            "AND status!='rejected_duplicate' ORDER BY priority DESC,id",
            (query.research_question_id,),
        ).fetchall()
        duplicate = (
            None
            if target_id
            else next(
                (
                    old
                    for old in existing
                    if overlap(query.query, old["query"])
                    and (
                        old["status"] != "executed"
                        or query.admission_basis != "primary_source"
                        or self._query_has_primary_evidence(old["id"])
                    )
                ),
                None,
            )
        )
        if exhausted and target_id is None:
            status = "exhausted"
        elif duplicate and duplicate["priority"] >= score:
            status = "rejected_duplicate"
        elif proposed and query.information_value == "low":
            status = "deferred"
        else:
            status = "active"
        cursor = self.conn.execute(
            "INSERT INTO queries (research_question_id,query,rationale,generated_by,created_at,"
            "executed_at,status,gap,information_value,novelty,admission_basis,avenue,priority,"
            "duplicate_of,source_target_id,target_purpose) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                query.research_question_id,
                query.query,
                query.rationale,
                query.generated_by,
                query.created_at,
                query.executed_at,
                status,
                query.gap or query.rationale,
                query.information_value,
                query.novelty or "Directly entered search",
                query.admission_basis,
                key,
                score,
                duplicate["id"] if status == "rejected_duplicate" else None,
                target_id,
                query.target_purpose,
            ),
        )
        query_id = int(cursor.lastrowid)
        if duplicate and status == "active" and duplicate["status"] in {"active", "deferred"}:
            self.conn.execute(
                "UPDATE queries SET status='rejected_duplicate',duplicate_of=? WHERE id=?",
                (query_id, duplicate["id"]),
            )
        self._enforce_frontier_cap_tx(query.research_question_id)
        return query_id

    def _query_has_primary_evidence(self, query_id: int) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM evidence WHERE discovered_by_query_id=? "
                "AND primary_source_verified=1 LIMIT 1",
                (query_id,),
            ).fetchone()
            is not None
        )

    def _enforce_frontier_cap_tx(self, question_id: int) -> None:
        targets = self.conn.execute(
            "SELECT id,status,dating_status FROM source_targets WHERE research_question_id=?",
            (question_id,),
        ).fetchall()
        if targets:
            allowed = {
                (row["id"], purpose)
                for row in targets
                for purpose in (
                    ["source"]
                    if row["status"] == "unresolved"
                    else ["dating"]
                    if row["dating_status"] == "unresolved"
                    else []
                )
            }
            rows = self.conn.execute(
                "SELECT q.id,q.source_target_id,q.target_purpose,q.query,q.information_value "
                "FROM queries q "
                "WHERE q.research_question_id=? "
                "AND q.executed_at IS NULL AND q.status IN ('active','deferred') "
                "ORDER BY q.priority DESC,q.id",
                (question_id,),
            ).fetchall()
            used: dict[int, int] = {}
            active_count = 0
            for row in rows:
                target_id = row["source_target_id"]
                active = (
                    (target_id, row["target_purpose"]) in allowed
                    and row["information_value"] != "low"
                    and not generic_query(row["query"])
                    and used.get(target_id, 0) < TARGET_ACTIVE_LIMIT
                    and active_count < self.active_query_limit
                )
                if active:
                    used[target_id] = used.get(target_id, 0) + 1
                    active_count += 1
                self.conn.execute(
                    "UPDATE queries SET status=? WHERE id=?",
                    ("active" if active else "deferred", row["id"]),
                )
            return
        active = self.conn.execute(
            "SELECT id FROM queries WHERE research_question_id=? AND status='active' "
            "AND executed_at IS NULL ORDER BY priority DESC,id",
            (question_id,),
        ).fetchall()
        for row in active[self.active_query_limit :]:
            self.conn.execute("UPDATE queries SET status='deferred' WHERE id=?", (row["id"],))

    def activate_deferred_query(self, question_id: int, query_id: int) -> bool:
        """Reconsider a saved lead without bypassing priority or the active cap."""
        with self.conn:
            row = self.conn.execute(
                "SELECT priority,source_target_id,query FROM queries "
                "WHERE id=? AND research_question_id=? "
                "AND status='deferred' AND executed_at IS NULL",
                (query_id, question_id),
            ).fetchone()
            if row is None:
                raise ValueError("Query is not a deferred lead")
            if self.conn.execute(
                "SELECT 1 FROM source_targets WHERE research_question_id=?",
                (question_id,),
            ).fetchone():
                if row["source_target_id"] is None or generic_query(row["query"]):
                    return False
                self.conn.execute("UPDATE queries SET status='active' WHERE id=?", (query_id,))
                self._enforce_frontier_cap_tx(question_id)
                return bool(
                    self.conn.execute(
                        "SELECT 1 FROM queries WHERE id=? AND status='active'", (query_id,)
                    ).fetchone()
                )
            active = self.conn.execute(
                "SELECT priority FROM queries WHERE research_question_id=? AND status='active' "
                "ORDER BY priority DESC,id LIMIT ?",
                (question_id, self.active_query_limit),
            ).fetchall()
            if len(active) >= self.active_query_limit and row["priority"] <= active[-1]["priority"]:
                return False
            self.conn.execute("UPDATE queries SET status='active' WHERE id=?", (query_id,))
            self._enforce_frontier_cap_tx(question_id)
            return True

    def retire_avenue(self, question_id: int, key: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Avenue retirement requires a reason")
        with self.conn:
            self._retire_avenue_tx(question_id, key, reason)

    def _retire_avenue_tx(self, question_id: int, key: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Avenue retirement requires a reason")
        if not self.conn.execute(
            "SELECT 1 FROM queries WHERE research_question_id=? AND avenue=?",
            (question_id, key),
        ).fetchone():
            raise ValueError("Unknown research avenue")
        if (
            self.conn.execute(
                "SELECT 1 FROM queries WHERE research_question_id=? AND avenue=? "
                "AND source_target_id IS NOT NULL LIMIT 1",
                (question_id, key),
            ).fetchone()
            and not self.conn.execute(
                "SELECT 1 FROM evidence e JOIN queries q ON q.id=e.discovered_by_query_id "
                "WHERE q.research_question_id=? AND q.avenue=? "
                "AND e.primary_source_verified=1 LIMIT 1",
                (question_id, key),
            ).fetchone()
        ):
            raise ValueError("Unresolved source target cannot be retired after search failures")
        self.conn.execute(
            "INSERT OR IGNORE INTO exhausted_avenues VALUES (?,?,?,?)",
            (question_id, key, reason, utc_now()),
        )
        self.conn.execute(
            "UPDATE queries SET status='exhausted' WHERE research_question_id=? "
            "AND avenue=? AND executed_at IS NULL AND status!='rejected_duplicate'",
            (question_id, key),
        )

    def mark_query_executed(self, query_id: int) -> None:
        with self.conn:
            row = self.conn.execute(
                "SELECT research_question_id FROM queries WHERE id=?", (query_id,)
            ).fetchone()
            self.conn.execute(
                "UPDATE queries SET executed_at=?,status='executed' WHERE id=?",
                (utc_now(), query_id),
            )
            if row:
                self._enforce_frontier_cap_tx(row["research_question_id"])

    def review_due(self, question_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT q.id FROM queries q WHERE q.research_question_id=? "
            "AND q.executed_at IS NOT NULL AND q.reviewed_at IS NULL "
            "AND NOT EXISTS (SELECT 1 FROM candidates c WHERE c.query_id=q.id "
            "AND c.assessed_at IS NULL AND c.failure_at IS NULL) ORDER BY q.id LIMIT 1",
            (question_id,),
        ).fetchone()
        return int(row["id"]) if row else None

    def mark_query_reviewed(self, query_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE queries SET reviewed_at=? WHERE id=?", (utc_now(), query_id))

    def add_candidate(self, query_id: int, url: str, title: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO candidates (query_id,url,title) VALUES (?,?,?)",
                (query_id, url, title),
            )

    def add_archive_target(
        self,
        question_id: int,
        key: str,
        url: str,
        title: str = "",
        start_year: int | None = None,
        end_year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        url = canonicalize_url(url)
        if start_year and end_year and end_year < start_year:
            raise ValueError("Archive target end year precedes start year")
        first = date.fromisoformat(start_date) if start_date else None
        last = date.fromisoformat(end_date) if end_date else None
        if first and last and last < first:
            raise ValueError("Archive target end date precedes start date")
        with self.conn:
            target = self.conn.execute(
                "SELECT id FROM source_targets WHERE research_question_id=? AND key=? "
                "AND status='unresolved'",
                (question_id, key),
            ).fetchone()
            if target is None:
                raise ValueError("Unknown or resolved source target")
            self.conn.execute(
                "INSERT INTO archive_targets "
                "(source_target_id,original_url,title,start_year,end_year,start_date,end_date) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(source_target_id,original_url) DO UPDATE SET "
                "title=CASE WHEN excluded.title!='' THEN excluded.title ELSE archive_targets.title END,"
                "start_year=COALESCE(excluded.start_year,archive_targets.start_year),"
                "end_year=COALESCE(excluded.end_year,archive_targets.end_year),"
                "start_date=COALESCE(excluded.start_date,archive_targets.start_date),"
                "end_date=COALESCE(excluded.end_date,archive_targets.end_date)",
                (target["id"], url, title, start_year, end_year, start_date, end_date),
            )

    def next_archive_lookup(self, question_id: int) -> dict | None:
        targets = self.conn.execute(
            "SELECT a.original_url,a.title,a.start_year,a.end_year,a.start_date,a.end_date,"
            "MIN(q.id) query_id "
            "FROM archive_targets a JOIN source_targets t ON t.id=a.source_target_id "
            "JOIN queries q ON q.source_target_id=t.id "
            "WHERE t.research_question_id=? AND t.status='unresolved' "
            "GROUP BY a.id ORDER BY a.id",
            (question_id,),
        ).fetchall()
        failed = self.conn.execute(
            "SELECT c.url original_url,c.title,q.id query_id,a.start_year,a.end_year,"
            "a.start_date,a.end_date,"
            "a.original_url target_url "
            "FROM candidates c JOIN queries q ON q.id=c.query_id "
            "JOIN archive_targets a ON a.source_target_id=q.source_target_id "
            "JOIN source_targets t ON t.id=q.source_target_id "
            "WHERE q.research_question_id=? AND q.source_target_id IS NOT NULL "
            "AND t.status='unresolved' "
            "AND c.failure_at IS NOT NULL AND c.original_url IS NULL ORDER BY c.id",
            (question_id,),
        ).fetchall()
        for row in [*targets, *failed]:
            item = dict(row)
            try:
                url = canonicalize_url(item["original_url"])
            except ValueError:
                continue
            if urlsplit(url).hostname == "web.archive.org":
                continue
            if "target_url" in item and urlsplit(url).path.rstrip("/") != urlsplit(
                item["target_url"]
            ).path.rstrip("/"):
                continue
            if self.conn.execute(
                "SELECT 1 FROM archive_lookups WHERE research_question_id=? AND original_url=?",
                (question_id, url),
            ).fetchone():
                continue
            return item | {"original_url": url}
        return None

    def record_archive_lookup(
        self,
        question_id: int,
        query_id: int,
        original_url: str,
        snapshots: list[Snapshot],
        detail: str,
    ) -> None:
        with self.conn:
            query = self.conn.execute(
                "SELECT research_question_id FROM queries WHERE id=?", (query_id,)
            ).fetchone()
            if query is None or query["research_question_id"] != question_id:
                raise ValueError("Archive lookup query must belong to the question")
            self.conn.execute(
                "INSERT INTO archive_lookups "
                "(research_question_id,query_id,original_url,status,detail,attempted_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    question_id,
                    query_id,
                    original_url,
                    "snapshots" if snapshots else "gap",
                    detail,
                    utc_now(),
                ),
            )
            for snapshot in snapshots:
                self.conn.execute(
                    "INSERT OR IGNORE INTO candidates "
                    "(query_id,url,title,original_url,resolved_original_url,archive_timestamp) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        query_id,
                        snapshot.url,
                        "Archived snapshot",
                        original_url,
                        snapshot.original_url,
                        snapshot.timestamp,
                    ),
                )

    def persist_plan(self, question_id: int, queries: list[SearchQuery]) -> None:
        with self.conn:
            for query in queries:
                if query.research_question_id != question_id:
                    raise ValueError("Planned query belongs to another question")
                self._add_query_tx(query, proposed=True)

    def persist_search_results(
        self, question_id: int, query_id: int, results: list[tuple[str, str]]
    ) -> None:
        with self.conn:
            query = self.conn.execute(
                "SELECT research_question_id,executed_at,status FROM queries WHERE id=?",
                (query_id,),
            ).fetchone()
            if (
                query is None
                or query["research_question_id"] != question_id
                or query["executed_at"]
                or query["status"] != "active"
            ):
                raise ValueError("Search query is unavailable")
            for url, title in results:
                self.conn.execute(
                    "INSERT OR IGNORE INTO candidates (query_id,url,title) VALUES (?,?,?)",
                    (query_id, url, title),
                )
            self.conn.execute(
                "UPDATE queries SET executed_at=?,status='executed' WHERE id=?",
                (utc_now(), query_id),
            )

    def pending_candidate(self, question_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT c.* FROM candidates c JOIN queries q ON q.id=c.query_id "
            "WHERE q.research_question_id=? AND c.assessed_at IS NULL "
            "AND c.failure_at IS NULL "
            "ORDER BY c.id LIMIT 1",
            (question_id,),
        ).fetchone()

    def set_candidate_source(self, candidate_id: int, source_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET source_id=? WHERE id=?", (source_id, candidate_id)
            )

    def finish_candidate(self, candidate_id: int, error: str | None = None) -> None:
        if error is not None:
            raise ValueError("Fetch failures must remain unresolved")
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET assessed_at=?,fetch_error=NULL,assessment_error=NULL,"
                "fetch_status=NULL,fetch_final_url=NULL,fetch_redirects_json=NULL,"
                "fetch_retryable=NULL "
                "WHERE id=?",
                (utc_now(), candidate_id),
            )

    def record_fetch_failure(
        self,
        candidate_id: int,
        reason: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        final_url: str | None = None,
        redirect_chain: list[str] | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET fetch_error=?,failure_at=?,fetch_status=?,"
                "fetch_final_url=?,fetch_redirects_json=?,fetch_retryable=? WHERE id=?",
                (
                    reason[:200],
                    utc_now(),
                    status_code,
                    final_url,
                    json.dumps(redirect_chain) if redirect_chain is not None else None,
                    int(retryable),
                    candidate_id,
                ),
            )

    def requeue_retryable_fetches(self, question_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET failure_at=NULL WHERE fetch_retryable=1 "
                "AND query_id IN (SELECT id FROM queries WHERE research_question_id=?)",
                (question_id,),
            )

    def record_assessment_failure(self, candidate_id: int, error: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET assessment_error=? "
                "WHERE id=? AND source_id IS NOT NULL AND assessed_at IS NULL",
                (error, candidate_id),
            )

    def document_already_assessed(self, content_hash: str, question_id: int) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM candidates c JOIN sources s ON s.id=c.source_id "
                "JOIN queries q ON q.id=c.query_id "
                "WHERE s.content_hash=? AND q.research_question_id=? "
                "AND c.assessed_at IS NOT NULL LIMIT 1",
                (content_hash, question_id),
            ).fetchone()
            is not None
        )

    def add_source(self, source: Source) -> int:
        with self.conn:
            return self._add_source_tx(source)

    def attach_source(self, candidate_id: int, source: Source) -> int:
        with self.conn:
            candidate = self.conn.execute(
                "SELECT source_id,assessed_at,failure_at FROM candidates WHERE id=?",
                (candidate_id,),
            ).fetchone()
            if (
                candidate is None
                or candidate["source_id"] is not None
                or candidate["assessed_at"]
                or candidate["failure_at"]
            ):
                raise ValueError("Candidate is unavailable for source attachment")
            source_id = self._add_source_tx(source)
            self.conn.execute(
                "UPDATE candidates SET source_id=?,fetch_error=NULL,fetch_status=NULL,"
                "fetch_final_url=NULL,fetch_redirects_json=NULL,fetch_retryable=NULL "
                "WHERE id=?",
                (source_id, candidate_id),
            )
        return source_id

    def _add_source_tx(self, source: Source) -> int:
        if hashlib.sha256(source.content.encode()).hexdigest() != source.content_hash:
            raise ValueError("Source content hash does not match stored text")
        if source.document_hash is None:
            source = source.model_copy(update={"document_hash": source.content_hash})
        for region in source.content_regions:
            if not 0 <= region.start < region.end <= len(source.content):
                raise ValueError("Source region lies outside stored text")
        self.conn.execute(
            "INSERT OR IGNORE INTO sources "
            "(canonical_url,retrieved_url,title,author,page_publication_date,retrieval_date,"
            "source_type,archive_url,content_hash,document_hash,content,dating_notes,regions_json,"
            "original_url,resolved_original_url,archive_timestamp,http_status,redirect_history_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                source.canonical_url,
                source.retrieved_url,
                source.title,
                source.author,
                source.page_publication_date,
                source.retrieval_date,
                source.source_type,
                source.archive_url,
                source.content_hash,
                source.document_hash,
                source.content,
                source.dating_notes,
                json.dumps([r.model_dump(mode="json") for r in source.content_regions]),
                source.original_url,
                source.resolved_original_url,
                source.archive_timestamp,
                source.http_status,
                json.dumps(source.redirect_history),
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM sources WHERE canonical_url=? AND document_hash=? AND content_hash=?",
            (source.canonical_url, source.document_hash, source.content_hash),
        ).fetchone()
        return int(row["id"])

    def add_evidence(self, evidence: Evidence) -> int:
        with self.conn:
            return self._add_evidence_tx(evidence)

    def quote_region(self, source_id: int, exact_quote: str) -> QuoteRegion | None:
        source = self.conn.execute(
            "SELECT content,regions_json FROM sources WHERE id=?", (source_id,)
        ).fetchone()
        if source is None:
            return None
        quote_start = source["content"].find(exact_quote)
        if quote_start < 0:
            return None
        quote_end = quote_start + len(exact_quote)
        overlaps = {
            (item["region"], item.get("block_id", 0))
            for item in json.loads(source["regions_json"])
            if item["start"] < quote_end and item["end"] > quote_start
        }
        return QuoteRegion(next(iter(overlaps))[0]) if len(overlaps) == 1 else QuoteRegion.UNKNOWN

    def quote_target(self, source_id: int, exact_quote: str) -> str | None:
        source = self.conn.execute(
            "SELECT content,regions_json FROM sources WHERE id=?", (source_id,)
        ).fetchone()
        if source is None:
            return None
        start = source["content"].find(exact_quote)
        if start < 0:
            return None
        targets = {
            item["target_url"]
            for item in json.loads(source["regions_json"])
            if item["start"] < start + len(exact_quote)
            and item["end"] > start
            and item.get("target_url")
        }
        return next(iter(targets)) if len(targets) == 1 else None

    def _add_evidence_tx(self, evidence: Evidence) -> int:
        source = self.conn.execute(
            "SELECT content,regions_json,source_type FROM sources WHERE id=?", (evidence.source_id,)
        ).fetchone()
        quote_start = -1 if source is None else source["content"].find(evidence.exact_quote)
        if quote_start < 0:
            raise ValueError("Evidence quote must occur verbatim in the stored source text")
        if source["source_type"] == "search_aggregator":
            raise ValueError("Aggregator snippets cannot be evidence")
        corrected_type = _correct_evidence_type(evidence.exact_quote, evidence.evidence_type.value)
        if corrected_type != evidence.evidence_type.value:
            evidence = evidence.model_copy(update={"evidence_type": EpistemicType(corrected_type)})
        quote_end = quote_start + len(evidence.exact_quote)
        region = self.quote_region(evidence.source_id, evidence.exact_quote)
        if (
            evidence.evidence_type.value
            in {
                "observed_usage",
                "attributed_usage",
                "attributed_origin_claim",
                "attributed_intent",
                "chronological_precedence",
            }
            and region != QuoteRegion.ARTICLE_BODY
            and evidence.quote_verified_date is None
        ):
            raise ValueError("Historical usage requires article-body or verified contemporary text")
        if evidence.quote_verified_date is not None:
            try:
                verified_date = date.fromisoformat(evidence.quote_verified_date)
            except ValueError as exc:
                raise ValueError("Invalid verified quote date") from exc
            if verified_date.isoformat() != evidence.quote_verified_date:
                raise ValueError("Verified quote dates must use YYYY-MM-DD")
            if not evidence.date_verification_note.strip():
                raise ValueError("Verified quote dates require an independent verification note")
        if (
            evidence.evidence_type.value == "chronological_precedence"
            and not evidence.quote_verified_date
        ):
            raise ValueError("Chronological precedence requires a verified quote date")
        if evidence.evidence_type.value == "demonstrated_transmission":
            if not (evidence.primary_source_verified and evidence.transmission_verified):
                raise ValueError("Demonstrated transmission requires verified primary evidence")
            if not re.search(
                r"\b(named after|borrowed from|derived from|inspired by|coined from|took (?:the )?name from)\b",
                evidence.exact_quote,
                re.IGNORECASE,
            ):
                raise ValueError(
                    "Verified transmission quote must explicitly state the relationship"
                )
        if evidence.discovered_by_query_id is not None:
            query = self.conn.execute(
                "SELECT research_question_id FROM queries WHERE id=?",
                (evidence.discovered_by_query_id,),
            ).fetchone()
            if query is None or query["research_question_id"] != evidence.research_question_id:
                raise ValueError("Discovery query must belong to the evidence question")
        if evidence.normalized_claim != evidence.exact_quote:
            raise ValueError("Normalized claim must match the exact excerpt")
        values = evidence.model_dump(exclude={"id"}, mode="json")
        values["quote_start"] = quote_start
        values["quote_region"] = region.value
        values["quote_context"] = source["content"][
            max(0, quote_start - 240) : min(len(source["content"]), quote_end + 240)
        ]
        columns = list(values)
        self.conn.execute(
            f"INSERT OR IGNORE INTO evidence ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            tuple(values.values()),
        )
        row = self.conn.execute(
            "SELECT id FROM evidence WHERE source_id=? AND research_question_id=? "
            "AND exact_quote=? AND normalized_claim=?",
            (
                evidence.source_id,
                evidence.research_question_id,
                evidence.exact_quote,
                evidence.normalized_claim,
            ),
        ).fetchone()
        self._refresh_source_targets_tx()
        if evidence.evidence_type in {
            EpistemicType.ATTRIBUTED_USAGE,
            EpistemicType.ATTRIBUTED_ORIGIN_CLAIM,
        }:
            self._add_attribution_lead_tx(
                evidence.research_question_id, evidence.source_id, evidence.exact_quote
            )
        return int(row["id"])

    def _add_attribution_lead_tx(self, question_id: int, source_id: int, quote: str) -> None:
        self._add_lead_tx(
            ResearchLead(
                research_question_id=question_id,
                source_id=source_id,
                kind="citation",
                value=quote[:500],
                rationale="Find the historical source or artifact identified here and verify its text and date.",
            )
        )

    def add_lead(self, lead: ResearchLead) -> int:
        with self.conn:
            return self._add_lead_tx(lead)

    def _add_lead_tx(self, lead: ResearchLead) -> int:
        source = self.conn.execute(
            "SELECT content FROM sources WHERE id=?", (lead.source_id,)
        ).fetchone()
        if source is None or lead.value not in source["content"]:
            raise ValueError("Lead value must occur verbatim in the stored source text")
        values = lead.model_dump(exclude={"id"}, mode="json")
        self.conn.execute(
            "INSERT OR IGNORE INTO leads "
            "(research_question_id,source_id,kind,value,rationale,created_at) "
            "VALUES (?,?,?,?,?,?)",
            tuple(values.values()),
        )
        row = self.conn.execute(
            "SELECT id FROM leads WHERE research_question_id=? AND source_id=? "
            "AND kind=? AND value=?",
            (lead.research_question_id, lead.source_id, lead.kind.value, lead.value),
        ).fetchone()
        return int(row["id"])

    def add_relationship(
        self, evidence_id: int, hypothesis_id: int, kind: str, rationale: str
    ) -> None:
        with self.conn:
            self._add_relationship_tx(evidence_id, hypothesis_id, kind, rationale)

    def _add_relationship_tx(
        self, evidence_id: int, hypothesis_id: int, kind: str, rationale: str
    ) -> None:
        kind = RelationshipKind(kind).value
        evidence = self.conn.execute(
            "SELECT research_question_id,evidence_type,term_sense,transmission_verified "
            "FROM evidence WHERE id=?",
            (evidence_id,),
        ).fetchone()
        hypothesis = self.conn.execute(
            "SELECT research_question_id FROM hypotheses WHERE id=? AND archived_at IS NULL",
            (hypothesis_id,),
        ).fetchone()
        if evidence is None or hypothesis is None or evidence[0] != hypothesis[0]:
            raise ValueError("Relationship records must belong to the same research question")
        allowed = {
            "possible_transmission": {
                "possible_transmission",
                "attributed_transmission",
                "demonstrated_transmission",
            },
            "attributed_transmission": {"attributed_transmission", "demonstrated_transmission"},
            "demonstrated_transmission": {"demonstrated_transmission"},
        }
        if kind in allowed and evidence["evidence_type"] not in allowed[kind]:
            raise ValueError("Relationship cannot promote the evidence category")
        if kind == "demonstrated_transmission" and not evidence["transmission_verified"]:
            raise ValueError("Demonstrated relationship requires verified transmission")
        if (
            kind in {"supports", "contradicts"}
            and evidence["term_sense"]
            in {
                TermSense.EARLIER_UNRELATED_USAGE.value,
                TermSense.SONG_TITLE.value,
                TermSense.GENERIC_HYPER_MODIFIER.value,
                TermSense.RETROSPECTIVE_LABEL.value,
            }
            and not evidence["transmission_verified"]
        ):
            raise ValueError("Unrelated usage cannot establish a lineage relationship")
        self.conn.execute(
            "INSERT OR IGNORE INTO relationships (evidence_id,hypothesis_id,kind,rationale) "
            "VALUES (?,?,?,?)",
            (evidence_id, hypothesis_id, kind, rationale),
        )

    def persist_candidate_assessment(
        self,
        candidate_id: int,
        evidence: list[tuple[Evidence, list[tuple[int, str, str]]]],
        leads: list[ResearchLead],
        queries: list[SearchQuery],
        mode: str = "full_document",
    ) -> int:
        if mode not in {"full_document", "relevance_windows"}:
            raise ValueError("Unknown assessment mode")
        candidate = self.conn.execute(
            "SELECT source_id,assessed_at,failure_at FROM candidates WHERE id=?", (candidate_id,)
        ).fetchone()
        if (
            candidate is None
            or candidate["source_id"] is None
            or candidate["assessed_at"]
            or candidate["failure_at"]
        ):
            raise ValueError("Candidate is unavailable for assessment")
        with self.conn:
            for item, links in evidence:
                if item.source_id != candidate["source_id"]:
                    raise ValueError("Evidence source does not match candidate")
                evidence_id = self._add_evidence_tx(item)
                for hypothesis_id, kind, rationale in links:
                    self._add_relationship_tx(evidence_id, hypothesis_id, kind, rationale)
            for lead in leads:
                if lead.source_id != candidate["source_id"]:
                    raise ValueError("Lead source does not match candidate")
                self._add_lead_tx(lead)
            for query in queries:
                self._add_query_tx(query, proposed=True)
            self.conn.execute(
                "UPDATE candidates SET assessed_at=?,fetch_error=NULL,assessment_error=NULL,"
                "assessment_mode=? "
                "WHERE id=?",
                (utc_now(), mode, candidate_id),
            )
        return len(evidence)

    def persist_review(
        self,
        run_id: int,
        question_id: int,
        query_id: int,
        content_json: str,
        queries: list[SearchQuery],
        exhausted_avenues: list[tuple[str, str]] | None = None,
    ) -> None:
        with self.conn:
            reviewed = self.conn.execute(
                "SELECT reviewed_at FROM queries WHERE id=? AND research_question_id=?",
                (query_id, question_id),
            ).fetchone()
            if reviewed is None or reviewed["reviewed_at"] is not None:
                raise ValueError("Review query is missing or already reviewed")
            known_keys = {
                row["avenue"]
                for row in self.conn.execute(
                    "SELECT DISTINCT avenue FROM queries WHERE research_question_id=? "
                    "AND status IN ('active','deferred','executed')",
                    (question_id,),
                )
            }
            review_id = self.conn.execute(
                "INSERT INTO reviews (run_id,research_question_id,created_at,content_json) "
                "VALUES (?,?,?,?)",
                (run_id, question_id, utc_now(), content_json),
            ).lastrowid
            for query in queries:
                if query.research_question_id != question_id:
                    raise ValueError("Review query belongs to another question")
                self._add_query_tx(query, proposed=True)
            self.conn.execute(
                "UPDATE queries SET reviewed_at=? WHERE id=? AND research_question_id=?",
                (utc_now(), query_id, question_id),
            )
            for key, reason in exhausted_avenues or []:
                if not key or key != key.strip() or any(char.isspace() for char in key):
                    rejection = "Not an exact avenue ID"
                elif not reason.strip():
                    rejection = "Retirement reason is empty"
                elif self.conn.execute(
                    "SELECT 1 FROM exhausted_avenues WHERE research_question_id=? AND avenue=?",
                    (question_id, key),
                ).fetchone():
                    rejection = "Avenue already retired"
                elif key not in known_keys:
                    known = self.conn.execute(
                        "SELECT 1 FROM queries WHERE research_question_id=? AND avenue=?",
                        (question_id, key),
                    ).fetchone()
                    rejection = "No reviewable avenue" if known else "Unknown avenue ID"
                elif self.conn.execute(
                    "SELECT 1 FROM queries q WHERE q.research_question_id=? "
                    "AND q.avenue=? AND q.source_target_id IS NOT NULL "
                    "AND NOT EXISTS (SELECT 1 FROM evidence e "
                    "WHERE e.discovered_by_query_id=q.id AND e.primary_source_verified=1) LIMIT 1",
                    (question_id, key),
                ).fetchone():
                    rejection = "Unresolved source target needs primary evidence"
                else:
                    self._retire_avenue_tx(question_id, key, reason)
                    continue
                self.conn.execute(
                    "INSERT INTO review_retirement_rejections "
                    "(review_id,avenue_id,reason,rejection_reason) VALUES (?,?,?,?)",
                    (review_id, key, reason, rejection),
                )
            reviewed = self.conn.execute(
                "SELECT avenue FROM queries WHERE id=? AND research_question_id=?",
                (query_id, question_id),
            ).fetchone()
            if reviewed:
                key = reviewed["avenue"]
                yield_row = self.conn.execute(
                    "SELECT COUNT(DISTINCT q.id) searches,COUNT(DISTINCT e.id) evidence_count "
                    "FROM queries q LEFT JOIN evidence e ON e.discovered_by_query_id=q.id "
                    "WHERE q.research_question_id=? AND q.avenue=? AND q.reviewed_at IS NOT NULL",
                    (question_id, key),
                ).fetchone()
                targeted = self.conn.execute(
                    "SELECT 1 FROM queries WHERE research_question_id=? AND avenue=? "
                    "AND source_target_id IS NOT NULL LIMIT 1",
                    (question_id, key),
                ).fetchone()
                if yield_row["searches"] >= 3 and not yield_row["evidence_count"] and not targeted:
                    self._retire_avenue_tx(
                        question_id, key, "Three reviewed searches yielded no evidence"
                    )

    def record_failed_review(
        self, run_id: int, question_id: int, query_id: int, error: str
    ) -> None:
        diagnostics = [
            json.loads(row["diagnostics_json"])
            for row in self.rows(
                "SELECT diagnostics_json FROM provider_attempts WHERE run_id=? "
                "AND logical_action='review' ORDER BY id",
                (run_id,),
            )
        ]
        self._insert(
            "review_attempts",
            {
                "run_id": run_id,
                "research_question_id": question_id,
                "query_id": query_id,
                "created_at": utc_now(),
                "status": "failed",
                "error": error,
                "diagnostics_json": json.dumps(diagnostics),
            },
        )

    def add_hypothesis_suggestion(
        self,
        run_id: int,
        hypothesis_id: int,
        status: str,
        rationale: str,
        evidence_ids: list[int],
        question_id: int,
    ) -> None:
        if not evidence_ids:
            return
        if not self.rows(
            "SELECT id FROM hypotheses WHERE id=? AND research_question_id=? "
            "AND archived_at IS NULL",
            (hypothesis_id, question_id),
        ):
            return
        placeholders = ",".join("?" for _ in evidence_ids)
        rows = self.rows(
            f"SELECT id FROM evidence WHERE research_question_id=? AND id IN ({placeholders})",
            (question_id, *evidence_ids),
        )
        if len({row["id"] for row in rows}) != len(set(evidence_ids)):
            return
        with self.conn:
            self.conn.execute(
                "INSERT INTO hypothesis_suggestions "
                "(run_id,hypothesis_id,proposed_status,rationale,evidence_ids_json,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, hypothesis_id, status, rationale, json.dumps(evidence_ids), utc_now()),
            )

    def start_run(self, run: ResearchRun) -> int:
        return self._insert("runs", run.model_dump(exclude={"id"}, mode="json"))

    def record_action(
        self,
        run_id: int,
        action: str,
        detail: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost: float = 0.0,
        aggregate_usage: bool = True,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO actions (run_id,occurred_at,action,detail,input_tokens,"
                "output_tokens,estimated_cost) VALUES (?,?,?,?,?,?,?)",
                (run_id, utc_now(), action, detail, input_tokens, output_tokens, estimated_cost),
            )
            if aggregate_usage:
                self.conn.execute(
                    "UPDATE runs SET actions_taken=actions_taken+1,input_tokens=input_tokens+?,"
                    "output_tokens=output_tokens+?,estimated_cost=estimated_cost+? WHERE id=?",
                    (input_tokens, output_tokens, estimated_cost, run_id),
                )
            else:
                self.conn.execute(
                    "UPDATE runs SET actions_taken=actions_taken+1 WHERE id=?", (run_id,)
                )

    def start_provider_attempt(
        self,
        run_id: int,
        action: str,
        model: str,
        local_input_tokens: int,
        provider: str | None = None,
        provider_role: str | None = None,
        attempt_order: int | None = None,
        retry_reason: str | None = None,
    ) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO provider_attempts "
                "(run_id,logical_action,model,started_at,outcome,local_input_tokens,"
                "provider,provider_role,attempt_order,retry_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    action,
                    model,
                    utc_now(),
                    "in_flight",
                    local_input_tokens,
                    provider,
                    provider_role,
                    attempt_order,
                    retry_reason,
                ),
            )
            self.conn.execute(
                "UPDATE runs SET provider_requests=provider_requests+1,"
                "unknown_spend_requests=unknown_spend_requests+1 WHERE id=?",
                (run_id,),
            )
        return int(cursor.lastrowid)

    def finish_provider_attempt(
        self,
        attempt_id: int,
        outcome: str,
        input_tokens: int | None,
        output_tokens: int | None,
        input_rate: float,
        output_rate: float,
        local_output_tokens: int,
        local_usage_multiplier: float,
        diagnostics: dict | None = None,
    ) -> float:
        known = input_tokens is not None and output_tokens is not None
        reported_cost = (
            (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000 if known else None
        )
        with self.conn:
            attempt = self.conn.execute(
                "SELECT run_id,ended_at,local_input_tokens FROM provider_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            if attempt is None or attempt["ended_at"] is not None:
                raise ValueError("Provider attempt already finished or missing")
            local_cost = (
                (attempt["local_input_tokens"] * input_rate + local_output_tokens * output_rate)
                * local_usage_multiplier
                / 1_000_000
            )
            cost = reported_cost if reported_cost is not None else local_cost
            self.conn.execute(
                "UPDATE provider_attempts SET ended_at=?,outcome=?,input_tokens=?,"
                "output_tokens=?,estimated_cost=?,usage_reported=?,local_output_tokens=?,"
                "local_estimated_cost=?,provider_billing_unknown=?,diagnostics_json=?,"
                "http_status=? WHERE id=?",
                (
                    utc_now(),
                    outcome,
                    input_tokens,
                    output_tokens,
                    cost,
                    int(known),
                    local_output_tokens,
                    local_cost,
                    int(not known),
                    json.dumps(diagnostics or {}),
                    (diagnostics or {}).get("http_status"),
                    attempt_id,
                ),
            )
            self.conn.execute(
                "UPDATE runs SET estimated_cost=estimated_cost+?,"
                "provider_reported_cost=provider_reported_cost+?,"
                "locally_estimated_cost=locally_estimated_cost+?,"
                "unknown_spend_requests=unknown_spend_requests-?,"
                "input_tokens=input_tokens+?,output_tokens=output_tokens+? WHERE id=?",
                (
                    cost,
                    reported_cost or 0,
                    0 if known else local_cost,
                    int(known),
                    input_tokens or 0,
                    output_tokens or 0,
                    attempt["run_id"],
                ),
            )
        return float(cost)

    def finish_run(self, run_id: int, status: str, reason: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET ended_at=?,status=?,stop_reason=? WHERE id=?",
                (utc_now(), status, reason, run_id),
            )

    def latest_question(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM questions ORDER BY id DESC LIMIT 1").fetchone()

    def rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()
