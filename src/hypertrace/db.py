"""Small SQLite store. Every fetched revision remains an immutable source row."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Self

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
CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY, research_question_id INTEGER NOT NULL REFERENCES questions(id),
  query TEXT NOT NULL, rationale TEXT NOT NULL, generated_by TEXT NOT NULL,
  created_at TEXT NOT NULL, executed_at TEXT, reviewed_at TEXT,
  UNIQUE(research_question_id, query)
);
CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY, canonical_url TEXT NOT NULL, retrieved_url TEXT NOT NULL,
  title TEXT NOT NULL, author TEXT, publication_date TEXT,
  page_publication_date TEXT, retrieval_date TEXT NOT NULL,
  source_type TEXT NOT NULL, archive_url TEXT, content_hash TEXT NOT NULL,
  document_hash TEXT NOT NULL,
  content TEXT NOT NULL, regions_json TEXT NOT NULL DEFAULT '[]', dating_notes TEXT NOT NULL,
  UNIQUE(canonical_url, document_hash, content_hash)
);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY, query_id INTEGER NOT NULL REFERENCES queries(id),
  url TEXT NOT NULL, title TEXT NOT NULL, source_id INTEGER REFERENCES sources(id),
  assessed_at TEXT, fetch_error TEXT, failure_at TEXT, assessment_error TEXT,
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
  unknown_spend_requests INTEGER NOT NULL DEFAULT 0
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
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        additions = {
            "sources": {
                "page_publication_date": "TEXT",
                "regions_json": "TEXT NOT NULL DEFAULT '[]'",
            },
            "candidates": {
                "failure_at": "TEXT",
                "assessment_error": "TEXT",
                "assessment_mode": "TEXT",
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
            },
            "provider_attempts": {"diagnostics_json": "TEXT NOT NULL DEFAULT '{}'"},
        }
        legacy_evidence = "quote_region" not in {
            row["name"] for row in self.conn.execute("PRAGMA table_info(evidence)")
        }
        with self.conn:
            for table, columns in additions.items():
                existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                for name, declaration in columns.items():
                    if name not in existing:
                        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
            self.conn.execute(
                "UPDATE sources SET page_publication_date=publication_date "
                "WHERE page_publication_date IS NULL AND publication_date IS NOT NULL"
            )
            self.conn.execute(
                "UPDATE candidates SET failure_at=NULL,fetch_error=NULL "
                "WHERE source_id IS NOT NULL AND assessed_at IS NULL "
                "AND fetch_error='Fetched text exceeds the 16,000-character assessment window'"
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

    def add_query(self, query: SearchQuery) -> int:
        with self.conn:
            return self._add_query_tx(query)

    def _add_query_tx(self, query: SearchQuery) -> int:
        values = query.model_dump(exclude={"id"}, mode="json")
        self.conn.execute(
            "INSERT OR IGNORE INTO queries "
            "(research_question_id,query,rationale,generated_by,created_at,executed_at) "
            "VALUES (?,?,?,?,?,?)",
            tuple(values.values()),
        )
        row = self.conn.execute(
            "SELECT id FROM queries WHERE research_question_id=? AND query=?",
            (query.research_question_id, query.query),
        ).fetchone()
        return int(row["id"])

    def mark_query_executed(self, query_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE queries SET executed_at=? WHERE id=?", (utc_now(), query_id))

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

    def persist_plan(self, question_id: int, queries: list[SearchQuery]) -> None:
        with self.conn:
            for query in queries:
                if query.research_question_id != question_id:
                    raise ValueError("Planned query belongs to another question")
                self._add_query_tx(query)

    def persist_search_results(
        self, question_id: int, query_id: int, results: list[tuple[str, str]]
    ) -> None:
        with self.conn:
            query = self.conn.execute(
                "SELECT research_question_id,executed_at FROM queries WHERE id=?", (query_id,)
            ).fetchone()
            if (
                query is None
                or query["research_question_id"] != question_id
                or query["executed_at"]
            ):
                raise ValueError("Search query is unavailable")
            for url, title in results:
                self.conn.execute(
                    "INSERT OR IGNORE INTO candidates (query_id,url,title) VALUES (?,?,?)",
                    (query_id, url, title),
                )
            self.conn.execute("UPDATE queries SET executed_at=? WHERE id=?", (utc_now(), query_id))

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
                "UPDATE candidates SET assessed_at=?,fetch_error=NULL,assessment_error=NULL "
                "WHERE id=?",
                (utc_now(), candidate_id),
            )

    def record_fetch_failure(self, candidate_id: int, reason: str, *, retryable: bool) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE candidates SET fetch_error=?,failure_at=? WHERE id=?",
                (reason, None if retryable else utc_now(), candidate_id),
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
                "UPDATE candidates SET source_id=? WHERE id=?", (source_id, candidate_id)
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
        values = source.model_dump(exclude={"id", "content_regions"}, mode="json")
        values["regions_json"] = json.dumps(
            [region.model_dump(mode="json") for region in source.content_regions]
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO sources "
            "(canonical_url,retrieved_url,title,author,page_publication_date,retrieval_date,"
            "source_type,archive_url,content_hash,document_hash,content,dating_notes,regions_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(values.values()),
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
            "SELECT research_question_id FROM hypotheses WHERE id=?", (hypothesis_id,)
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
                self._add_query_tx(query)
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
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO reviews (run_id,research_question_id,created_at,content_json) "
                "VALUES (?,?,?,?)",
                (run_id, question_id, utc_now(), content_json),
            )
            for query in queries:
                if query.research_question_id != question_id:
                    raise ValueError("Review query belongs to another question")
                self._add_query_tx(query)
            self.conn.execute(
                "UPDATE queries SET reviewed_at=? WHERE id=? AND research_question_id=?",
                (utc_now(), query_id, question_id),
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

    def start_provider_attempt(self, run_id: int, action: str, model: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO provider_attempts (run_id,logical_action,model,started_at,outcome) "
                "VALUES (?,?,?,?,?)",
                (run_id, action, model, utc_now(), "in_flight"),
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
        diagnostics: dict | None = None,
    ) -> float:
        known = input_tokens is not None and output_tokens is not None
        cost = (
            (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000 if known else None
        )
        with self.conn:
            attempt = self.conn.execute(
                "SELECT run_id,ended_at FROM provider_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None or attempt["ended_at"] is not None:
                raise ValueError("Provider attempt already finished or missing")
            self.conn.execute(
                "UPDATE provider_attempts SET ended_at=?,outcome=?,input_tokens=?,"
                "output_tokens=?,estimated_cost=?,usage_reported=?,diagnostics_json=? WHERE id=?",
                (
                    utc_now(),
                    outcome,
                    input_tokens,
                    output_tokens,
                    cost,
                    int(known),
                    json.dumps(diagnostics or {}),
                    attempt_id,
                ),
            )
            if known:
                self.conn.execute(
                    "UPDATE runs SET input_tokens=input_tokens+?,output_tokens=output_tokens+?,"
                    "estimated_cost=estimated_cost+?,"
                    "unknown_spend_requests=unknown_spend_requests-1 WHERE id=?",
                    (input_tokens, output_tokens, cost, attempt["run_id"]),
                )
        return float(cost or 0)

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
