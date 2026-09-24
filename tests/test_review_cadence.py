"""Persisted review cadence at the runner/database boundary."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import replace

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.base import LLMResult
from hypertrace.models import (
    AdversarialReview,
    ContentRegion,
    Evidence,
    Hypothesis,
    PlannedQuery,
    QuoteRegion,
    ResearchQuestion,
    ResearchRun,
    SearchQuery,
    Source,
)
from hypertrace.research import Limits, Researcher


class Reviewer:
    def __init__(self, queries: list[PlannedQuery] | None = None):
        self.calls = 0
        self.queries = queries or []

    async def complete(self, model, system, user, schema):
        assert schema is AdversarialReview
        self.calls += 1
        return LLMResult(value=AdversarialReview(next_queries=self.queries))


def _config(tmp_path, *, threshold=25, query_limit=3):
    return Config(
        db_path=tmp_path / "cadence.db",
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
        review_action_threshold=threshold,
        review_query_limit=query_limit,
    )


def _runner(db, config, question_id, reviewer):
    researcher = Researcher(
        db,
        reviewer,
        object(),
        config,
        question_id,
        Limits(max_actions=100, max_minutes=10),
    )
    researcher.deadline = time.monotonic() + 600
    researcher.run_id = db.start_run(
        ResearchRun(research_question_id=question_id, model="research", provider="test")
    )
    return researcher


def _prime_review(runner):
    config = runner.config
    runner.config = replace(config, review_action_threshold=1)
    assert asyncio.run(runner._maybe_review())
    runner.config = config


def _question(db):
    question_id = db.add_question(ResearchQuestion(question="Where did the term originate?"))
    query_id = db.add_query(
        SearchQuery(research_question_id=question_id, query="first archival source")
    )
    db.mark_query_executed(query_id)
    return question_id


def _next_query(db, question_id):
    query_id = db.add_query(
        SearchQuery(research_question_id=question_id, query="second independent archive")
    )
    db.mark_query_executed(query_id)
    return query_id


def _evidence(db, question_id, query_id, *, primary):
    content = "An article uses the term in its body."
    source_id = db.add_source(
        Source(
            canonical_url="https://example.org/article",
            retrieved_url="https://example.org/article",
            content=content,
            content_regions=[
                ContentRegion(start=0, end=len(content), region=QuoteRegion.ARTICLE_BODY)
            ],
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )
    )
    return db.add_evidence(
        Evidence(
            source_id=source_id,
            research_question_id=question_id,
            exact_quote="uses the term",
            normalized_claim="uses the term",
            evidence_type="observed_usage" if primary else "interpretive_context",
            primary_source_verified=primary,
            discovered_by_query_id=query_id,
        )
    )


def test_one_search_waits_but_threshold_reviews(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path, threshold=3), question_id, reviewer)
        _prime_review(runner)
        _next_query(db, question_id)
        db.record_action(runner.run_id, "search", "query_id=2 results=0")
        assert not asyncio.run(runner._maybe_review())
        assert reviewer.calls == 1
        db.record_action(runner.run_id, "fetch", "source_id=5 url=https://example.org")
        db.record_action(runner.run_id, "assess", '{"model":"research","output":{}}')
        assert asyncio.run(runner._maybe_review())
        assert reviewer.calls == 2
        decisions = db.rows(
            "SELECT decision,reason,meaningful_actions FROM review_cadence_events ORDER BY id"
        )
        assert [tuple(row) for row in decisions] == [
            ("triggered", "action_threshold", 1),
            ("skipped", "below_action_threshold", 1),
            ("triggered", "action_threshold", 3),
        ]


def test_primary_evidence_and_target_resolution_trigger_early_review(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        db.install_source_targets(question_id, [("pc-music-2014", "Find source")], [])
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        query_id = db.add_query(
            SearchQuery(
                research_question_id=question_id,
                query="targeted primary document",
                source_target="pc-music-2014",
            )
        )
        db.mark_query_executed(query_id)
        evidence_id = _evidence(db, question_id, query_id, primary=True)
        assert asyncio.run(runner._maybe_review())
        decision = db.rows(
            "SELECT reason,meaningful_actions,evidence_ids_json,target_keys_json "
            "FROM review_cadence_events ORDER BY id DESC LIMIT 1"
        )[0]
        assert decision["reason"] == "source_target_changed"
        assert decision["meaningful_actions"] == 3
        assert decision["evidence_ids_json"] == f"[{evidence_id}]"
        assert decision["target_keys_json"] == '["pc-music-2014"]'


def test_primary_evidence_without_target_triggers_early_review(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        query_id = _next_query(db, question_id)
        evidence_id = _evidence(db, question_id, query_id, primary=True)
        assert asyncio.run(runner._maybe_review())
        decision = db.rows(
            "SELECT reason,evidence_ids_json,target_keys_json "
            "FROM review_cadence_events ORDER BY id DESC LIMIT 1"
        )[0]
        assert tuple(decision) == ("important_evidence", f"[{evidence_id}]", "[]")


def test_new_contradiction_triggers_early_review(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        hypothesis_id = db.add_hypothesis(
            Hypothesis(research_question_id=question_id, statement="A source caused the term.")
        )
        old_evidence_id = _evidence(db, question_id, 1, primary=False)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        _next_query(db, question_id)
        db.add_relationship(old_evidence_id, hypothesis_id, "contradicts", "Direct counterexample")
        assert asyncio.run(runner._maybe_review())
        decision = db.rows(
            "SELECT reason,evidence_ids_json FROM review_cadence_events ORDER BY id DESC LIMIT 1"
        )[0]
        assert tuple(decision) == ("important_evidence", f"[{old_evidence_id}]")


def test_secondary_activity_waits_and_final_review_needs_material_evidence(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        query_id = _next_query(db, question_id)
        _evidence(db, question_id, query_id, primary=False)
        db.record_action(runner.run_id, "assess", '{"model":"research","output":{}}')
        assert not asyncio.run(runner._maybe_review())
        assert not asyncio.run(runner._maybe_review(final=True))
        assert reviewer.calls == 1
        assert (
            db.rows("SELECT reason FROM review_cadence_events ORDER BY id DESC LIMIT 1")[0][0]
            == "no_material_change"
        )


def test_final_review_reconciles_new_primary_evidence(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        query_id = _next_query(db, question_id)
        evidence_id = _evidence(db, question_id, query_id, primary=True)
        assert asyncio.run(runner._maybe_review(final=True))
        assert reviewer.calls == 2
        decision = db.rows(
            "SELECT reason,evidence_ids_json FROM review_cadence_events ORDER BY id DESC LIMIT 1"
        )[0]
        assert tuple(decision) == ("final_material_evidence", f"[{evidence_id}]")


def test_final_review_can_revisit_reviewed_query_for_new_evidence(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path), question_id, reviewer)
        _prime_review(runner)
        _evidence(db, question_id, 1, primary=True)
        assert db.review_due(question_id) is None
        assert asyncio.run(runner._maybe_review(final=True))
        assert reviewer.calls == 2
        assert (
            len(db.rows("SELECT id FROM reviews WHERE research_question_id=?", (question_id,))) == 2
        )


def test_review_query_limit_keeps_only_high_value_proposals(tmp_path):
    proposals = [
        PlannedQuery(
            query=text,
            rationale="Independent source",
            gap="Missing primary source",
            novelty="Different archive",
            information_value=value,
            admission_basis="primary_source",
        )
        for text, value in [
            ("first high-value search", "high"),
            ("generic secondary recap", "medium"),
            ("second high-value search", "high"),
        ]
    ]
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer(proposals)
        config = replace(_config(tmp_path), review_query_limit=1)
        runner = _runner(db, config, question_id, reviewer)
        _prime_review(runner)
        added = db.rows(
            "SELECT query FROM queries WHERE research_question_id=? AND id>1 ORDER BY id",
            (question_id,),
        )
        assert [row[0] for row in added] == ["first high-value search"]


def test_resume_uses_persisted_action_count(tmp_path):
    config = _config(tmp_path, threshold=3)
    reviewer = Reviewer()
    with Database(config.db_path) as db:
        question_id = _question(db)
        runner = _runner(db, config, question_id, reviewer)
        _prime_review(runner)
        _next_query(db, question_id)
        db.record_action(runner.run_id, "search", "query_id=2 results=0")
        db.record_action(runner.run_id, "fetch", "source_id=5 url=https://example.org")
        assert not asyncio.run(runner._maybe_review())
    with Database(config.db_path) as db:
        assert db.review_cadence_state(question_id)["meaningful_actions"] == 2
        runner = _runner(db, config, question_id, reviewer)
        db.record_action(runner.run_id, "assess", '{"model":"research","output":{}}')
        assert asyncio.run(runner._maybe_review())
        assert reviewer.calls == 2


def test_review_waits_when_remaining_time_is_too_short(tmp_path):
    with Database(tmp_path / "cadence.db") as db:
        question_id = _question(db)
        reviewer = Reviewer()
        runner = _runner(db, _config(tmp_path, threshold=2), question_id, reviewer)
        _prime_review(runner)
        _next_query(db, question_id)
        db.record_action(runner.run_id, "fetch", "source_id=5 url=https://example.org")
        runner.deadline = time.monotonic() + 60
        assert not asyncio.run(runner._maybe_review())
        assert reviewer.calls == 1
        assert (
            db.rows("SELECT reason FROM review_cadence_events ORDER BY id DESC LIMIT 1")[0][0]
            == "insufficient_budget"
        )


def test_openrouter_primary_configuration_keeps_research_provider(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.cheaperinference.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "research-key")
    monkeypatch.setenv("RESEARCH_MODEL", "glm-5.3-flash")
    monkeypatch.setenv("REVIEW_PRIMARY_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("REVIEW_PRIMARY_API_KEY", "review-key")
    monkeypatch.setenv("REVIEW_MODEL", "glm-5.3")
    monkeypatch.delenv("REVIEW_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("REVIEW_FALLBACK_API_KEY", raising=False)
    config = Config.from_env()
    assert (config.llm_base_url, config.research_model) == (
        "https://api.cheaperinference.com/v1",
        "glm-5.3-flash",
    )
    assert (config.review_primary_base_url, config.review_model) == (
        "https://openrouter.ai/api/v1",
        "glm-5.3",
    )
    assert config.review_fallback_base_url is None
