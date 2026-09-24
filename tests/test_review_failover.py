"""Review provider failover at the persistence boundary."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.openai_compatible import OpenAICompatibleLLM
from hypertrace.models import ResearchQuestion, SearchQuery
from hypertrace.research import Limits, Researcher

REVIEW = {
    "choices": [{"finish_reason": "stop", "message": {"content": '{"overclaims": []}'}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


def _config(tmp_path) -> Config:
    return Config(
        db_path=tmp_path / "review.db",
        llm_base_url="https://research.example/v1",
        llm_api_key="research-key",
        router_model="research",
        research_model="research",
        review_model="glm-5.3",
        brave_api_key="search-key",
        input_cost_per_million=1,
        output_cost_per_million=1,
        min_yield=0,
        json_mode=True,
        review_primary_base_url="https://primary.example/v1",
        review_primary_api_key="primary-key",
        review_primary_input_cost_per_million=2,
        review_primary_output_cost_per_million=3,
        review_fallback_base_url="https://fallback.example/v1",
        review_fallback_api_key="fallback-key",
        review_fallback_model="z-ai/glm-5.3",
        review_fallback_input_cost_per_million=7,
        review_fallback_output_cost_per_million=11,
        review_action_threshold=1,
    )


async def _run(db, config, primary, fallback, question_id):
    researcher = Researcher(
        db,
        primary,
        object(),
        config,
        question_id,
        Limits(max_actions=1, max_minutes=1),
        review_primary=primary,
        review_fallback=fallback,
    )
    return await researcher.run()


def _question(db):
    question_id = db.add_question(ResearchQuestion(question="Where did the term originate?"))
    query_id = db.add_query(SearchQuery(research_question_id=question_id, query="term history"))
    db.mark_query_executed(query_id)
    return question_id, query_id


def _attempts(db, run_id):
    return db.rows(
        "SELECT provider,provider_role,model,attempt_order,retry_reason,http_status,"
        "input_tokens,output_tokens,estimated_cost,provider_billing_unknown "
        "FROM provider_attempts WHERE run_id=? ORDER BY id",
        (run_id,),
    )


def test_primary_review_succeeds_without_fallback(tmp_path):
    calls = []

    def primary_handler(request):
        calls.append(request)
        return httpx.Response(200, json=REVIEW)

    def fallback_handler(_):
        raise AssertionError("Fallback must remain unused")

    async def exercise():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)) as primary_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, query_id = _question(db)
                run_id = await _run(db, config, primary, fallback, question_id)
                attempts = _attempts(db, run_id)
                assert len(calls) == len(attempts) == 1
                assert tuple(attempts[0][:6]) == (
                    "primary.example",
                    "primary",
                    "glm-5.3",
                    1,
                    "initial",
                    200,
                )
                assert attempts[0]["estimated_cost"] == 35 / 1_000_000
                assert len(db.rows("SELECT id FROM reviews WHERE run_id=?", (run_id,))) == 1
                assert db.review_due(question_id) is None
                assert db.rows("SELECT reviewed_at FROM queries WHERE id=?", (query_id,))[0][0]

    asyncio.run(exercise())


def test_primary_429_honors_retry_after_then_succeeds(tmp_path, monkeypatch):
    calls = 0
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("hypertrace.llm.openai_compatible.asyncio.sleep", fake_sleep)

    def primary_handler(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429, headers={"Retry-After": "3"}, json={"error": {"code": "rate_limit_exceeded"}}
            )
        return httpx.Response(200, json=REVIEW)

    async def exercise():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)) as primary_client,
            httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: (_ for _ in ()).throw(AssertionError("fallback used"))
                )
            ) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, _ = _question(db)
                run_id = await _run(db, config, primary, fallback, question_id)
                attempts = _attempts(db, run_id)
                assert delays == [3.0]
                assert [(r["http_status"], r["retry_reason"]) for r in attempts] == [
                    (429, "initial"),
                    (200, "retry_after_rate_limit"),
                ]
                assert attempts[0]["provider_billing_unknown"] == 1
                assert len(db.rows("SELECT id FROM reviews")) == 1

    asyncio.run(exercise())


def test_two_primary_429s_fail_over_same_review_and_bill_actual_provider(tmp_path, monkeypatch):
    requests = []
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("hypertrace.llm.openai_compatible.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("hypertrace.llm.openai_compatible.random.uniform", lambda *_: 0)

    def primary_handler(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    def fallback_handler(request):
        requests.append(request)
        return httpx.Response(200, json=REVIEW)

    async def exercise():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)) as primary_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, query_id = _question(db)
                run_id = await _run(db, config, primary, fallback, question_id)
                attempts = _attempts(db, run_id)
                assert delays == [1.0]
                assert [r["provider_role"] for r in attempts] == ["primary", "primary", "fallback"]
                assert [r["attempt_order"] for r in attempts] == [1, 2, 3]
                assert [r["retry_reason"] for r in attempts] == [
                    "initial",
                    "retry_after_rate_limit",
                    "failover_after_rate_limit",
                ]
                assert [r["http_status"] for r in attempts] == [429, 429, 200]
                assert [r["model"] for r in attempts] == ["glm-5.3", "glm-5.3", "z-ai/glm-5.3"]
                bodies = [json.loads(request.content) for request in requests]
                assert bodies[0]["messages"] == bodies[1]["messages"] == bodies[2]["messages"]
                assert requests[0].headers["authorization"] == "Bearer primary-key"
                assert requests[2].headers["authorization"] == "Bearer fallback-key"
                assert attempts[2]["estimated_cost"] == 125 / 1_000_000
                assert attempts[0]["estimated_cost"] > 0
                assert attempts[1]["estimated_cost"] == attempts[0]["estimated_cost"]
                review_action = db.rows(
                    "SELECT detail FROM actions WHERE run_id=? AND action='review'", (run_id,)
                )[0]
                assert json.loads(review_action["detail"])["model"] == "z-ai/glm-5.3"
                assert db.rows("SELECT reviewed_at FROM queries WHERE id=?", (query_id,))[0][0]
                assert db.review_due(question_id) is None
                assert len(db.rows("SELECT id FROM reviews")) == 1
                run = db.rows("SELECT * FROM runs WHERE id=?", (run_id,))[0]
                assert run["provider_reported_cost"] == 125 / 1_000_000
                assert run["unknown_spend_requests"] == 2
                assert run["locally_estimated_cost"] == sum(
                    row[0]
                    for row in db.rows(
                        "SELECT local_estimated_cost FROM provider_attempts "
                        "WHERE run_id=? AND provider_billing_unknown=1",
                        (run_id,),
                    )
                )

    asyncio.run(exercise())


def test_fallback_failure_leaves_review_due_and_resume_commits_once(tmp_path, monkeypatch):
    async def fake_sleep(_):
        pass

    monkeypatch.setattr("hypertrace.llm.openai_compatible.asyncio.sleep", fake_sleep)
    calls = 0

    def primary_handler(_):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
        return httpx.Response(200, json=REVIEW)

    def fallback_handler(_):
        return httpx.Response(503, json={"error": {"code": "unavailable"}})

    async def exercise():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)) as primary_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client, max_retries=0
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, query_id = _question(db)
                first = await _run(db, config, primary, fallback, question_id)
                assert (
                    db.rows("SELECT stop_reason FROM runs WHERE id=?", (first,))[0][0]
                    == "review_failed"
                )
                assert db.review_due(question_id) == query_id
                assert len(db.rows("SELECT id FROM reviews")) == 0
                assert [r["http_status"] for r in _attempts(db, first)] == [429, 429, 503]
                second = await _run(db, config, primary, fallback, question_id)
                assert len(_attempts(db, second)) == 1
                assert db.review_due(question_id) is None
                assert len(db.rows("SELECT id FROM reviews")) == 1
                assert db.rows("SELECT reviewed_at FROM queries WHERE id=?", (query_id,))[0][0]

    asyncio.run(exercise())


def test_primary_403_does_not_fail_over(tmp_path):
    fallback_calls = 0

    def fallback_handler(_):
        nonlocal fallback_calls
        fallback_calls += 1
        return httpx.Response(200, json=REVIEW)

    async def exercise():
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(403))
            ) as primary_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, query_id = _question(db)
                run_id = await _run(db, config, primary, fallback, question_id)
                assert fallback_calls == 0
                assert [r["http_status"] for r in _attempts(db, run_id)] == [403]
                assert db.review_due(question_id) == query_id
                assert len(db.rows("SELECT id FROM reviews")) == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("content", ["not json", '{"next_queries": [{"query": "term"}]}'])
def test_invalid_primary_review_does_not_fail_over(tmp_path, content):
    fallback_calls = 0

    def fallback_handler(_):
        nonlocal fallback_calls
        fallback_calls += 1
        return httpx.Response(200, json=REVIEW)

    def primary_handler(_):
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def exercise():
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(primary_handler)) as primary_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as fallback_client,
        ):
            primary = OpenAICompatibleLLM(
                "https://primary.example/v1", "primary-key", client=primary_client, max_retries=0
            )
            fallback = OpenAICompatibleLLM(
                "https://fallback.example/v1", "fallback-key", client=fallback_client
            )
            with Database(tmp_path / "review.db") as db:
                config = _config(tmp_path)
                question_id, query_id = _question(db)
                run_id = await _run(db, config, primary, fallback, question_id)
                assert fallback_calls == 0
                assert [r["provider_role"] for r in _attempts(db, run_id)] == ["primary", "primary"]
                assert db.review_due(question_id) == query_id
                assert len(db.rows("SELECT id FROM reviews")) == 0

    asyncio.run(exercise())
