"""Command-line interface."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from hypertrace.config import Config
from hypertrace.db import Database
from hypertrace.llm.openai_compatible import OpenAICompatibleLLM
from hypertrace.models import Hypothesis, ResearchQuestion
from hypertrace.planner import seed_motivating_case
from hypertrace.reports import markdown_report
from hypertrace.research import Limits, Researcher
from hypertrace.retrieval.web import BraveWeb


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hypertrace")
    parser.add_argument("--db", type=Path, help="SQLite database path (default HYPERTRACE_DB)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create database and seed motivating case")
    create = sub.add_parser("research", help="Add a new research question")
    create.add_argument("question")
    create.add_argument("--hypothesis", action="append", default=[])
    run = sub.add_parser("run", help="Run bounded research")
    run.add_argument("--question-id", type=int)
    run.add_argument("--max-actions", type=int, default=30)
    run.add_argument("--max-cost", type=float)
    run.add_argument("--max-minutes", type=float, default=20)
    run.add_argument("--min-yield", type=float)
    run.add_argument("--model")
    activate = sub.add_parser("activate-query", help="Reconsider a deferred query")
    activate.add_argument("query_id", type=int)
    activate.add_argument("--question-id", type=int)
    for name in ("status", "evidence", "hypotheses", "report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--question-id", type=int)
        if name == "report":
            cmd.add_argument("--output", type=Path)
    return parser


def _question_id(db: Database, requested: int | None) -> int:
    if requested is not None:
        if not db.rows("SELECT id FROM questions WHERE id=?", (requested,)):
            raise ValueError(f"Unknown question ID {requested}")
        return requested
    latest = db.latest_question()
    if latest is None:
        raise ValueError("No research question; use `hypertrace init` or `hypertrace research`")
    return int(latest["id"])


async def _run(db: Database, config: Config, args: argparse.Namespace, question_id: int) -> int:
    config.require_online(args.max_cost)
    llm = OpenAICompatibleLLM(config.llm_base_url, config.llm_api_key, json_mode=config.json_mode)
    web = BraveWeb(config.brave_api_key)
    try:
        limits = Limits(
            max_actions=args.max_actions,
            max_cost=args.max_cost,
            max_minutes=args.max_minutes,
            min_yield=args.min_yield if args.min_yield is not None else config.min_yield,
        )
        return await Researcher(db, llm, web, config, question_id, limits, args.model).run()
    finally:
        await llm.aclose()
        await web.aclose()


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = Config.from_env()
    with Database(args.db or config.db_path) as db:
        if args.command == "init":
            qid = seed_motivating_case(db)
            print(f"Initialized {db.path}; seeded question {qid}")
            return
        if args.command == "research":
            qid = db.add_question(ResearchQuestion(question=args.question))
            for statement in args.hypothesis:
                db.add_hypothesis(
                    Hypothesis(
                        research_question_id=qid,
                        statement=statement,
                        rationale="User-supplied starting hypothesis.",
                    )
                )
            print(f"Created question {qid}")
            return
        qid = _question_id(db, args.question_id)
        if args.command == "activate-query":
            activated = db.activate_deferred_query(qid, args.query_id)
            print(
                "Activated deferred query" if activated else "Active frontier has higher priorities"
            )
            return
        if args.command == "run":
            run_id = asyncio.run(_run(db, config, args, qid))
            row = db.rows("SELECT * FROM runs WHERE id=?", (run_id,))[0]
            print(
                f"Run {run_id}: {row['status']} ({row['stop_reason']}), "
                f"{row['actions_taken']} logical actions, "
                f"{row['provider_requests']} provider requests, "
                f"known estimated cost ${row['estimated_cost']:.4f}, "
                f"{row['unknown_spend_requests']} requests with unknown spend"
            )
        elif args.command == "status":
            q = db.rows("SELECT * FROM questions WHERE id=?", (qid,))[0]
            counts = db.rows(
                "SELECT (SELECT COUNT(*) FROM hypotheses "
                "WHERE research_question_id=? AND archived_at IS NULL) h,"
                "(SELECT COUNT(*) FROM evidence WHERE research_question_id=?) e,"
                "(SELECT COUNT(*) FROM queries WHERE research_question_id=? AND status='active') active,"
                "(SELECT COUNT(*) FROM queries WHERE research_question_id=? AND status='deferred') deferred,"
                "(SELECT COUNT(*) FROM queries WHERE research_question_id=? AND status='exhausted') exhausted,"
                "(SELECT COUNT(*) FROM queries WHERE research_question_id=? AND status='rejected_duplicate') duplicates,"
                "(SELECT COUNT(*) FROM candidates c JOIN queries q ON q.id=c.query_id "
                "WHERE q.research_question_id=? AND c.assessed_at IS NULL "
                "AND c.failure_at IS NULL) c",
                (qid, qid, qid, qid, qid, qid, qid),
            )[0]
            print(f"Question {qid} [{q['status']}]: {q['question']}")
            print(
                f"{counts['h']} hypotheses; {counts['e']} evidence; "
                f"{counts['active']} active queries; {counts['deferred']} deferred leads; "
                f"{counts['exhausted']} exhausted queries; "
                f"{counts['duplicates']} rejected duplicates; {counts['c']} pending pages"
            )
            runs = db.rows(
                "SELECT * FROM runs WHERE research_question_id=? ORDER BY id DESC LIMIT 1", (qid,)
            )
            if runs:
                r = runs[0]
                print(
                    f"Last run {r['id']}: {r['status']} ({r['stop_reason']}); "
                    f"{r['provider_requests']} provider requests; "
                    f"{r['unknown_spend_requests']} with unknown spend"
                )
        elif args.command == "evidence":
            for row in db.rows(
                "SELECT e.id,e.evidence_type,e.exact_quote,s.retrieved_url "
                "FROM evidence e JOIN sources s ON s.id=e.source_id "
                "WHERE e.research_question_id=? ORDER BY e.id",
                (qid,),
            ):
                print(
                    f"E{row['id']} [{row['evidence_type']}] {row['exact_quote']}\n  {row['retrieved_url']}"
                )
        elif args.command == "hypotheses":
            for row in db.rows(
                "SELECT * FROM hypotheses WHERE research_question_id=? "
                "AND archived_at IS NULL ORDER BY id",
                (qid,),
            ):
                print(f"H{row['id']} [{row['status']}] {row['statement']}\n  {row['rationale']}")
        elif args.command == "report":
            report = markdown_report(db, qid)
            if args.output:
                args.output.write_text(report, encoding="utf-8")
                print(args.output)
            else:
                print(report, end="")


if __name__ == "__main__":
    main()
