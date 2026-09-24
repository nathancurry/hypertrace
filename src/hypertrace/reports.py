"""Markdown reports rendered solely from persisted records."""

from __future__ import annotations

import json

from hypertrace.db import Database


def _cell(value: object) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def _cite(url: str, title: str, source_id: int) -> str:
    label = title or f"Source {source_id}"
    return f"[{_cell(label)}]({url})"


def markdown_report(db: Database, question_id: int) -> str:
    questions = db.rows("SELECT * FROM questions WHERE id=?", (question_id,))
    if not questions:
        raise ValueError(f"Unknown question ID {question_id}")
    question = questions[0]
    latest_run = db.rows(
        "SELECT status,stop_reason,actions_taken,provider_requests,unknown_spend_requests "
        "FROM runs WHERE research_question_id=? ORDER BY id DESC LIMIT 1",
        (question_id,),
    )
    hypotheses = db.rows(
        "SELECT * FROM hypotheses WHERE research_question_id=? AND archived_at IS NULL ORDER BY id",
        (question_id,),
    )
    evidence = db.rows(
        "SELECT e.*,s.canonical_url,s.retrieved_url,s.title,s.page_publication_date,s.dating_notes,"
        "s.author,s.retrieval_date,s.content_hash,s.document_hash,s.source_type FROM evidence e "
        "JOIN sources s ON s.id=e.source_id WHERE e.research_question_id=? "
        "ORDER BY e.quote_verified_date IS NULL,e.quote_verified_date,e.id",
        (question_id,),
    )
    source_captures = db.rows(
        "SELECT s.id source_id,s.content_hash,s.retrieved_url,s.canonical_url,"
        "s.page_publication_date,s.document_hash FROM sources s WHERE "
        "EXISTS (SELECT 1 FROM evidence e WHERE e.source_id=s.id AND e.research_question_id=?) "
        "OR EXISTS (SELECT 1 FROM candidates c JOIN queries q ON q.id=c.query_id "
        "WHERE c.source_id=s.id AND q.research_question_id=?) ORDER BY s.id",
        (question_id, question_id),
    )
    parent = {row["source_id"]: row["source_id"] for row in source_captures}

    def root(source_id: int) -> int:
        while parent[source_id] != source_id:
            source_id = parent[source_id]
        return source_id

    seen_hash: dict[str, int] = {}
    seen_url: dict[str, int] = {}
    for row in source_captures:
        source_id = row["source_id"]
        for seen, key in ((seen_hash, row["content_hash"]), (seen_url, row["retrieved_url"])):
            if key in seen:
                parent[root(source_id)] = root(seen[key])
            else:
                seen[key] = source_id
    group_by_source = {source_id: root(source_id) for source_id in parent}
    displayed = []
    seen_quotes: set[tuple] = set()
    for row in evidence:
        key = (
            group_by_source[row["source_id"]],
            row["exact_quote"],
            row["evidence_type"],
            row["quote_verified_date"],
        )
        if key not in seen_quotes:
            displayed.append(row)
            seen_quotes.add(key)
    relationships = db.rows(
        "SELECT r.* FROM relationships r JOIN evidence e ON e.id=r.evidence_id "
        "JOIN hypotheses h ON h.id=r.hypothesis_id "
        "WHERE e.research_question_id=? AND h.archived_at IS NULL",
        (question_id,),
    )
    links: dict[int, list] = {}
    for row in relationships:
        links.setdefault(row["hypothesis_id"], []).append(row)
    by_id = {row["id"]: row for row in evidence}
    reviews = db.rows(
        "SELECT content_json FROM reviews WHERE research_question_id=? ORDER BY id DESC LIMIT 3",
        (question_id,),
    )
    suggestions = db.rows(
        "SELECT h.id hypothesis_id,s.proposed_status,s.rationale,s.evidence_ids_json "
        "FROM hypothesis_suggestions s JOIN hypotheses h ON h.id=s.hypothesis_id "
        "WHERE h.research_question_id=? AND h.archived_at IS NULL "
        "ORDER BY s.id DESC LIMIT 20",
        (question_id,),
    )
    pending = db.rows(
        "SELECT id,query,rationale,gap,information_value FROM queries WHERE research_question_id=? "
        "AND status='active' ORDER BY priority DESC,id LIMIT 15",
        (question_id,),
    )
    frontier_counts = {
        row["status"]: row["count"]
        for row in db.rows(
            "SELECT status,COUNT(*) count FROM queries WHERE research_question_id=? "
            "AND executed_at IS NULL GROUP BY status",
            (question_id,),
        )
    }
    deferred = db.rows(
        "SELECT id,query,rationale FROM queries WHERE research_question_id=? AND status='deferred' "
        "ORDER BY priority DESC,id LIMIT 10",
        (question_id,),
    )
    candidates = db.rows(
        "SELECT c.url,c.title FROM candidates c JOIN queries q ON q.id=c.query_id "
        "WHERE q.research_question_id=? AND c.assessed_at IS NULL "
        "AND c.failure_at IS NULL ORDER BY c.id LIMIT 20",
        (question_id,),
    )
    failures = db.rows(
        "SELECT c.url,c.fetch_error,c.fetch_retryable,c.fetch_status,c.fetch_final_url,"
        "c.fetch_redirects_json,q.query FROM candidates c "
        "JOIN queries q ON q.id=c.query_id "
        "WHERE q.research_question_id=? AND c.fetch_error IS NOT NULL ORDER BY c.id",
        (question_id,),
    )
    leads = db.rows(
        "SELECT l.kind,l.value,l.rationale,s.retrieved_url,s.title,s.id source_id "
        "FROM leads l JOIN sources s ON s.id=l.source_id "
        "WHERE l.research_question_id=? ORDER BY l.id DESC LIMIT 30",
        (question_id,),
    )
    notes = db.rows(
        "SELECT former_hypothesis_id,kind,statement FROM research_notes "
        "WHERE research_question_id=? ORDER BY former_hypothesis_id",
        (question_id,),
    )
    lines = [f"# Research report: {question['question']}", "", "## Current conclusion", ""]
    demonstrated = [
        r
        for r in relationships
        if r["kind"] == "demonstrated_transmission"
        and by_id[r["evidence_id"]]["evidence_type"] == "demonstrated_transmission"
        and by_id[r["evidence_id"]]["primary_source_verified"]
        and by_id[r["evidence_id"]]["transmission_verified"]
    ]
    if demonstrated:
        lines.append(
            "A direct transmission claim is recorded below. Its source and dating still require independent review; the overall conclusion remains provisional."
        )
    else:
        lines.append(
            "Inconclusive. No demonstrated transmission is recorded. Observed usage, chronology, and similarity alone do not establish origin or influence."
        )
    if latest_run:
        run = latest_run[0]
        lines.append(
            f"Latest run: {_cell(run['status'])} ({_cell(run['stop_reason'])}); "
            f"{run['actions_taken']} logical actions, {run['provider_requests']} provider requests, "
            f"{run['unknown_spend_requests']} requests with unknown spend."
        )
    lines.extend(["", "## Competing hypotheses (interpretations)", ""])
    if hypotheses:
        for h in hypotheses:
            lines.append(
                f"- **H{h['id']} [{h['status']}]** {_cell(h['statement'])} — {_cell(h['rationale'])}"
            )
    else:
        lines.append("No hypotheses recorded.")
    if suggestions:
        lines.extend(["", "## Provisional model status suggestions", ""])
        lines.append(
            "These unaccepted model rationales may describe secondary attribution as direct usage; check the evidence categories below."
        )
        for suggestion in suggestions:
            lines.append(
                f"- H{suggestion['hypothesis_id']} → {suggestion['proposed_status']} "
                f"(not accepted); cites E{', E'.join(str(i) for i in json.loads(suggestion['evidence_ids_json']))}: "
                f"{_cell(suggestion['rationale'])}"
            )
    lines.extend(["", "## Verified chronology", ""])
    dated = [
        e
        for e in displayed
        if e["evidence_type"] == "observed_usage"
        and e["quote_verified_date"]
        and e["primary_source_verified"]
    ]
    if not dated:
        lines.append("No independently dated quote of directly observed usage is recorded.")
    lines.extend(
        [
            "",
            "| Verified quote date | Evidence | Type | Term sense (provisional) | Exact excerpt | Surrounding context | Source group | Source |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for e in dated:
        lines.append(
            f"| {_cell(e['quote_verified_date'])} | E{e['id']} | {_cell(e['evidence_type'])} | "
            f"{_cell(e['term_sense'])} | “{_cell(e['exact_quote'])}” | "
            f"{_cell(e['quote_context'])} | "
            f"G{group_by_source[e['source_id']]} (independence unresolved) | "
            f"{_cite(e['retrieved_url'], e['title'], e['source_id'])} |"
        )
    lines.extend(["", "## Evidence outside verified chronology", ""])
    undated = [e for e in displayed if e not in dated]
    sections = (
        ("Other directly observed historical usage", "observed_usage"),
        ("Later attribution of historical usage", "attributed_usage"),
        ("Origin and coinage claims", "attributed_origin_claim"),
        ("Attributed intent", "attributed_intent"),
        ("Interpretive commentary", "interpretive_context"),
    )
    shown: set[int] = set()
    for heading, category in sections:
        rows = [e for e in undated if e["evidence_type"] == category]
        if not rows and category != "observed_usage":
            continue
        lines.extend(["", f"### {heading}", ""])
        if not rows:
            lines.append("No additional direct usage outside the verified chronology.")
            continue
        for e in rows:
            shown.add(e["id"])
            verified_date = (
                f" Independently verified date of this excerpt: {e['quote_verified_date']}."
                if e["quote_verified_date"]
                else ""
            )
            lines.append(
                f"- E{e['id']} [G{group_by_source[e['source_id']]}; {e['evidence_type']}; "
                f"{e['term_sense']}; {e['quote_region']}]: “{_cell(e['exact_quote'])}” "
                f"{_cite(e['retrieved_url'], e['title'], e['source_id'])}. "
                f"Page publication metadata: {_cell(e['page_publication_date'])} (unverified for this quote)."
                f"{verified_date} Context: “{_cell(e['quote_context'])}”"
            )
    for e in undated:
        if e["id"] not in shown:
            lines.append(
                f"- E{e['id']} [{e['evidence_type']}]: “{_cell(e['exact_quote'])}” "
                f"{_cite(e['retrieved_url'], e['title'], e['source_id'])}."
            )
    if source_captures:
        lines.extend(["", "## Source captures and independence", ""])
        groups: dict[int, list] = {}
        for capture in source_captures:
            groups.setdefault(group_by_source[capture["source_id"]], []).append(capture)
        for group_id, captures in sorted(groups.items()):
            hashes = {capture["content_hash"] for capture in captures}
            relation = "revisions and copies" if len(hashes) > 1 else "identical extracted text"
            lines.append(
                f"- G{group_id}: {len(captures)} capture(s); {relation}; independence unresolved."
            )
            for capture in captures:
                lines.append(
                    f"  - Source {capture['source_id']}: [{_cell(capture['retrieved_url'])}]"
                    f"({capture['retrieved_url']}); page date {_cell(capture['page_publication_date'])} "
                    "(unverified for any quote)."
                )
    lines.extend(["", "## Evidence in relation to each hypothesis", ""])
    for h in hypotheses:
        lines.extend([f"### H{h['id']}: {_cell(h['statement'])}", ""])
        for heading, kinds in (
            (
                "For",
                {
                    "supports",
                    "possible_transmission",
                    "attributed_transmission",
                    "demonstrated_transmission",
                },
            ),
            ("Against", {"contradicts"}),
        ):
            lines.append(f"**{heading}**")
            relevant = [
                r
                for r in links.get(h["id"], [])
                if r["kind"] in kinds and by_id[r["evidence_id"]]["transmission_verified"]
            ]
            relevant.sort(
                key=lambda r: (
                    by_id[r["evidence_id"]]["evidence_type"] != "demonstrated_transmission",
                    by_id[r["evidence_id"]]["confidence"] != "high",
                )
            )
            if not relevant:
                lines.append("- None recorded.")
            shown_groups: set[int] = set()
            for r in relevant:
                e = by_id[r["evidence_id"]]
                group = group_by_source[e["source_id"]]
                if group in shown_groups or len(shown_groups) >= 5:
                    continue
                shown_groups.add(group)
                lines.append(
                    f"- E{e['id']} [G{group}; independence unresolved] "
                    f"({e['evidence_type']}; {r['kind']}): "
                    f"{_cell(e['normalized_claim'])}. Interpretation: {_cell(r['rationale'])}. "
                    f"{_cite(e['retrieved_url'], e['title'], e['source_id'])}"
                )
            lines.append("")
    provisional_links = [
        r for r in relationships if not by_id[r["evidence_id"]]["transmission_verified"]
    ]
    if provisional_links:
        lines.extend(["## Provisional relationship labels", ""])
        for relation in provisional_links:
            lines.append(
                f"- E{relation['evidence_id']} → H{relation['hypothesis_id']} "
                f"[{relation['kind']}], not accepted: {_cell(relation['rationale'])}"
            )
        lines.append("")
    lines.extend(["## Unresolved gaps and adversarial review", ""])
    gaps: list[str] = []
    for row in reviews:
        review = json.loads(row["content_json"])
        for key in (
            "overclaims",
            "dating_concerns",
            "source_dependence",
            "transmission_gaps",
            "falsification_tests",
        ):
            gaps.extend(review.get(key, []))
    for gap in dict.fromkeys(gaps):
        lines.append(f"- {_cell(gap)}")
    for note in notes:
        if note["kind"] in {"gap", "reliability", "adjudication"}:
            lines.append(
                f"- Former H{note['former_hypothesis_id']} [{note['kind']}]: "
                f"{_cell(note['statement'])}"
            )
    if not gaps and not any(
        note["kind"] in {"gap", "reliability", "adjudication"} for note in notes
    ):
        if reviews:
            lines.append(
                "- Reviews recorded no explicit gaps; publication dating and source independence still require verification."
            )
        else:
            lines.append(
                "- No adversarial review recorded yet; publication dates and source independence remain unverified."
            )
    if leads:
        lines.extend(["", "## Extracted leads", ""])
        for lead in leads:
            lines.append(
                f"- {_cell(lead['kind'])}: {_cell(lead['value'])} — "
                f"{_cell(lead['rationale'])}. "
                f"{_cite(lead['retrieved_url'], lead['title'], lead['source_id'])}"
            )
    research_leads = [note for note in notes if note["kind"] == "lead"]
    if research_leads:
        lines.extend(["", "## Research leads", ""])
        for note in research_leads:
            lines.append(f"- Former H{note['former_hypothesis_id']}: {_cell(note['statement'])}")
    overlaps = [note for note in notes if note["kind"] == "overlap"]
    if overlaps:
        lines.extend(["", "## Archived overlapping proposals", ""])
        for note in overlaps:
            lines.append(f"- Former H{note['former_hypothesis_id']}: {_cell(note['statement'])}")
    lines.extend(
        [
            "",
            "## Research frontier",
            "",
            (
                f"{frontier_counts.get('active', 0)} active queries; "
                f"{frontier_counts.get('deferred', 0)} deferred leads; "
                f"{frontier_counts.get('exhausted', 0)} exhausted queries; "
                f"{frontier_counts.get('rejected_duplicate', 0)} rejected duplicates."
            ),
            "",
            "### Active searches",
            "",
        ]
    )
    if pending:
        for query in pending:
            lines.append(
                f"- Q{query['id']} `{query['query']}` [{query['information_value']}] — {_cell(query['gap'])}"
            )
    else:
        lines.append("- No active searches recorded.")
    lines.extend(["", "### Deferred leads", ""])
    for query in deferred:
        lines.append(f"- Q{query['id']} `{query['query']}` — {_cell(query['rationale'])}")
    if not deferred:
        lines.append("- No deferred leads recorded.")
    if candidates:
        lines.extend(["", "## Pending page candidates", ""])
        for candidate in candidates:
            lines.append(f"- [{_cell(candidate['title'] or candidate['url'])}]({candidate['url']})")
    lines.extend(["", "## Inaccessible or failed sources", ""])
    if failures:
        for failure in failures:
            state = "retryable" if failure["fetch_retryable"] else "unresolved"
            destination = (
                f"; final URL {_cell(failure['fetch_final_url'])}"
                if failure["fetch_final_url"] and failure["fetch_final_url"] != failure["url"]
                else ""
            )
            lines.append(
                f"- [{_cell(failure['url'])}]({failure['url']}) — {state}: "
                f"{_cell(failure['fetch_error'])}{destination}; "
                f"discovered by `{_cell(failure['query'])}`"
            )
    else:
        lines.append(
            "None recorded. Ordinary HTML retrieval cannot establish archive or PDF coverage."
        )
    lines.extend(
        [
            "",
            "## Provenance notes",
            "",
            "Each excerpt is checked against stored fetched text, with bounded context and page region. Page publication metadata never dates the excerpt. Source groups combine identical text and revisions; independence remains unresolved. Canonical URLs are unverified aliases. Search snippets are leads only. Earliest observed use of a string is distinct from origin of the modern genre term.",
            "",
        ]
    )
    for e in displayed:
        lines.append(
            f"- E{e['id']} → source {e['source_id']}, retrieved {e['retrieval_date']}, "
            f"quote offset {e['quote_start']}, text SHA-256 `{e['content_hash']}`, "
            f"document SHA-256 `{e['document_hash']}`, "
            f"type {_cell(e['source_type'])}; "
            f"author {_cell(e['author'])}; canonical alias {_cell(e['canonical_url'])}."
        )
    return "\n".join(lines).rstrip() + "\n"
