"""Small, deterministic admission policy for research searches."""

from __future__ import annotations

import re

_STOP = {
    "a",
    "an",
    "and",
    "are",
    "article",
    "as",
    "at",
    "by",
    "claim",
    "earliest",
    "find",
    "first",
    "for",
    "from",
    "how",
    "in",
    "interview",
    "is",
    "name",
    "named",
    "naming",
    "of",
    "on",
    "or",
    "origin",
    "original",
    "source",
    "term",
    "text",
    "the",
    "to",
    "usage",
    "use",
    "verify",
    "was",
    "who",
    "with",
}
_VALUE = {"low": 1, "medium": 2, "high": 3}
TARGET_RANK = {
    "pc-music-2014": 5,
    "bjork-transmission": 4,
    "hyperballad-title": 3,
    "spotify-naming": 2,
    "scene-2014-2018": 1,
}
TARGET_ACTIVE_LIMIT = 3


def generic_query(query: str) -> bool:
    """Broad retrospectives do not displace searches for a named source."""
    text = query.lower()
    broad = any(
        phrase in text
        for phrase in (
            "hyperpop history",
            "history of hyperpop",
            "origin of hyperpop",
            "what is hyperpop",
        )
    )
    anchored = any(
        word in text
        for word in (
            "sherburne",
            "mcdonald",
            "szabo",
            "hyperballad",
            "pc music",
            "485-pc-musics-twisted-electronic-pop-a-users-manual",
        )
    )
    return broad and not anchored


def target_priority(target: str, query: str, value: str) -> int:
    score = TARGET_RANK[target] * 100 + _VALUE[value] * 10
    score += min(query.count('"'), 4)
    if "site:" in query:
        score += 3
    if "web.archive.org" in query or "archive.org" in query:
        score += 2
    if generic_query(query):
        score -= 50
    return score


def _words(query: str) -> set[str]:
    text = re.sub(r"\ba\s*\.?\s*g\b", "ag", query.lower())
    text = text.replace("hyper-pop", "hyperpop").replace("hyper pop", "hyperpop")
    text = text.replace("2014..2018", "2014 2018")
    return {word for word in re.findall(r"[a-z0-9]+", text) if word not in _STOP}


def avenue(query: str) -> str:
    """Identify repeated searches for one source or historical distinction."""
    words = _words(query)
    if "sherburne" in words and "pitchfork" in words:
        return "sherburne-pitchfork-2014"
    if "shewey" in words and ("1988" in words or "cocteau" in words):
        if {"village", "voice"} <= words:
            return "shewey-1988-village-voice"
        if {"7", "days"} <= words:
            return "shewey-1988-7-days"
        if "source magazine" in query.lower():
            return "shewey-1988-source-magazine"
        return "shewey-1988"
    if "glenn" in words and "mcdonald" in words:
        if "2014" in words and {"pc", "music"} <= words:
            return "mcdonald-2014-source"
        if "everynoise" in words:
            return "mcdonald-everynoise"
        if "spotify" in words:
            return "mcdonald-spotify-statement"
        return "mcdonald-direct-statement"
    if "lizzy" in words and "szabo" in words:
        return "szabo-playlist-statement"
    if "bj" in words or "bjork" in words or "björk" in query.lower():
        if "hyperballad" in words and ("title" in words or "meaning" in words):
            return "bjork-hyperballad-title"
        if "hyperballad" in words and "hyperpop" in words:
            return "bjork-hyperpop-transmission"
    if "pc" in words and "music" in words and "2014" in words:
        return "pc-music-2014-usage"
    years = sorted(word for word in words if re.fullmatch(r"(?:19|20)\d{2}", word))
    return "words:" + "-".join(sorted(words - set(years))) + ":" + "-".join(years)


def overlap(left: str, right: str) -> bool:
    a, b = avenue(left), avenue(right)
    if a == b:
        return True
    if (a.startswith("shewey-1988") and b.startswith("shewey-1988")) or (
        a.startswith("mcdonald-") and b.startswith("mcdonald-")
    ):
        return False
    first, second = _words(left), _words(right)
    if not first or not second:
        return False
    years_a = {word for word in first if re.fullmatch(r"(?:19|20)\d{2}", word)}
    years_b = {word for word in second if re.fullmatch(r"(?:19|20)\d{2}", word)}
    if years_a and years_b and not years_a.intersection(years_b):
        return False
    return len(first & second) / len(first | second) >= 0.68


def legacy_value(query: str, rationale: str) -> str:
    text = f"{query} {rationale}".lower()
    if any(
        word in text
        for word in ("primary", "original", "contemporaneous", "direct statement", "archive scan")
    ):
        return "high"
    if any(
        word in text for word in ("2014", "1988", "1995", "2019", "björk", "björk", "hyperballad")
    ):
        return "medium"
    return "low"


def priority(value: str, query: str, basis: str) -> int:
    score = _VALUE[value] * 10
    if basis == "primary_source" or legacy_value(query, "") == "high":
        score += 5
    words = _words(query)
    if (
        {"sherburne", "pitchfork", "2014"} <= words
        or {"shewey", "1988"} <= words
        or {"glenn", "mcdonald", "2014", "pc", "music"} <= words
        or {"glenn", "mcdonald", "spotify"} <= words
        or {"bj", "hyperballad", "title"} <= words
        or {"bjork", "hyperballad", "title"} <= words
    ):
        score += 10
    if "site:" in query or "web.archive.org" in query.lower():
        score += 1
    return score
