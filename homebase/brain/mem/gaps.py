"""Computed day/week/month gaps for "how long between A and B" questions.

Every dated clause in the retrieved memories is a candidate; each side of the question must
match one clause clearly better than any rival date, otherwise nothing is said. The output
names the two clauses it used so a reader (or reviewer) can check it.
"""
from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from homebase.brain.mem.relative import day_phrases
from homebase.brain.mem.store import split_sides, stem
from homebase.brain.mem.timeparse import _MONTH_RE, _MONTHS

_ASKS_GAP = re.compile(r"\bhow\s+(?:many\s+(?:days?|weeks?|months?|years?)|long)\b", re.I)
_CLAUSE = re.compile(r"(?<=[.!?])\s+|\n+|[,;]\s+|\s+(?:and|but|while)\s+", re.I)
_MONTH_FIRST = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.I)
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?(?:,?\s+(\d{{4}}))?\b", re.I)
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")


_DAY_COMMA_YEAR = re.compile(r"(\d{1,2}(?:st|nd|rd|th)?),\s+(\d{4})\b")


def _clauses(text: str) -> list[str]:
    """Clause split that keeps "January 15, 2023" together (its comma is not a clause break)."""
    return [c for c in _CLAUSE.split(_DAY_COMMA_YEAR.sub(r"\1 \2", text)) if c.strip()]


@dataclass(frozen=True)
class Mention:
    clause: str
    epoch: float


def _utc(year: int, month: int, day: int) -> float | None:
    if 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]:
        return datetime(year, month, day, tzinfo=timezone.utc).timestamp()
    return None


def _nearest_year(month: int, day: int, said_at: float) -> float | None:
    year = datetime.fromtimestamp(said_at, timezone.utc).year
    options = [e for e in (_utc(y, month, day) for y in (year - 1, year, year + 1)) if e is not None]
    return min(options, key=lambda e: abs(e - said_at), default=None)


def _explicit_dates(clause: str, said_at: float) -> list[float]:
    found = []
    for m in _ISO.finditer(clause):
        found.append(_utc(int(m[1]), int(m[2]), int(m[3])))
    for m in _MONTH_FIRST.finditer(clause):
        if m[1] == "may":      # the verb, not the month
            continue
        month, day = _MONTHS[m[1].lower()], int(m[2])
        found.append(_utc(int(m[3]), month, day) if m[3] else _nearest_year(month, day, said_at))
    for m in _DAY_FIRST.finditer(clause):
        month, day = _MONTHS[m[2].lower()], int(m[1])
        found.append(_utc(int(m[3]), month, day) if m[3] else _nearest_year(month, day, said_at))
    return [e for e in found if e is not None]


def date_mentions(text: str, said_at: float | None) -> list[Mention]:
    """Dated clauses of ``text``; a date without a year takes the year nearest ``said_at``."""
    if said_at is None:
        return []
    mentions = []
    for clause in _clauses(text):
        clause = clause.strip()
        if len(clause) < 8:
            continue
        for epoch in _explicit_dates(clause, said_at) + [e for _, e in day_phrases(clause, said_at)]:
            mentions.append(Mention(clause, epoch))
    return mentions


def _weights(tokens: list[str], clauses: list[str]) -> dict[str, float]:
    """Rarer words identify an event better: weight = ln(1 + clauses / clauses containing the word)."""
    blobs = [c.lower() for c in clauses]
    return {t: math.log(1 + len(blobs) / max(1, sum(stem(t) in blob for blob in blobs))) for t in tokens}


def _score(tokens: list[str], rival: list[str], clause: str, weight: dict[str, float]) -> tuple[float, int]:
    blob = clause.lower()
    hits = [t for t in tokens if stem(t) in blob]
    against = [t for t in rival if t not in tokens and stem(t) in blob]   # words both sides share are neutral
    return sum(weight[t] for t in hits) - sum(weight[t] for t in against), len(hits)


def _pick(tokens: list[str], rival: list[str], mentions: list[Mention], clauses: list[str]) -> Mention | None:
    """The mention matching ``tokens`` best, only if clearly better than any other date."""
    weight = _weights(tokens + rival, clauses)
    scored = sorted(((_score(tokens, rival, m.clause, weight), m) for m in mentions),
                    key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0][1] < 2 or scored[0][0][0] <= 0:
        return None
    best_score, best = scored[0]
    # A clause matching only generic words ("deadline", "features") is not evidence for the event:
    # it must contain a word of this side that (almost) no clause with another date shares.
    # Near-ties between dates are not trusted either.
    own = {m.clause for m in mentions if m.epoch == best.epoch}
    elsewhere = [c.lower() for c in clauses if c not in own]
    blob = best.clause.lower()
    if not any(stem(t) in blob and sum(stem(t) in c for c in elsewhere) <= len(clauses) // 20 for t in tokens):
        return None
    blob = best.clause.lower()
    if sum(stem(t) in blob for t in tokens) < 0.5 * len(tokens):   # mostly-unmatched side: too weak to trust
        return None
    if any(score[0] >= 0.85 * best_score[0] and m.epoch != best.epoch for score, m in scored[1:]):
        return None
    return best


def _day(epoch: float) -> str:
    moment = datetime.fromtimestamp(epoch, timezone.utc)
    return f"{moment.day} {calendar.month_name[moment.month]} {moment.year}"


def _span(days: int, first: float, second: float) -> str:
    parts = [f"{days} day{'s' if days != 1 else ''}"]
    if days >= 14:
        weeks, rest = divmod(days, 7)
        parts.append(f"{weeks} weeks" if not rest else f"about {weeks} weeks")
    a, b = (datetime.fromtimestamp(e, timezone.utc) for e in (first, second))
    months = (b.year - a.year) * 12 + b.month - a.month - (1 if b.day < a.day else 0)
    if months >= 2:
        parts.append(f"{months} months")
    if months >= 12:
        parts.append(f"{months // 12} year{'s' if months >= 24 else ''}")
    return f"{parts[0]} ({', '.join(parts[1:])})" if parts[1:] else parts[0]


def _clip(text: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def gap_fact(question: str, items: Iterable[tuple[str, float | None]]) -> str | None:
    """Computed gap for a "how many days/weeks/... between A and B" question, or None."""
    if not _ASKS_GAP.search(question or ""):
        return None
    sides = split_sides(question)
    if len(sides) != 2:
        return None
    items = list(items)
    mentions = [m for text, said in items for m in date_mentions(text, said)]
    clauses = [c for text, _ in items for c in _clauses(text)]
    first = _pick(sides[0], sides[1], mentions, clauses)
    second = _pick(sides[1], sides[0], mentions, clauses)
    # The two sides are different events: once one is clear, the other is chosen among other dates.
    if first and not second:
        second = _pick(sides[1], sides[0], [m for m in mentions if m.epoch != first.epoch], clauses)
    elif second and not first:
        first = _pick(sides[0], sides[1], [m for m in mentions if m.epoch != second.epoch], clauses)
    if first is None or second is None:
        return None
    early, late = sorted((first, second), key=lambda m: m.epoch)
    days = round((late.epoch - early.epoch) / 86400)
    return (f'Computed from the memories below: "{_clip(first.clause)}" is on {_day(first.epoch)}; '
            f'"{_clip(second.clause)}" is on {_day(second.epoch)}. '
            f"The gap between them is {_span(days, early.epoch, late.epoch)}.")
