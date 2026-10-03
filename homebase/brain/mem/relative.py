"""Resolve relative time phrases ("yesterday", "last week") against the date they were said.

Each phrase is resolved at its own granularity: a day phrase gives a date, "last week"
stays "the week before <date>", "last month" gives the month. Nothing is guessed when
the message has no date.
"""
from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta, timezone

_NUMBER = r"\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_PHRASE = re.compile(
    rf"(?<![\w-])(?:(?P<ago>(?:{_NUMBER})\s+(?:day|week|month|year)s?\s+ago)"
    rf"|(?P<simple>yesterday|tomorrow|last\s+night)"
    rf"|(?P<weekday>last\s+(?:{_DAYS}))"
    rf"|(?P<weekend>last\s+weekend)"
    rf"|(?P<unit>(?:last|this|next)\s+(?:week|month|year)))(?![\w-])", re.I)
_WORDS = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10,
              eleven=11, twelve=12)


def _day(moment: datetime) -> str:
    return f"{moment.day} {calendar.month_name[moment.month]} {moment.year}"


def _month(moment: datetime, offset: int) -> str:
    index = moment.year * 12 + moment.month - 1 + offset
    return f"{calendar.month_name[index % 12 + 1]} {index // 12}"


def _shift_months(said: datetime, months: int) -> datetime:
    index = said.year * 12 + said.month - 1 - months
    year, month = index // 12, index % 12 + 1
    return said.replace(year=year, month=month, day=min(said.day, calendar.monthrange(year, month)[1]))


def _ago_date(phrase: str, said: datetime) -> datetime:
    count, unit = phrase.lower().split()[:2]
    count, unit = int(count) if count.isdigit() else _WORDS[count], unit.rstrip("s")
    if unit in ("month", "year"):
        return _shift_months(said, count * (12 if unit == "year" else 1))
    return said - timedelta(days=count * (7 if unit == "week" else 1))


def _ago(phrase: str, said: datetime) -> str:
    """Days are exact; weeks, months and years are only approximate, so they are not given as exact days."""
    unit = phrase.lower().split()[1].rstrip("s")
    moment = _ago_date(phrase, said)
    if unit == "day":
        return _day(moment)
    if unit == "week":
        return f"around {_day(moment)}"
    return f"around {_month(moment, 0)}" if unit == "month" else f"around {moment.year}"


def _resolve(match: re.Match, said: datetime) -> str:
    if match["ago"]:
        return _ago(match["ago"], said)
    if match["simple"]:
        word = " ".join(match["simple"].lower().split())
        return _day(said + timedelta(days=1 if word == "tomorrow" else -1))
    if match["weekday"]:
        target = list(calendar.day_name).index(match["weekday"].split()[-1].capitalize())
        return _day(said - timedelta(days=(said.weekday() - target) % 7 or 7))
    if match["weekend"]:
        return f"the weekend before {_day(said)}"
    which, unit = match["unit"].lower().split()
    offset = {"last": -1, "this": 0, "next": 1}[which]
    if unit == "week":
        return {-1: f"the week before {_day(said)}", 0: f"the week of {_day(said)}",
                1: f"the week after {_day(said)}"}[offset]
    return _month(said, offset) if unit == "month" else str(said.year + offset)


def resolve_phrases(text: str, said_at: float | None) -> list[str]:
    """``["last week = the week before 27 March 2023", ...]``, first use of each phrase, in order."""
    if said_at is None:
        return []
    said = datetime.fromtimestamp(said_at, timezone.utc)
    notes, seen = [], set()
    for match in _PHRASE.finditer(text):
        key = " ".join(match[0].lower().split())
        if key not in seen:
            seen.add(key)
            notes.append(f"{match[0]} = {_resolve(match, said)}")
    return notes


def day_phrases(text: str, said_at: float) -> list[tuple[str, float]]:
    """Phrases that name one calendar day ("yesterday", "3 days ago", "last Friday") with that day's epoch."""
    said = datetime.fromtimestamp(said_at, timezone.utc)
    midnight = said.replace(hour=0, minute=0, second=0, microsecond=0)
    found = []
    for match in _PHRASE.finditer(text):
        if match["ago"]:
            if match["ago"].lower().split()[1].rstrip("s") != "day":
                continue   # weeks/months/years ago are approximate, not a calendar day
            day = _ago_date(match["ago"], midnight)
        elif match["simple"]:
            day = midnight + timedelta(days=1 if " ".join(match["simple"].lower().split()) == "tomorrow" else -1)
        elif match["weekday"]:
            target = list(calendar.day_name).index(match["weekday"].split()[-1].capitalize())
            day = midnight - timedelta(days=(said.weekday() - target) % 7 or 7)
        else:
            continue
        found.append((match[0], day.timestamp()))
    return found
