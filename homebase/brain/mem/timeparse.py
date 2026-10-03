"""Deterministic time parsing for Lattice observe (PRD W3).

Never hashes a label into a number. A label that cannot be parsed yields
``epoch=None`` with ``kind="malformed"`` so callers can skip or flag it instead of
inventing an ordering. All times are UTC epoch seconds.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_MONTHS = {
    name.lower(): i
    for i, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_WEEKDAYS.update({name.lower(): i for i, name in enumerate(calendar.day_abbr)})

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "couple": 2, "few": 3, "several": 3,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO = re.compile(
    r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s]+(?:\([A-Za-z]{3}\)\s*)?(\d{1,2}):(\d{2})(?::(\d{2}))?)?"
)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?[-\s]+(\d{{1,2}})(?:st|nd|rd|th)?,?[-\s]+(\d{{4}})\b", re.I)
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)

_AGO = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|few|several)"
    r"\s+(day|week|month|year)s?\s+ago\b",
    re.I,
)
_LAST_UNIT = re.compile(r"\b(last|past|previous)\s+(week|month|year)\b", re.I)
_LAST_WEEKDAY = re.compile(
    r"\b(last|previous)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I
)
_LAST_WEEKEND = re.compile(r"\blast\s+weekend\b", re.I)
# "I've been playing X for about 2 months": the thing started N units before the anchor.
_SINCE_DURATION = re.compile(
    r"\b(?:been|being)\b[^.?!]{0,80}?\bfor\s+(?:about |around |almost |nearly |over |roughly )?"
    r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(day|week|month|year)s?\b",
    re.I,
)
# "February 27th", "mid-February", "early March", "in December" — no year given:
# resolved to the anchor's year (previous year if that would be after the anchor).
_MONTH_DAY_NOYEAR = re.compile(
    rf"\b(?:(?P<part>early|mid|late)[- ]?(?P<m1>{_MONTH_RE})|(?P<m2>{_MONTH_RE})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"|\bin\s+(?P<m3>{_MONTH_RE}))\b(?!,?\s*\d{{4}})",
    re.I,
)
_SIMPLE = re.compile(r"\b(yesterday|today|tonight|tomorrow|last night)\b", re.I)


@dataclass(frozen=True)
class ParsedTime:
    epoch: float | None
    label: str | None
    kind: str  # absolute | relative | ordinal | malformed | none
    anchored: bool = False  # relative phrase resolved against an anchor date

    @property
    def usable(self) -> bool:
        return self.epoch is not None


def _utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> float | None:
    try:
        return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def parse_absolute(text: str) -> tuple[float, str] | None:
    """First absolute date in ``text`` -> (epoch, matched label), else None."""
    m = _ISO.search(text or "")
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4) or 0)
        mm = int(m.group(5) or 0)
        ss = int(m.group(6) or 0)
        ts = _utc(y, mo, d, hh, mm, ss)
        if ts is not None:
            return ts, m.group(0).strip()
    m = _MDY.search(text or "")
    if m:
        ts = _utc(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        if ts is not None:
            return ts, m.group(0)
    m = _DMY.search(text or "")
    if m:
        ts = _utc(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        if ts is not None:
            return ts, m.group(0)
    m = _MONTH_YEAR.search(text or "")
    if m:
        ts = _utc(int(m.group(2)), _MONTHS[m.group(1).lower()], 1)
        if ts is not None:
            return ts, m.group(0)
    return None


def _shift_months(dt: datetime, months: int) -> datetime:
    total = dt.year * 12 + (dt.month - 1) - months
    y, m = divmod(total, 12)
    m += 1
    day = min(dt.day, calendar.monthrange(y, m)[1])
    return dt.replace(year=y, month=m, day=day)


def _count(token: str) -> int:
    tok = token.lower().replace(" of", "")
    return int(tok) if tok.isdigit() else _NUM_WORDS.get(tok, 1)


def _month_day(text: str, anchor: datetime) -> tuple[float, str] | None:
    for m in _MONTH_DAY_NOYEAR.finditer(text or ""):
        name = (m.group("m1") or m.group("m2") or m.group("m3") or "").lower()
        if name == "may" and not (m.group("day") or m.group("part")):
            continue  # "in may" is too often the verb
        month = _MONTHS.get(name)
        if not month:
            continue
        if m.group("day"):
            day = int(m.group("day"))
        else:
            day = {"early": 5, "mid": 15, "late": 25}.get((m.group("part") or "").lower(), 15)
        for year in (anchor.year, anchor.year - 1):
            ts = _utc(year, month, min(day, calendar.monthrange(year, month)[1]))
            if ts is not None and ts <= anchor.timestamp() + 86400:
                return ts, m.group(0).strip()
    return None


def parse_relative(text: str, anchor_epoch: float) -> tuple[float, str] | None:
    """First relative phrase in ``text`` resolved against ``anchor_epoch``."""
    anchor = datetime.fromtimestamp(anchor_epoch, tz=timezone.utc)
    low = text or ""
    md = _month_day(low, anchor)
    if md is not None:
        return md
    m = _AGO.search(low)
    if m:
        n, unit = _count(m.group(1)), m.group(2).lower()
        if unit == "day":
            dt = anchor - timedelta(days=n)
        elif unit == "week":
            dt = anchor - timedelta(weeks=n)
        elif unit == "month":
            dt = _shift_months(anchor, n)
        else:
            dt = _shift_months(anchor, 12 * n)
        return dt.timestamp(), m.group(0)
    m = _SINCE_DURATION.search(low)
    if m:
        n, unit = _count(m.group(1)), m.group(2).lower()
        if unit == "day":
            dt = anchor - timedelta(days=n)
        elif unit == "week":
            dt = anchor - timedelta(weeks=n)
        elif unit == "month":
            dt = _shift_months(anchor, n)
        else:
            dt = _shift_months(anchor, 12 * n)
        return dt.timestamp(), m.group(0)
    m = _LAST_WEEKDAY.search(low)
    if m:
        target = _WEEKDAYS[m.group(2).lower()]
        delta = (anchor.weekday() - target) % 7 or 7
        return (anchor - timedelta(days=delta)).timestamp(), m.group(0)
    m = _LAST_WEEKEND.search(low)
    if m:
        # Most recent Saturday strictly before the anchor's current week-end.
        delta = (anchor.weekday() - 5) % 7 or 7
        return (anchor - timedelta(days=delta)).timestamp(), m.group(0)
    m = _LAST_UNIT.search(low)
    if m:
        unit = m.group(2).lower()
        if unit == "week":
            dt = anchor - timedelta(weeks=1)
        elif unit == "month":
            dt = _shift_months(anchor, 1)
        else:
            dt = _shift_months(anchor, 12)
        return dt.timestamp(), m.group(0)
    m = _SIMPLE.search(low)
    if m:
        word = m.group(1).lower()
        delta = {"yesterday": -1, "last night": -1, "tomorrow": 1}.get(word, 0)
        return (anchor + timedelta(days=delta)).timestamp(), m.group(0)
    return None


def extract_time(text: str, *, anchor_epoch: float | None = None) -> ParsedTime:
    """Best time for a chunk of text.

    Priority: explicit ``date=``/``@anchor`` label (must parse, else ``malformed``),
    then any absolute date in the text, then relative phrases against ``anchor_epoch``,
    then ``turn=N`` as an ordinal (ordering within a session only), else ``none``.
    """
    text = text or ""
    labelled = re.search(
        r"(?:(?<![\w.])@|\bdate[=:])\s*([^\s\]]+(?:\s+\([A-Za-z]{3}\)\s*\d{1,2}:\d{2})?)",
        text,
        re.I,
    )
    if labelled:
        label = labelled.group(1).strip().rstrip(",;:")
        hit = parse_absolute(label)
        if hit is not None:
            return ParsedTime(hit[0], label, "absolute")
        return ParsedTime(None, label, "malformed")
    hit = parse_absolute(text)
    if hit is not None:
        return ParsedTime(hit[0], hit[1], "absolute")
    if anchor_epoch is not None:
        rel = parse_relative(text, anchor_epoch)
        if rel is not None:
            return ParsedTime(rel[0], rel[1], "relative", anchored=True)
    m = re.search(r"\bturn[=:]([^\s]+)", text, re.I)
    if m:
        turn_id = m.group(1).rstrip(",;:")
        # Compound source IDs (for example "2,25") identify a location in the
        # source data, not a scalar chronology. BEAM ingestion adds an independent
        # monotonic ``position=`` field for transcript order.
        if re.fullmatch(r"\d+", turn_id):
            return ParsedTime(float(turn_id), f"turn={turn_id}", "ordinal")
        return ParsedTime(None, f"turn={turn_id}", "malformed")
    return ParsedTime(None, None, "none")


def iso_date(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
