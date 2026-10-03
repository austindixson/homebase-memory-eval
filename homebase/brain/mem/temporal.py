"""Deterministic temporal resolution at answer time (PRD W6).

Given the packed evidence and the question, compute event order / day, week and month
differences from *dated events* in code, so the answerer is handed the arithmetic
instead of doing it from relative phrases. Event dates (a relative phrase such as
"two weeks ago" resolved against the session date) are distinguished from session
dates (when the thing was *mentioned*), and the distinction is stated in the output.
Nothing is emitted unless every event it relies on has a real date.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from homebase.brain.mem.store import MemNode, MemStore, query_tokens, split_sides, stem
from homebase.brain.mem.timeparse import iso_date, parse_relative

_BETWEEN = re.compile(
    r"between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s*(?:\?|$)", re.I
)
_UNIT = re.compile(r"\b(days?|weeks?|months?|years?)\b", re.I)
_AGO = re.compile(r"\bhow\s+(?:many|long)\b.*\bago\b", re.I)


@dataclass
class DatedEvent:
    node_id: str
    text: str
    epoch: float
    label: str | None
    kind: str  # event (resolved from a relative phrase) | session (mention date)


@dataclass
class TemporalResult:
    lines: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def computed(self) -> bool:
        return bool(self.lines)


def _midnight(epoch: float) -> float:
    return float(int(epoch // 86400) * 86400)


def day_diff(earlier: float, later: float) -> int:
    """Calendar-day difference (UTC dates; time of day is ignored)."""
    return int(round((_midnight(later) - _midnight(earlier)) / 86400.0))


def months_between(earlier: float, later: float) -> tuple[int, int]:
    """Whole calendar months and remaining days between two epochs (earlier <= later)."""
    a = datetime.fromtimestamp(_midnight(earlier), tz=timezone.utc)
    b = datetime.fromtimestamp(_midnight(later), tz=timezone.utc)
    months = (b.year - a.year) * 12 + (b.month - a.month)
    anchor_day = min(a.day, calendar.monthrange(b.year, b.month)[1])
    if b.day < anchor_day:
        months -= 1
    y, m = divmod(a.year * 12 + (a.month - 1) + months, 12)
    m += 1
    start = a.replace(year=y, month=m, day=min(a.day, calendar.monthrange(y, m)[1]))
    return max(0, months), max(0, int((b - start).total_seconds() // 86400))


def describe_gap(earlier: float, later: float) -> str:
    days = day_diff(earlier, later)
    weeks, rem = divmod(days, 7)
    months, mdays = months_between(earlier, later)
    parts = [f"{days} days"]
    if weeks:
        parts.append(f"{weeks} week{'s' if weeks != 1 else ''}" + (f" {rem} day{'s' if rem != 1 else ''}" if rem else ""))
    parts.append(f"{months} month{'s' if months != 1 else ''} {mdays} day{'s' if mdays != 1 else ''}")
    return "; ".join(parts)


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sentence_events(n: MemNode, anchor: float) -> list[DatedEvent]:
    """Event dates inside a session chunk: each sentence with a relative / duration
    phrase resolved against the chunk's session date (``anchor``)."""
    out: list[DatedEvent] = []
    for sent in _SENT.split(n.content or ""):
        sent = sent.strip()
        if len(sent) < 12:
            continue
        hit = parse_relative(sent, anchor)
        if hit is not None:
            out.append(DatedEvent(n.id, sent, hit[0], hit[1], "event"))
    return out


def dated_events(nodes: Sequence[MemNode], store: MemStore) -> list[DatedEvent]:
    """Evidence with a real date: timeline events / explicit ``valid_from`` (session
    dates), plus per-sentence event dates resolved inside session chunks."""
    tl = store.timeline_for_nodes([n.id for n in nodes])
    out: list[DatedEvent] = []
    for n in nodes:
        ev0 = tl.get(n.id)
        anchor = None
        if ev0 and ev0.get("t_kind") == "absolute":
            anchor = ev0["t_sort"]
        elif n.valid_from is not None and abs(n.valid_from - (n.ts or 0.0)) > 1.0 and not ev0:
            anchor = float(n.valid_from)
        if anchor is not None:
            # Claims dated only by their session can still name the event's own date
            # ("in mid-February", "on February 27th"): resolve it here too.
            out.extend(_sentence_events(n, anchor))
        ev = tl.get(n.id)
        if ev and ev.get("t_kind") in ("absolute", "relative"):
            kind = "event" if ev["t_kind"] == "relative" else "session"
            out.append(DatedEvent(n.id, n.content, ev["t_sort"], ev.get("t_label"), kind))
        elif n.valid_from is not None and abs(n.valid_from - (n.ts or 0.0)) > 1.0:
            out.append(DatedEvent(n.id, n.content, float(n.valid_from), None, "session"))
    return out


def _best_event(
    tokens: list[str],
    events: Sequence[DatedEvent],
    exclude: set[str],
    other: list[str] | None = None,
) -> tuple[DatedEvent | None, int]:
    """Event that matches ``tokens`` best. With ``other`` (the opposite side of an
    A-vs-B question) an event only qualifies if it matches this side *more* than the
    other one, so question-frame words ("vehicle", "take care") need not match."""
    if not tokens:
        return None, 0
    need = 1 if other is not None else max(1, (len(tokens) + 1) // 2)
    best: DatedEvent | None = None
    best_key: tuple[int, int, int, int] = (0, 0, 0, 0)
    for ev in events:
        if ev.node_id in exclude:
            continue
        blob = ev.text.lower()
        hits = sum(1 for t in tokens if stem(t) in blob)
        rival = sum(1 for t in (other or []) if stem(t) in blob)
        if hits < need or (other is not None and hits <= rival):
            continue
        key = (hits - rival, hits, 1 if ev.kind == "event" else 0, -len(ev.text))
        if key > best_key:
            best, best_key = ev, key
    return best, best_key[1]


def _snip(text: str, n: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _tag(ev: DatedEvent) -> str:
    when = iso_date(ev.epoch)
    if ev.kind == "event":
        return f"{when}, event date resolved from \"{ev.label}\""
    return f"{when}, session/mention date (not necessarily the event date)"


def _pair(question: str) -> list[list[str]] | None:
    sides = split_sides(question)
    if len(sides) == 2:
        return sides
    m = _BETWEEN.search(question or "")
    if m:
        a, b = query_tokens(m.group(1)), query_tokens(m.group(2))
        if a and b:
            return [a, b]
    return None


def select_pair_evidence(
    question: str, nodes: Sequence[MemNode], store: MemStore,
) -> list[MemNode]:
    """Reserve a dated source for each named event before whole-query ranking.

    A source must match its event more strongly than the opposite event. Selection
    uses the same event matcher as temporal computation, without answer annotations.
    If either event lacks a dated source, leave the ordinary ranking unchanged.
    """
    pair = _pair(question)
    if pair is None:
        return []
    events = dated_events(nodes, store)
    a, _ = _best_event(pair[0], events, set(), other=pair[1])
    b, _ = _best_event(pair[1], events, {a.node_id} if a else set(), other=pair[0])
    if a is None or b is None:
        return []
    by_id = {node.id: node for node in nodes}
    return [by_id[a.node_id], by_id[b.node_id]]


def compute_temporal(
    question: str,
    nodes: Sequence[MemNode],
    store: MemStore,
    *,
    as_of: float | None = None,
) -> TemporalResult:
    """Order / gap facts for A-vs-B, ``between A and B`` and ``how long ago`` questions."""
    res = TemporalResult()
    events = dated_events(nodes, store)
    if not events:
        return res

    pair = _pair(question)
    if pair is not None:
        a_ev, a_hits = _best_event(pair[0], events, set(), other=pair[1])
        b_ev, b_hits = _best_event(
            pair[1], events, {a_ev.node_id} if a_ev else set(), other=pair[0]
        )
        if a_ev is None or b_ev is None:
            res.facts["both_events_dated"] = False
            return res
        if a_ev.kind != b_ev.kind:
            # An event date and a session/mention date are not comparable: asserting an
            # order across them is how a wrong "EARLIER" would be produced.
            res.facts.update(both_events_dated=False, mixed_date_kinds=True)
            res.lines.append(
                f"Event A \"{_snip(a_ev.text)}\" [{_tag(a_ev)}]; Event B \"{_snip(b_ev.text)}\" "
                f"[{_tag(b_ev)}]. One is an event date and the other only a session date, so "
                "their order is NOT determined by these dates; reason from the evidence text."
            )
            return res
        first, second = (a_ev, b_ev) if a_ev.epoch <= b_ev.epoch else (b_ev, a_ev)
        gap = describe_gap(first.epoch, second.epoch)
        same_day = day_diff(first.epoch, second.epoch) == 0
        res.facts.update(
            both_events_dated=True,
            earlier_id=first.node_id,
            later_id=second.node_id,
            days=day_diff(first.epoch, second.epoch),
        )
        res.lines.append(
            f"Event A \"{_snip(a_ev.text)}\" [{_tag(a_ev)}]; "
            f"Event B \"{_snip(b_ev.text)}\" [{_tag(b_ev)}]."
        )
        if same_day:
            res.lines.append("A and B fall on the same date; order is not determined by dates.")
        else:
            which = "A" if first is a_ev else "B"
            res.lines.append(f"Event {which} is EARLIER. Gap: {gap}.")
        return res

    if as_of is not None and _AGO.search(question or ""):
        toks = query_tokens(re.sub(r"\bhow\s+(?:many|long)\b.*?\bago\b", " ", question, flags=re.I))
        ev, _hits = _best_event(toks, events, set())
        if ev is not None and ev.epoch <= as_of:
            res.facts.update(days=day_diff(ev.epoch, as_of), as_of=iso_date(as_of))
            res.lines.append(
                f"Event \"{_snip(ev.text)}\" [{_tag(ev)}] was {describe_gap(ev.epoch, as_of)} "
                f"before the question date {iso_date(as_of)}."
            )
    return res
