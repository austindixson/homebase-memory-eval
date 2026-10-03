"""Dated fact cards: short, ordinary memories with the resolved date up front.

A card restates, verbatim, one sentence (or one clause when the sentence holds several dates)
of a message that carries a calendar date, and puts that date first so retrieval and a reader
find it. A card says nothing the message did not; it answers no question.
"""
from __future__ import annotations

import re

from homebase.brain.mem.gaps import _day, date_mentions

MAX_CARDS = 4
MAX_LENGTH = 260
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _card(epoch: float, text: str) -> str:
    text = re.sub(r"^(?:and|but|while)\s+", "", re.sub(r"\s+", " ", text).strip(), flags=re.I).rstrip(".!?")
    prefix = f"{_day(epoch)}: "
    if len(prefix) + len(text) > MAX_LENGTH:
        text = text[: MAX_LENGTH - len(prefix) - 1].rstrip() + "…"
    return prefix + text


def fact_cards(text: str, said_at: float | None) -> list[str]:
    """Up to MAX_CARDS cards for the dated sentences of ``text``, in order, without duplicates."""
    if said_at is None:
        return []
    cards: list[str] = []
    for sentence in _SENTENCE.split(text):
        if len(sentence.strip()) < 8:
            continue
        mentions = date_mentions(sentence, said_at)
        if not mentions:
            continue
        dates = {m.epoch for m in mentions}
        found = [(mentions[0].epoch, sentence)] if len(dates) == 1 else [(m.epoch, m.clause) for m in mentions]
        for epoch, body in found:
            card = _card(epoch, body)
            if card not in cards:
                cards.append(card)
    return cards[:MAX_CARDS]
