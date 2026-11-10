"""Group context for asks (docs/whatsapp/ARCHITECTURE.md W10.2-W10.5, agentd-api 1.2 §4.1).

- `Ring`: per chat, in memory only, the last 50 messages (8,000 characters in all, each text
  clipped to 1,000): other members' context messages, your own non-trigger messages and Mac's
  replies. The whole ring goes with every ask; agentd frames only what the session has not seen.
- `quoted()`: the message your trigger replies to, as `ask.context.quoted`.
- `resolve_mentions()`: "@Priya" in Mac's final text becomes "@<number>" plus the JID, only for a
  name that exactly one member has. Deterministic; unknown or ambiguous names stay plain text.
"""

from __future__ import annotations

import bisect
from typing import Any

from agentd import schema as s
from chatd.channels.base import ContextMessage, Quoted
from macd.brain.prompts import (
    RESERVED_NAMES,  # one set, shared with the prompt labels (V4)
)

MAX_MENTIONS = 5
HISTORY_COUNT = 30


def clip(text: str, limit: int) -> str:
    """At most `limit` characters; a cut text ends with "…" so the model knows it is partial. Any
    text the bridge already cut (4,000) is longer than this limit, so it is marked here too."""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class Ring:
    def __init__(self, max_messages: int = s.CONTEXT_MAX_MESSAGES, max_chars: int = s.CONTEXT_MAX_CHARS,
                 max_text: int = s.CONTEXT_MAX_TEXT, *, fresh: bool = False) -> None:
        self.max_messages, self.max_chars, self.max_text = max_messages, max_chars, max_text
        self.fresh = fresh  # started by "new chat": no history fetch refills it
        self._items: list[s.ContextMessage] = []  # time order
        self._deleted: set[str] = set()  # deleted on WhatsApp: never added back (a late history page)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, msg_ref: str, *, sender: str = "", owner: bool = False, mac: bool = False, at: float = 0.0,
            text: str = "", media: dict[str, Any] | None = None, earlier: bool = False) -> None:
        if not msg_ref or msg_ref in self._deleted or any(m.msg_ref == msg_ref for m in self._items):
            return
        cm = s.ContextMessage(msg_ref=msg_ref, sender=sender, owner=owner, mac=mac, at=at,
                              text=clip(text, self.max_text) or None,
                              media=s.ContextMedia(kind=str(media.get("kind", "")), name=media.get("name") or None)
                              if media else None,
                              earlier=True if earlier else None)
        bisect.insort(self._items, cm, key=lambda m: m.at)
        while len(self._items) > self.max_messages or self.chars() > self.max_chars:
            self._items.pop(0)

    def add_context(self, cm: ContextMessage) -> None:
        self.add(cm.ref.message_id, sender=cm.sender_name or cm.sender, owner=cm.from_owner, at=cm.at,
                 text=cm.text, media=cm.media, earlier=cm.history)

    def remove(self, msg_ref: str) -> None:
        if len(self._deleted) >= 1000:  # bounded: the oldest ids have long left WhatsApp's history
            self._deleted.clear()
        self._deleted.add(msg_ref)
        self._items = [m for m in self._items if m.msg_ref != msg_ref]

    def chars(self) -> int:
        return sum(len(m.text or "") for m in self._items)

    def messages(self, exclude: str = "") -> list[s.ContextMessage]:
        return [m for m in self._items if m.msg_ref != exclude]


def display_name(member: dict[str, Any]) -> str:
    """Contact name, else push name (the bridge's `name`), else the number's last 4 digits."""
    name = str(member.get("name") or "").strip()
    return name or f"…{str(member.get('number') or '')[-4:]}"


def quoted(q: Quoted | None) -> s.QuotedMessage | s.OtherChatQuote | None:
    if q is None:
        return None
    if q.kind == "other_chat":
        return s.OtherChatQuote(kind="other_chat")
    media = None
    if q.media is not None:
        media = s.ContextMedia(kind=q.media.kind, name=q.media.name or None,
                               path=str(q.media.path) if q.media.path else None)
    return s.QuotedMessage(sender=q.sender_name or q.sender, owner=q.from_owner, mac=q.from_mac, at=q.at,
                           text=clip(q.text, s.CONTEXT_MAX_TEXT) or None, links=list(q.links),  # the bridge filters links
                           media=media)


def _word(c: str) -> bool:
    return c.isalnum() or c == "_"


def resolve_mentions(text: str, members: list[dict[str, Any]],
                     reserved: frozenset[str] = RESERVED_NAMES) -> tuple[str, tuple[str, ...]]:
    """Each `@<name>` matching (longest first, case-insensitive, at word boundaries) the display
    name of exactly one member becomes `@<number>`. At most 5 members; names shared by two
    members or equal to a reserved label stay plain text."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for m in members:
        name = str(m.get("name") or "").strip()
        if name and m.get("jid") and m.get("number"):
            by_name.setdefault(name.casefold(), []).append(m)
    names = sorted(by_name, key=len, reverse=True)
    out, jids, i = [], [], 0
    while i < len(text):
        c = text[i]
        if c == "@" and (i == 0 or not _word(text[i - 1])):
            rest = text[i + 1:]
            hit = next((n for n in names if rest[:len(n)].casefold() == n
                        and (len(rest) == len(n) or not _word(rest[len(n)]))), None)
            if hit is not None:
                cands = by_name[hit]
                jid = str(cands[0]["jid"]) if len(cands) == 1 and hit not in reserved else ""
                if jid and (jid in jids or len(jids) < MAX_MENTIONS):
                    if jid not in jids:
                        jids.append(jid)
                    out.append(f"@{cands[0]['number']}")
                else:
                    out.append(text[i:i + 1 + len(hit)])  # plain text: unknown, ambiguous or over the cap
                i += 1 + len(hit)
                continue
        out.append(c)
        i += 1
    return "".join(out), tuple(jids)
