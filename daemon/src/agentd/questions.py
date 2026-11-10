"""Question binding (W3.2, M5, F14): agentd alone decides whether a chat reply answers the open
question. A conversation has at most one open question; anything that does not bind is ordinary
text and never an approval.

Binding: a 👍 whose `reaction_on` is the prompt; a text that quotes the prompt; or an unquoted
text while the question is open (never for `folder_approval`, which needs a quote or a 👍).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from agentd import schema as s

YES = frozenset({"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "send it", "go ahead", "do it",
                 "go", "yes please", "please do", "approve", "approved", "confirm", "sounds good"})
NO = frozenset({"no", "n", "nope", "cancel", "don't", "dont", "do not", "no thanks", "don't send it",
                "dont send it", "deny", "skip"})
_THUMBS = re.compile("^\U0001F44D[\U0001F3FB-\U0001F3FF]?️?$")


def yes_no(text: str) -> str | None:
    """"yes", "no" or None (any other text)."""
    t = unicodedata.normalize("NFKC", text).strip().lower().replace("’", "'")
    t = re.sub(r"[.!?,]+$", "", t).strip()
    if _THUMBS.match(t):
        return "yes"
    if t in YES:
        return "yes"
    if t in NO:
        return "no"
    return None


def is_thumbs_up(reaction: str | None) -> bool:
    return bool(reaction) and bool(_THUMBS.match(reaction.strip()))


@dataclass
class Question:
    """The open question of a conversation. `confirm` questions are outward confirmations
    (memory only); the rest mirror an `OpenQuestion` in the record."""

    question_id: str
    kind: s.QuestionKind
    text: str
    expires_at: float  # wall time
    prompt_ref: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    turn_internal: int = 0  # the brain turn a confirm belongs to


@dataclass
class Binding:
    bound: bool
    outcome: s.ReplyOutcome | None = None
    choice: int | None = None  # route: the chosen option index
    steer_text: str = ""  # confirm "changed": the text to steer in


def bind(q: Question | None, reply: s.Reply, now: float) -> Binding:
    if q is None or now >= q.expires_at:
        return Binding(False)
    if reply.reaction:
        if not (q.prompt_ref and reply.reaction_on == q.prompt_ref and is_thumbs_up(reply.reaction)):
            return Binding(False)
        answer: str | None = "yes"
        text = ""
    else:
        text = (reply.text or "").strip()
        if not text:
            return Binding(False)
        if reply.quoted_ref:
            if not q.prompt_ref or reply.quoted_ref != q.prompt_ref:
                return Binding(False)
        elif q.kind is s.QuestionKind.FOLDER_APPROVAL:
            return Binding(False)
        answer = yes_no(text)
    k = q.kind
    if k is s.QuestionKind.CONFIRM:
        if answer == "yes":
            return Binding(True, s.ReplyOutcome.APPROVED)
        if answer == "no":
            return Binding(True, s.ReplyOutcome.DENIED)
        return Binding(True, s.ReplyOutcome.CHANGED, steer_text=text)
    if k in (s.QuestionKind.FOLDER_APPROVAL, s.QuestionKind.RESTART):
        if answer is None:
            return Binding(False)
        return Binding(True, s.ReplyOutcome.APPROVED if answer == "yes" else s.ReplyOutcome.DENIED)
    if k is s.QuestionKind.CATCH_UP:
        if answer is None:
            return Binding(False)
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=0 if answer == "yes" else 1)
    # route: an option's key words, or yes/no for a two-way question
    options = q.payload.get("options", [])
    if answer is not None and len(options) == 2:
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=0 if answer == "yes" else 1)
    low = text.lower()
    hits = [i for i, o in enumerate(options)
            if any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", low) for k in o.get("keys", []))]
    if len(hits) == 1:
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=hits[0])
    return Binding(False)
