"""Question binding (W3.2, M5, F14): agentd alone decides whether a chat reply answers the open
question. A conversation has at most one open question; anything that does not bind is ordinary
text and never an approval.

Binding: a 👍 whose `reaction_on` is the prompt; a text that quotes the prompt; or an unquoted
text while the question is open (never for `folder_approval`, which needs a quote or a 👍).

Yes/no stays deterministic (AG6): `folder_approval` takes only the exact words; `confirm`,
`restart` and `catch_up` also take the voice rule (`macd.phrases.classify_confirmation`: an
affirmative with no negation is a yes). A `route`, `catch_up` or `restart` reply that is neither
closes the question as `answered` with the text (AG4): the model reads it, nothing waits an hour.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from agentd import schema as s
from macd.phrases import classify_confirmation, normalize

YES = frozenset({"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "send it", "go ahead", "do it",
                 "go", "yes please", "please do", "approve", "approved", "confirm", "sounds good"})
NO = frozenset({"no", "n", "nope", "cancel", "don't", "dont", "do not", "no thanks", "don't send it",
                "dont send it", "deny", "skip"})
# a change to what Mac may do or where: a quoted yes or a 👍 on the prompt only, exact words (M5, N7)
_QUOTE_ONLY = frozenset({s.QuestionKind.FOLDER_APPROVAL, s.QuestionKind.PROFILE})
_THUMBS = re.compile("^\U0001F44D[\U0001F3FB-\U0001F3FF]?️?$")


_NEG_LEAD = frozenset({"no", "nope", "nah", "don't", "dont", "cancel", "stop", "not", "never"})


def yes_no(text: str, *, strict: bool = True) -> str | None:
    """"yes", "no" or None (any other text). `strict`: the exact word sets only (folder approval);
    otherwise also the voice rule: an affirmative with no negation ("Yes, go ahead", "sure,
    thanks"), and a short reply that leads with a negative ("no not now")."""
    t = unicodedata.normalize("NFKC", text).strip().lower().replace("’", "'")
    t = re.sub(r"[.!?,]+$", "", t).strip()
    if _THUMBS.match(t):
        return "yes"
    if t in YES:
        return "yes"
    if t in NO:
        return "no"
    if strict:
        return None
    if classify_confirmation(t) == "yes":
        return "yes"
    words = normalize(t).split()
    if words and len(words) <= 3 and words[0] in _NEG_LEAD:
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
    choice: int | None = None  # route: the chosen option index (None with `steer_text`: free text)
    steer_text: str = ""  # confirm "changed", or a free-text answer (AG4): the user's words


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
        elif q.kind in _QUOTE_ONLY:
            return Binding(False)
        answer = yes_no(text, strict=q.kind in _QUOTE_ONLY)
    k = q.kind
    if k is s.QuestionKind.CONFIRM:
        if answer == "yes":
            return Binding(True, s.ReplyOutcome.APPROVED)
        if answer == "no":
            return Binding(True, s.ReplyOutcome.DENIED)
        return Binding(True, s.ReplyOutcome.CHANGED, steer_text=text)
    if k in _QUOTE_ONLY:
        if answer is None:
            return Binding(False)
        return Binding(True, s.ReplyOutcome.APPROVED if answer == "yes" else s.ReplyOutcome.DENIED)
    if k is s.QuestionKind.RESTART:
        if answer is None:
            return Binding(True, s.ReplyOutcome.ANSWERED, steer_text=text)
        return Binding(True, s.ReplyOutcome.APPROVED if answer == "yes" else s.ReplyOutcome.DENIED)
    if k is s.QuestionKind.CATCH_UP:
        if answer is None:
            return Binding(True, s.ReplyOutcome.ANSWERED, steer_text=text)
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=0 if answer == "yes" else 1)
    # route and move (P2-1): an option only when the whole cleaned reply is exactly one of its
    # keys (label or number) or an exact yes/no for the option "yes" means (AG3); any other
    # text, "yes, but in Notes" included, is a free-text answer for the router (AG4)
    options = q.payload.get("options", [])
    yes = q.payload.get("yes")
    exact = yes_no(text) if text else answer
    if exact is not None and len(options) == 2 and isinstance(yes, int) and yes in (0, 1):
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=yes if exact == "yes" else 1 - yes)
    if not text:
        return Binding(False)  # a reaction alone cannot pick between options
    clean = _clean(text)
    hits = [i for i, o in enumerate(options) if clean in {_clean(str(k)) for k in o.get("keys", [])}]
    if len(hits) == 1:
        return Binding(True, s.ReplyOutcome.ANSWERED, choice=hits[0])
    return Binding(True, s.ReplyOutcome.ANSWERED, steer_text=text)


def _clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    t = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", t)
    return " ".join(t.split())
