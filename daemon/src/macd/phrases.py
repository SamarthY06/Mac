"""Everything Mac says on its own, and how it reads short human replies.

Pure functions and constants only (no I/O), tested table-driven in tests/test_orchestrator.py.
"""

from __future__ import annotations

import re

# ----------------------------------------------------------------------------- spoken lines

FILLER_ACTION = ("On it.", "Okay.")
FILLER_QUESTION = "Let me think."
FILLER_SHORT_QUESTION = "Sure."
SLOW_FILLER = "One moment."  # when the claude process is asleep: covers the connect latency
STEER_ACK = "Got it."
STOPPED = "Okay, stopped."
DENIED = "Okay, I won't."
CONFIRM_TIMEOUT = "No answer, so I won't."
GOING_OFF = "Okay, going to sleep. Click the menu bar icon to wake me."
STILL_WORKING = "Still working on it."
HEARTBEATS = ("Still on it.", "This is taking a while — I'll keep going.")
PHONE_CONNECTED = "Phone connected."
PHONE_PREEMPTED = "Someone joined from the phone."
PIN_PROMPT = "Please say the four-digit code shown on the Mac."
PIN_OK = "Thanks, go ahead."
PIN_WRONG = "That's not the code. Please try again."
VOICE_LOST = "I lost my connection to Claude. Please try again."
CLONE_LOAD_FAILED = "I couldn't load the cloned voice, using my default voice."

SLOW_BLIP_S = 3.0  # soft tick if the brain is still silent this long after the turn started
SLOW_SAY_S = 7.0  # "Still working on it." if nothing at all by then
HEARTBEAT_EVERY_S = 20.0
NARRATION_REPEAT_S = 45.0  # the same tool line may be repeated ("Still …") after this long

PREWARM_PHRASES: tuple[str, ...] = (
    *FILLER_ACTION, FILLER_QUESTION, FILLER_SHORT_QUESTION, SLOW_FILLER, STEER_ACK, STOPPED, DENIED,
    CONFIRM_TIMEOUT, GOING_OFF, STILL_WORKING, *HEARTBEATS, PHONE_CONNECTED,
    PHONE_PREEMPTED, PIN_PROMPT, PIN_OK, PIN_WRONG, VOICE_LOST, CLONE_LOAD_FAILED,
)


# ----------------------------------------------------------------------------- confirmation line

_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_APP_TARGET_RE = re.compile(r"^(?P<app>[A-Za-z][\w .'&-]*?)\s*:\s*(?P<rest>.*)$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().-]{6,}$")
_HOST_RE = re.compile(r"^(?:https?://)?(?P<host>[a-z0-9-]+(?:\.[a-z0-9-]+)+)(?:[/:?#].*)?$", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"\s+(?:saying|that says|which says|with the text|reading|containing)\s+|[,;:]\s|\s[—–-]\s")
_SEND_RE = re.compile(r"\b(send|sends|message|text|email|mail|reply|whatsapp|dm|tell|forward)\b")
_POST_RE = re.compile(r"\b(post|publish|tweet|share|comment)\b")
_DELETE_RE = re.compile(r"\b(delete|remove|erase|discard|clear|unsubscribe)\b")
_PAY_RE = re.compile(r"\b(pay|buy|order|purchase|checkout|transfer|book|subscribe)\b")
MAX_SUMMARY_WORDS = 25
_TRAILING_PUNCT_RE = re.compile(r"[.?!]+(['\"”’)\]]*)$")  # drop the final period, even inside closing quotes


def visible(text: str) -> str:
    """Strip bidi marks and other invisible characters (never speak them), collapse whitespace."""
    return " ".join(_INVISIBLE_RE.sub("", text).split())


def humanize_target(target: str) -> tuple[str, str]:
    """(core, spoken phrase) for a gate target. The raw target is never spoken.

    "WhatsApp:AXButton Send"  → ("whatsapp", "in WhatsApp")
    "Slack: Send"             → ("slack", "in Slack")
    "linkedin.com"            → ("linkedin.com", "on linkedin.com")
    "bob@example.com"         → ("bob@example.com", "to bob@example.com")
    """
    t = visible(target)
    if not t:
        return "", ""
    m = _APP_TARGET_RE.match(t)
    if m and not _EMAIL_RE.match(t) and not _HOST_RE.match(t):
        app = m.group("app").strip()
        return app.lower(), f"in {app}"
    if _EMAIL_RE.match(t):
        return t.lower(), f"to {t}"
    if _PHONE_RE.match(t):
        return t, f"to {t}"
    m = _HOST_RE.match(t)
    if m:
        host = m.group("host").lower().removeprefix("www.")
        return host, f"on {host}"
    return t.lower(), f"to {t}"


def confirm_question(summary: str) -> str:
    s = summary.lower()
    if _SEND_RE.search(s):
        return "Send it?"
    if _POST_RE.search(s):
        return "Post it?"
    if _DELETE_RE.search(s):
        return "Delete it?"
    if _PAY_RE.search(s):
        return "Go ahead and pay?"
    return "Go ahead?"


def confirmation_line(summary: str, target: str = "") -> str:
    """The one spoken confirmation: "<statement>[ <where>]. <action question>".

    Long summaries are cut to their first clause; the target is added only when the summary
    doesn't already mention it (case-insensitive)."""
    s = _TRAILING_PUNCT_RE.sub(r"\1", visible(summary).strip()).strip() or "I'm about to take an action"
    if len(s.split()) > MAX_SUMMARY_WORDS:
        first = _CLAUSE_SPLIT_RE.split(s, maxsplit=1)[0].strip().rstrip(".?!,;:")
        s = first if first else s
        if len(s.split()) > MAX_SUMMARY_WORDS:
            s = " ".join(s.split()[:MAX_SUMMARY_WORDS])
    core, phrase = humanize_target(target)
    if phrase and core and core not in s.lower():
        s = f"{s} {phrase}"
    return f"{s}. {confirm_question(summary)}"


def elapsed_phrase(seconds: float) -> str:
    """'Two minutes in, still going.' — spoken every 45 s once the heartbeat list is exhausted."""
    minutes = max(1, round(seconds / 60))
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
             9: "Nine", 10: "Ten"}
    n = words.get(minutes, str(minutes))
    return f"{n} minute{'s' if minutes != 1 else ''} in, still going."


# ----------------------------------------------------------------------------- parsing


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\bok\b", "okay", t)
    return " ".join(t.split())


STOP_PHRASES = frozenset(
    {"stop", "cancel", "never mind", "nevermind", "quiet", "shut up", "that's enough", "thats enough",
     "be quiet", "stop it", "stop talking", "enough", "cancel that"}
)
_TRAILERS = frozenset({"please", "now", "it", "that", "talking", "mac", "thanks", "thank", "you", "okay"})
_LEADERS = frozenset({"okay", "mac", "hey", "no", "please"})


def _strip_edges(n: str, leaders: frozenset[str], trailers: frozenset[str]) -> list[str]:
    """All phrases reachable by stripping leader/trailer words (e.g. 'okay stop please' → 'stop')."""
    words = n.split()
    out = [n]
    while words and words[0] in leaders:
        words = words[1:]
        out.append(" ".join(words))
    while words and words[-1] in trailers:
        words = words[:-1]
        out.append(" ".join(words))
    return out


def is_stop(text: str) -> bool:
    n = normalize(text)
    return bool(n) and any(p in STOP_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS))


OFF_PHRASES = frozenset(
    {"go to sleep", "turn off", "turn yourself off", "stop listening", "power off", "switch off", "shut down"}
)


def is_off_request(text: str) -> bool:
    n = normalize(text)
    return bool(n) and any(p in OFF_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS - {"it", "that"}))


VOICE_RESET_PHRASES = frozenset(
    {"use your default voice", "use the default voice", "default voice", "go back to your default voice",
     "switch to your default voice", "reset your voice", "use your normal voice", "use your own voice",
     "your default voice", "back to your default voice"}
)


_RENAME_RE = re.compile(
    r"^(?:(?:please|okay|ok|hey|mac)\s+)*(?:call yourself|change your name to|your name is now|rename yourself(?: to)?|"
    r"from now on you're|from now on you are|your new name is|i'll call you|i will call you)\s+(?P<name>[a-z][a-z0-9 -]{0,31}?)"
    r"(?:\s+(?:please|now|from now on|okay))*[.!]?$",
    re.IGNORECASE,
)


def rename_request(text: str) -> str | None:
    """'Hey Mac, call yourself Jarvis' → 'Jarvis'; None if not a rename."""
    m = _RENAME_RE.match(" ".join(re.sub(r"[,!;:]", " ", text).split()))
    if not m:
        return None
    name = m.group("name").strip().strip(".!").strip()
    return name.title() if name else None


def is_voice_reset_request(text: str) -> bool:
    n = normalize(text)
    return bool(n) and any(p in VOICE_RESET_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS))


_NEG = re.compile(
    r"\b(no|nope|don'?t|not|never|cancel|stop|wait|hold on|but|instead|change|actually|after|once|before|first|"
    r"unless|except|only if)\b"
)
_AFF = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|go ahead|do it|send it|confirm|looks good|sounds good|go for it|"
    r"please do|absolutely|correct|that's right)\b"
)


def classify_confirmation(text: str) -> str:
    """'yes' | 'no' | 'steer' for a reply to "<summary>. Shall I go ahead?".

    yes:   an affirmative with no negation ("yes", "looks good, send it", "okay do it")
    no:    a bare negative or anything short and unclear ("no", "nope", "hmm")
    steer: a longer reply that is not a plain yes ("yes but change the subject", "no, send it to Bob")
           — the orchestrator denies AND forwards it to the model.
    """
    n = normalize(text)
    if not n:
        return "no"
    if _AFF.search(n) and not _NEG.search(n):
        return "yes"
    return "no" if len(n.split()) <= 3 else "steer"


_CORRECTION_RE = re.compile(
    r"^(?:actually|no|nope|scratch that|i meant|instead)\b[\s,.;:!-]+(?P<rest>\S.*)$", re.IGNORECASE
)
_AGREEMENT_RE = re.compile(r"^(that|it|this|you)('s|'re| is| was|s)?\b", re.IGNORECASE)


def correction_text(text: str) -> str | None:
    """'Actually, make it Friday' → 'make it Friday'.

    None when the lead-in is not followed by an alternative instruction: "no, that's right",
    "actually never mind" (a stop), "wait for the page to load" (not a lead-in at all) — those are
    steers or stops, never a cancel + redirect.
    """
    m = _CORRECTION_RE.match(text.strip())
    if not m:
        return None
    rest = m.group("rest").strip()
    if len(rest.split()) < 2 or _AGREEMENT_RE.match(rest) or is_stop(rest):
        return None
    return rest


#: One-word replies that are a real follow-up (vs. a bystander fragment like "Then").
SHORT_INTENTS = frozenset(
    {"continue", "next", "again", "why", "more", "repeat", "what", "how", "when", "where", "who", "yes", "yeah",
     "yep", "yup", "no", "nope", "okay", "sure", "correct", "right", "wrong", "done", "thanks", "skip", "back",
     "undo", "pause", "louder", "quieter", "slower", "faster", "hello", "hi"}
)


def is_followup_intent(text: str) -> bool:
    """Is a wake-less follow-up utterance worth a turn? ≥ 2 words, or a known short intent
    (affirmative/negative, stop word, "continue"/"next"/…, a number)."""
    words = normalize(text).split()
    if not words:
        return False
    if len(words) >= 2:
        return True
    w = words[0]
    return w in SHORT_INTENTS or w.isdigit() or w in _DIGIT_WORDS or is_stop(w) or classify_confirmation(w) == "yes"


_QUESTION_RE = re.compile(
    r"^(what|who|whom|whose|when|where|why|how|which|is|are|am|was|were|can|could|do|does|did|"
    r"should|would|will|shall|have|has|tell me|explain|teach)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") or bool(_QUESTION_RE.match(t))


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
    "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def spoken_digits(text: str) -> str:
    """'one two three four' / '12 34' / 'one-two-three-four' → '1234' (all digits found, in order)."""
    out = []
    for w in normalize(text).split():
        if w.isdigit():
            out.append(w)
        elif w in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[w])
    return "".join(out)


def pin_matches(text: str, pin: str) -> bool:
    return pin in spoken_digits(text)
