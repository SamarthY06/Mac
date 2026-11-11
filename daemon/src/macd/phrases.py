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
HEARTBEATS = ("Still on it.", "This is taking a while. I'll keep going.")
STOPPED_ALL = "Stopped everything."
PHONE_CONNECTED = "Phone connected."
PHONE_PREEMPTED = "Someone joined from the phone."
PIN_PROMPT = "Please say the four-digit code shown on the Mac."
PIN_OK = "Thanks, go ahead."
PIN_WRONG = "That's not the code. Please try again."
VOICE_LOST = "I lost my connection to Claude. Please try again."
CLONE_LOAD_FAILED = "I couldn't load the cloned voice, using my default voice."
STOPPED_OTHER = "Okay, I stopped the other task."  # "Hey Mac, stop the other task" (W6)
NO_OTHER_TASK = "Nothing else is running."
STOP_OTHER_FAILED = "I couldn't reach the other task. Use Stop in the Mac menu."

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
CONFIRM_FALLBACK_QUESTION = "Shall I go ahead?"  # when the model gave no question (AG18)
_TRAILING_PUNCT_RE = re.compile(r"[.?!]+(['\"”’)\]]*)$")  # drop the final period, even inside closing quotes


_DASH_RE = re.compile(r"\s*[—–]\s*|\s+-\s+")


def dedash(text: str) -> str:
    """No em/en dashes are ever spoken: "a — b" / "a – b" / "a - b" → "a, b"."""
    return _DASH_RE.sub(", ", text).replace(",, ", ", ")


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


def confirmation_line(summary: str, target: str = "", question: str | None = None) -> str:
    """The one spoken confirmation: "<summary>[ <where>]. <question>".

    The summary and question are the model's (its tool checks their shape). Kept deterministic:
    invisible characters out, the hook-derived target added when the summary doesn't mention it,
    no dashes, and a neutral question when the model gave none."""
    s = _TRAILING_PUNCT_RE.sub(r"\1", visible(summary).strip()).strip() or "I'm about to take an action"
    core, phrase = humanize_target(target)
    if phrase and core and core not in s.lower():
        s = f"{s} {phrase}"
    q = visible(question or "").strip() or CONFIRM_FALLBACK_QUESTION
    return dedash(f"{s}. {q}")


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


STOP_ALL_PHRASES = frozenset(
    {"stop everything", "stop all", "stop it all", "hard stop", "kill everything", "stop and reset", "stop all tasks",
     "stop everything now", "stop everything and start fresh"}
)


def is_stop_all(text: str) -> bool:
    """"Stop everything": the turn, the background agents and the session, at once."""
    n = normalize(text)
    return bool(n) and any(p in STOP_ALL_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS))


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


STOP_OTHER_PHRASES = frozenset(
    {"stop the other task", "stop the other one", "stop the other thing", "cancel the other task",
     "stop whatsapp", "stop the whatsapp task", "cancel whatsapp", "stop the background task"}
)


def is_stop_other(text: str) -> bool:
    """"Hey Mac, stop the other task" / "stop WhatsApp": cancel the running chat turns (W6)."""
    n = normalize(text)
    return bool(n) and any(p in STOP_OTHER_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS))


# ----------------------------------------------------------------------------- barge-in (talking over Mac)

#: Listener feedback that must NOT stop Mac (Vapi "acknowledgementPhrases", LiveKit/Pipecat backchannels).
BACKCHANNEL_PHRASES = frozenset(
    {"mm", "mmm", "mm hmm", "mhm", "mmhmm", "hmm", "hm", "uh huh", "uhhuh", "uh", "um", "ah", "oh", "ooh", "wow",
     "yeah", "yes", "yep", "yup", "yah", "okay", "alright", "all right", "right", "sure", "cool", "nice", "great",
     "good", "true", "exactly", "perfect", "interesting", "got it", "gotcha", "i see", "makes sense", "fair enough",
     "of course", "totally", "indeed", "thanks", "thank you", "go on", "keep going", "uh hm"}
)
#: Short words that DO interrupt despite being under the word minimum (Vapi "interruptionPhrases").
INTERJECTION_PHRASES = frozenset(
    {"wait", "hold on", "hang on", "one sec", "one second", "pause", "excuse me", "sorry", "sorry what", "what",
     "pardon", "huh", "hey"}
)


def _covered_by(words: list[str], phrases: frozenset[str]) -> bool:
    """Can `words` be split entirely into `phrases` ("yeah okay got it" → yes)?"""
    if not words:
        return True
    return any(" ".join(words[:k]) in phrases and _covered_by(words[k:], phrases) for k in range(min(3, len(words)), 0, -1))


#: Short appraisals over Mac's speech ("that's a good idea", "it's really pretty", "I love it") are the
#: listener agreeing, not taking the turn (live 2026-11-12: each one cancelled a long answer and
#: started a new turn on the compliment). Glue words carry no meaning on their own; a cue word
#: (an objection, a question, an instruction) makes it an interruption whatever else it contains.
_APPRAISALS = frozenset(
    {"good", "great", "nice", "pretty", "cool", "awesome", "perfect", "interesting", "smart", "clever", "helpful",
     "useful", "lovely", "brilliant", "excellent", "fantastic", "amazing", "fine", "right", "true", "correct", "fair",
     "sensible", "reasonable", "solid", "clear", "impressive", "beautiful", "neat", "sweet", "love", "like", "agree",
     "agreed", "sure", "okay", "yes", "yeah", "wonderful", "superb", "best", "better", "promising", "strong"}
)
_GLUE = frozenset(
    {"that", "thats", "that's", "it", "its", "it's", "this", "is", "a", "an", "the", "really", "so", "very", "quite",
     "one", "idea", "point", "plan", "approach", "way", "thought", "answer", "i", "to", "me", "sounds", "seems",
     "looks", "oh", "ah", "yeah", "yes", "okay", "all", "of", "them", "those", "these", "actually"})
_CUES = frozenset(
    {"stop", "wait", "no", "not", "dont", "don't", "never", "instead", "but", "what", "why", "how", "when", "where",
     "which", "who", "can", "could", "would", "you", "please", "make", "do", "send", "open", "find", "change", "use",
     "try", "lets", "let's", "should", "want", "need", "skip", "repeat", "again", "wrong", "hold"})


def is_backchannel(text: str) -> bool:
    """"mm-hmm", "yeah", "okay got it", "that's a good idea": the listener is following or agreeing,
    not taking the turn."""
    words = normalize(text).replace("-", " ").split()
    if not words or len(words) > 6:
        return False
    if _covered_by(words, BACKCHANNEL_PHRASES):
        return True
    if any(w in _CUES for w in words):
        return False
    rest = [w for w in words if w not in _GLUE]
    return bool(rest) and all(w in _APPRAISALS for w in rest)


def is_interjection(text: str) -> bool:
    n = normalize(text)
    return bool(n) and any(p in INTERJECTION_PHRASES for p in _strip_edges(n, _LEADERS, _TRAILERS))


def word_count(text: str) -> int:
    return len(normalize(text).split())


def is_echo(heard: str, spoken: str) -> bool:
    """Residual echo of Mac's own voice that leaked through AEC: (nearly) every heard word is
    one Mac was just saying."""
    words = normalize(heard).split()
    said = set(normalize(spoken).split())
    return bool(words) and bool(said) and sum(w in said for w in words) / len(words) >= 0.8


_NEG = re.compile(
    r"\b(no|nope|don'?t|not|never|cancel|stop|wait|hold on|but|instead|change|actually|after|once|before|first|"
    r"unless|except|only if)\b"
)
_AFF = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|go ahead|do it|send it|confirm|looks good|sounds good|go for it|"
    r"please do|absolutely|correct|that's right)\b"
)


def has_negation(text: str) -> bool:
    """An explicit negative or redirect word ("no", "not", "instead", "wait") is present."""
    return bool(_NEG.search(normalize(text)))


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


#: One-word replies that are a real follow-up (vs. a bystander fragment like "Then").
SHORT_INTENTS = frozenset(
    {"continue", "next", "again", "why", "more", "repeat", "what", "how", "when", "where", "who", "yes", "yeah",
     "yep", "yup", "no", "nope", "okay", "sure", "correct", "right", "wrong", "done", "thanks", "skip", "back",
     "undo", "pause", "louder", "quieter", "slower", "faster", "hello", "hi"}
)


def is_english(lang: str) -> bool:
    """The fixed fillers and the follow-up word list are English (AG19); `tts.lang` is e.g. "en-us"."""
    return lang.strip().lower().startswith("en")


def is_followup_intent(text: str, *, english: bool = True) -> bool:
    """Is a wake-less follow-up utterance worth a turn? ≥ 2 words, or a known short intent
    (affirmative/negative, stop word, "continue"/"next"/…, a number). The short-intent list is
    English, so in another language a one-word reply ("oui") is let through rather than dropped."""
    words = normalize(text).split()
    if not words:
        return False
    if len(words) >= 2 or not english:
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
