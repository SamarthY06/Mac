"""Classify brain failures into BrainErrorKind + a short spoken message (ARCHITECTURE §5)."""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any

from macd.contracts import BrainErrorKind


@dataclass(frozen=True)
class ClassifiedError:
    kind: BrainErrorKind
    message: str  # speakable
    detail: str = ""  # for logs

    def as_data(self) -> dict[str, Any]:
        return {"detail": self.detail[:2000]}


MESSAGES = {
    BrainErrorKind.AUTH: "I'm not signed in to Claude. Open Terminal, run claude, and sign in, then ask me again.",
    BrainErrorKind.RATE_LIMIT: "I've hit the Claude usage limit. Please try again later.",
    BrainErrorKind.NETWORK: "I can't reach Claude right now. Please check the internet connection.",
    BrainErrorKind.CLI_MISSING: "I can't find Claude Code on this Mac. Please install it, then try again.",
    BrainErrorKind.TIMEOUT: "That was taking too long, so I stopped.",
    BrainErrorKind.INTERNAL: "Something went wrong on my side. Please try again.",
    BrainErrorKind.MODEL: ("The Claude model set for this is not available. Pick another model, or go back to "
                           "the default, in Settings."),
}
CONNECTION_LOST = "I lost my connection to Claude. Please try again."

_AUTH = re.compile(
    r"not logged in|please run /?login|invalid api key|invalid x-api-key|authentication[_ ]"
    r"(failed|error)|oauth token (has )?expired|token (has )?expired|unauthori[sz]ed|\b401\b|"
    r"credentials? (are )?(missing|invalid)|login required|api key not found",
    re.IGNORECASE,
)
_RATE = re.compile(
    r"rate[_ -]?limit|usage limit|limit reached|\b429\b|too many requests|quota|"
    r"credit balance is too low|billing[_ ]error|out of (extra )?usage",
    re.IGNORECASE,
)
_NETWORK = re.compile(
    r"enotfound|econnrefused|econnreset|etimedout|eai_again|getaddrinfo|network|"
    r"connection (error|refused|reset)|socket hang up|fetch failed|unable to connect|"
    r"overloaded|\b529\b|\b50[234]\b|server_error|ssl|certificate",
    re.IGNORECASE,
)
_TIMEOUT = re.compile(r"timed? ?out|timeout", re.IGNORECASE)
_EPOCH = re.compile(r"\|(\d{10})\b")
_RESETS = re.compile(r"resets?\s+(?:at\s+)?([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?(?:\s*\([^)]*\))?)", re.IGNORECASE)


def _speak_time(ts: float) -> str:
    t = _dt.datetime.fromtimestamp(ts)
    hour = t.strftime("%I").lstrip("0") or "12"
    minute = "" if t.minute == 0 else t.strftime(":%M")
    return f"{hour}{minute} {t.strftime('%p')}"


def reset_time(text: str, resets_at: float | None = None) -> str | None:
    """Extract a speakable reset time from a limit message (epoch suffix or 'resets 3pm')."""
    if resets_at:
        return _speak_time(float(resets_at))
    m = _EPOCH.search(text or "")
    if m:
        return _speak_time(int(m.group(1)))
    m = _RESETS.search(text or "")
    if m:
        return re.sub(r"\s*\([^)]*\)", "", m.group(1)).strip()
    return None


def _rate_message(text: str, resets_at: float | None = None) -> str:
    if re.search(r"credit balance is too low|billing", text, re.IGNORECASE):
        return "My Claude account is out of credits. Please top it up, then try again."
    when = reset_time(text, resets_at)
    if when:
        return f"I've hit the Claude usage limit. Try again at {when}."
    return MESSAGES[BrainErrorKind.RATE_LIMIT]


def classify_text(text: str, default: BrainErrorKind = BrainErrorKind.INTERNAL,
                  resets_at: float | None = None) -> ClassifiedError:
    """Classify free text (stderr, result text, exception message)."""
    t = text or ""
    if _AUTH.search(t):
        return ClassifiedError(BrainErrorKind.AUTH, MESSAGES[BrainErrorKind.AUTH], t)
    if _RATE.search(t):
        return ClassifiedError(BrainErrorKind.RATE_LIMIT, _rate_message(t, resets_at), t)
    if re.search(r"command not found|no such file or directory.*claude|claude code not found|"
                 r"cli not found", t, re.IGNORECASE):
        return ClassifiedError(BrainErrorKind.CLI_MISSING, MESSAGES[BrainErrorKind.CLI_MISSING], t)
    if _NETWORK.search(t):
        msg = ("Claude is overloaded right now. Please try again in a minute."
               if re.search(r"overloaded|\b529\b", t, re.IGNORECASE) else MESSAGES[BrainErrorKind.NETWORK])
        return ClassifiedError(BrainErrorKind.NETWORK, msg, t)
    if _TIMEOUT.search(t):
        return ClassifiedError(BrainErrorKind.TIMEOUT, MESSAGES[BrainErrorKind.TIMEOUT], t)
    return ClassifiedError(default, MESSAGES[default], t)


def classify_exception(exc: BaseException, stderr_tail: str = "") -> ClassifiedError:
    """Classify an exception raised by the SDK/CLI, using recent stderr for context."""
    name = type(exc).__name__
    text = f"{name}: {exc}\n{getattr(exc, 'stderr', '') or ''}\n{stderr_tail}"
    if name == "CLINotFoundError" or (isinstance(exc, FileNotFoundError) and "claude" in str(exc)):
        return ClassifiedError(BrainErrorKind.CLI_MISSING, MESSAGES[BrainErrorKind.CLI_MISSING], text)
    if isinstance(exc, TimeoutError):
        return ClassifiedError(BrainErrorKind.TIMEOUT, MESSAGES[BrainErrorKind.TIMEOUT], text)
    status = getattr(exc, "api_error_status", None)
    if status == 401:
        return ClassifiedError(BrainErrorKind.AUTH, MESSAGES[BrainErrorKind.AUTH], text)
    if status == 429:
        return ClassifiedError(BrainErrorKind.RATE_LIMIT, _rate_message(text), text)
    if isinstance(exc, (ConnectionError, OSError)) and not isinstance(exc, FileNotFoundError):
        return ClassifiedError(BrainErrorKind.NETWORK, MESSAGES[BrainErrorKind.NETWORK], text)
    return classify_text(text)


_ASSISTANT_ERRORS = {
    "authentication_failed": BrainErrorKind.AUTH,
    "billing_error": BrainErrorKind.RATE_LIMIT,
    "rate_limit": BrainErrorKind.RATE_LIMIT,
    "server_error": BrainErrorKind.NETWORK,
}


def classify_assistant_error(code: str, text: str = "") -> ClassifiedError:
    """Classify AssistantMessage.error ("authentication_failed", "rate_limit", ...)."""
    kind = _ASSISTANT_ERRORS.get(code)
    if kind is None:
        return classify_text(f"{code} {text}")
    if kind is BrainErrorKind.RATE_LIMIT:
        return ClassifiedError(kind, _rate_message(f"{code} {text}"), f"{code} {text}")
    return ClassifiedError(kind, MESSAGES[kind], f"{code} {text}")


def classify_result(result: Any, stderr_tail: str = "") -> ClassifiedError | None:
    """Classify a ResultMessage; None if it is not an error (or was an interrupt)."""
    if not getattr(result, "is_error", False):
        return None
    if getattr(result, "terminal_reason", None) in ("aborted_streaming", "aborted_tools"):
        return None
    status = getattr(result, "api_error_status", None)
    errors = getattr(result, "errors", None) or []
    text = " ".join([str(getattr(result, "result", "") or ""), *map(str, errors),
                     str(getattr(result, "subtype", "")), stderr_tail])
    if status == 401:
        return ClassifiedError(BrainErrorKind.AUTH, MESSAGES[BrainErrorKind.AUTH], text)
    if status == 429:
        return ClassifiedError(BrainErrorKind.RATE_LIMIT, _rate_message(text), text)
    if status in (500, 502, 503, 504, 529):
        return classify_text(f"{text} {status}", BrainErrorKind.NETWORK)
    if status == 404 and re.search(r"\bmodel\b", text, re.IGNORECASE):  # Claude Code: "issue with the selected model"
        return ClassifiedError(BrainErrorKind.MODEL, MESSAGES[BrainErrorKind.MODEL], text)
    if getattr(result, "subtype", "") == "error_max_turns":
        return ClassifiedError(BrainErrorKind.INTERNAL,
                               "That needed more steps than I'm allowed, so I stopped.", text)
    return classify_text(text)
