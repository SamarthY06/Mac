"""Chat text: the `Mac · ` prefix, Markdown to the channel's format, no dashes, whole minutes,
long answers, and every fixed string chatd sends (docs/whatsapp/ARCHITECTURE.md W5.3, W5.5).

Every outbound text passes `outbound()`: prefixed, converted and free of em and en dashes.
"""

from __future__ import annotations

import re

from macd.phrases import CONFIRM_FALLBACK_QUESTION

# The echo prefix: a protocol constant shared with wa-bridge (`EchoPrefix` in
# bridges/whatsapp/internal/bridge/filter.go; a test keeps the two equal). Every text Mac writes
# starts with it, and inbound text starting with it is an echo and is dropped.
PREFIX = "Mac · "
MAX_CHUNKS = 3  # longer than this many messages: the full text goes out as a .md file

# -- fixed strings (W5.3, W5.5; verbatim where the design gives them) -----------------------------
HELP = (PREFIX + "Text me what you'd do on your Mac.\n"
        '"stop" stops the current task.\n'
        '"new chat" starts fresh. "new chat in {folder}" picks a folder.\n'
        '"status" shows what I\'m doing.\n'
        "No 👀 on your message means your Mac is asleep or offline.")
# a prefix chat (every shared chat): a bare "stop" is ignored there, so help says the word (UX7)
HELP_PREFIX = (PREFIX + 'Start each message with "{w}", like "{w}, find me 20 jobs".\n'
               '"{w} stop" stops the current task.\n'
               '"{w} new chat" starts fresh. "{w} new chat in {folder}" picks a folder.\n'
               '"{w} status" shows what I\'m doing.')
HELP_FOLDER = "Documents"  # the example folder when the chat has no default workspace
VOICE_NOTE = f"{PREFIX}I only read text here."
STOPPED = f"{PREFIX}Stopped."
NOTHING_TO_STOP = f"{PREFIX}Nothing is running."
FRESH = f"{PREFIX}Fresh start."
FRESH_IN = PREFIX + "Fresh start in {folder}."
STOP_FIRST = f"{PREFIX}Say stop first, then new chat."
LOOP_PAUSED = (f"{PREFIX}I paused myself because I seemed to be replying to my own messages. "
               "Unpause in the Mac menu.")
ON_IT = f"{PREFIX}On it"
DONE = f"{PREFIX}Done."
SKIPPED = f"{PREFIX}Skipped."
# Confirmation outcomes (AG22): neutral, never claiming the action ran. The answer says what happened.
APPROVED = f"{PREFIX}Okay, going ahead."
DECLINED = f"{PREFIX}Okay, I won't."
CHANGING = f"{PREFIX}Okay, changing it."
NO_ANSWER = f"{PREFIX}No answer, so I didn't. Say yes and I will."
NO_ANSWER_LEFT = f"{PREFIX}No answer, so I left it."  # a folder, route, profile or restart question (UX8)
CONFIRM_QUESTION = CONFIRM_FALLBACK_QUESTION  # the one copy is macd.phrases (UX N5)
RESUME_FAILED = "I couldn't reopen that chat, so I started a fresh one."
DESKTOP_WAIT = "Waiting for the browser, your voice request is using it"
HELLO = f'{PREFIX}Mac here. Text me anything you\'d do on your Mac, like "find me 20 jobs".'
GOODBYE = f"{PREFIX}Unlinked from this Mac. Link again in Mac's Settings."
DESKTOP_HELD = "The browser is in use by your voice request."
FULL_ANSWER = "My answer was too long for a message, so it's in the file below."

# per-chat profile commands (W9.3): agentd asks the question and words its outcome
# (status{profile_command}); these are chatd's replies before agentd is involved
GROUPS_UNAVAILABLE = f"{PREFIX}I can't see your groups right now. Try again in a minute."
MANY_GROUPS = PREFIX + "More than one group is called \u201c{subject}\u201d. Rename one, then try again."
NEVER_LISTED = f"{PREFIX}I never answer in that chat."
PROFILE_EXISTS = f"{PREFIX}That group is already set up. Change it in the Mac app."
PROFILE_FAILED = f"{PREFIX}That didn't work. Change it in the Mac app."
CAPABILITY_OFF = PREFIX + "{name} is off in this chat."  # name: agentd's CAPABILITY_WORDS, capitalized


def capitalized(words: str) -> str:
    return words[:1].upper() + words[1:]

ERRORS = {
    "rate_limit": PREFIX + "I've hit the Claude usage limit.{reset} Say \"go\" after that and I'll carry on.",
    "auth": f'{PREFIX}Claude isn\'t signed in on your Mac. Open Terminal there and run "claude", then send this again.',
    "unknown": f'{PREFIX}Something went wrong on my side, so I stopped. Say "try again" to have another go.',
}

# -- dashes -------------------------------------------------------------------------------------------
_DASHES = "\u2012\u2013\u2014\u2015"  # figure, en, em, horizontal bar
_SPACED_DASH = re.compile(rf"\s*[{_DASHES}]\s+|\s+[{_DASHES}]\s*")
_RANGE_DASH = re.compile(rf"(?<=\w)[{_DASHES}](?=\w)")


def no_dashes(text: str) -> str:
    """Em and en dashes out: a spaced dash becomes ", " and a dash inside a range becomes "-"."""
    text = _SPACED_DASH.sub(", ", text)
    text = _RANGE_DASH.sub("-", text)
    return re.sub(rf"[{_DASHES}]", "-", text)


# -- Markdown -> channel format ------------------------------------------------------------------------
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_BULLET = re.compile(r"^(\s*)[*+]\s+")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\((\S+?)\)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(rows: list[str]) -> list[str]:
    """A Markdown table -> `label: value` lines (the header row only names the columns)."""
    body = [r for r in rows if not _TABLE_SEP.match(r)]
    has_header = len(rows) > 1 and _TABLE_SEP.match(rows[1]) is not None
    header = _cells(body[0]) if has_header and body else []
    out = []
    for r in body[1:] if has_header else body:
        cells = _cells(r)
        if len(cells) == 1:
            out.append(cells[0])
        elif len(cells) == 2 or not header:
            out.append(f"{cells[0]}: {', '.join(c for c in cells[1:] if c)}")
        else:
            rest = [f"{h} {c}".strip() if h else c for h, c in zip(header[1:], cells[1:], strict=False) if c]
            out.append(f"{cells[0]}: {', '.join(rest)}")
    return out


def _link(m: re.Match[str]) -> str:
    label, url = m.group(1), m.group(2)
    return url if label.strip() == url else f"{label} ({url})"


def render(text: str, text_format: str = "whatsapp") -> str:
    """Model Markdown -> the channel's format. `whatsapp`: *bold*, _italic_, headings bold, tables as
    `label: value` lines. `plain`: the same without emphasis markers. `markdown`: unchanged.
    Dashes are replaced in every format."""
    if text_format in ("markdown", "slack"):
        return no_dashes(text.strip())
    lines, out, table, fence = text.strip().splitlines(), [], [], False
    bold, italic = ("*", "_") if text_format == "whatsapp" else ("", "")

    def flush() -> None:
        if table:
            out.extend(_table(table))
            table.clear()

    for line in lines:
        if line.strip().startswith("```"):
            flush()
            fence = not fence
            out.append(line)
            continue
        if fence:
            out.append(line)
            continue
        if line.strip().startswith("|"):
            table.append(line)
            continue
        flush()
        if _RULE.match(line):
            continue
        h = _HEADING.match(line)
        if h:
            line = f"**{h.group(1)}**"
        line = _BULLET.sub(r"\1- ", line)
        line = _LINK.sub(_link, line)
        line = _BOLD.sub("\0\\2\0", line)
        line = _ITALIC.sub(f"{italic}\\1{italic}", line)
        out.append(line.replace("\0", bold))
    flush()
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return no_dashes(joined)


def prefixed(text: str) -> str:
    t = text.strip()
    return t if t.startswith(PREFIX) else PREFIX + t


def strip_prefix(text: str) -> str:
    t = text.strip()
    return t[len(PREFIX):].strip() if t.startswith(PREFIX) else t


def outbound(text: str) -> str:
    """The last step before any text leaves chatd: prefix and no dashes."""
    return no_dashes(prefixed(text))


# -- long answers ----------------------------------------------------------------------------------------
def _split(text: str, size: int) -> list[str]:
    chunks: list[str] = []
    rest = text
    while len(rest) > size:
        cut = rest.rfind("\n", 0, size)
        if cut < size // 2:
            cut = rest.rfind(" ", 0, size)
        if cut < size // 2:
            cut = size
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks


def plan_answer(answer: str, max_text: int) -> tuple[list[str], str | None]:
    """The answer as message texts (each prefixed), plus the full text for a .md file when it
    would take more than MAX_CHUNKS messages. The model is asked to keep chat answers short and
    put long results in a file itself (AG21); this is only the fallback, and it never invents a
    summary: the message is a fixed notice and the file holds the model's full text."""
    size = max(80, max_text - len(PREFIX) - 2)
    chunks = _split(answer, size) or [""]
    if len(chunks) <= MAX_CHUNKS:
        return [prefixed(c) for c in chunks], None
    return [prefixed(FULL_ANSWER)], answer


# -- time and short quotes ---------------------------------------------------------------------------
def minutes(seconds: float) -> int:
    return int(max(0.0, seconds) // 60)


def ago(seconds: float) -> str:
    m = minutes(seconds)
    if m < 1:
        return "just now"
    if m < 60:
        return f"{m} min ago"
    h = m // 60
    if h < 24:
        return "1 hour ago" if h == 1 else f"{h} hours ago"
    d = h // 24
    return "yesterday" if d == 1 else f"{d} days ago"


def short(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def folder_name(folder: str | None) -> str:
    """"~/Desktop/Personal/Resume" -> "Resume"; general (None or "") -> ""."""
    if not folder:
        return ""
    return folder.rstrip("/").rsplit("/", 1)[-1].lstrip("~") or ""


def rate_limit_line(message: str) -> str:
    m = re.search(r"\b(\d{1,2}:\d{2}\s?[AP]M)\b", message or "", re.IGNORECASE)
    return ERRORS["rate_limit"].format(reset=f" It resets at {m.group(1).upper()}." if m else "")


def error_line(kind: str, message: str = "") -> str:
    if kind == "rate_limit":
        return rate_limit_line(message)
    return ERRORS.get(kind, ERRORS["unknown"])
