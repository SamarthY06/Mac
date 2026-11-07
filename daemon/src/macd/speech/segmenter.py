"""Turn streamed (markdown-ish) model text into speakable sentences (ARCHITECTURE §8).

``SentenceSegmenter.feed(delta)`` returns the sentences completed by ``delta``;
``flush()`` returns whatever is left at the end of the turn. Sentences shorter than
``min_chars`` are merged with the next one so TTS isn't fed tiny fragments.

Markdown handling is line-oriented: headers and bullets become sentences, tables and
horizontal rules are skipped, fenced code is replaced (once) by a spoken placeholder.
Inline markup is stripped, URLs become hostnames, emoji are removed.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

CODE_PLACEHOLDER = "I've written that out in the reply."
MIN_CHARS = 20

# Never a sentence end when followed by a space (titles, Latin abbreviations).
_NEVER_END = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "e.g", "i.e", "cf",
    "approx", "dept", "est", "fig", "eg", "ie", "a.m", "p.m",
}
# Not a sentence end when followed by a digit ("No. 5", "Jan. 3", "p. 12").
_BEFORE_DIGIT = {
    "no", "nos", "vol", "p", "pp", "ch", "sec", "art", "jan", "feb", "mar", "apr", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec", "v", "ver",
}

_BOUNDARY = re.compile(r"(\.{3}|…|[.!?]+)([\"'”’)\]*_`]*)(\s+)(?=(\S))")
_TOKEN_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_ACRONYM = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_HRULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE = re.compile(r"^\s*\|")
_TABLE_SEP = re.compile(r"^\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?(.*)$")
_QUOTE = re.compile(r"^\s*>+\s?(.*)$")
# A partial line that might still turn into a structural markdown line.
_MAYBE_STRUCTURAL = re.compile(r"^\s*(?:$|[#|>`~]|[-*+•](?:\s|$)|\d{1,3}(?:[.)](?:\s|$)|$))")

_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_AUTOLINK = re.compile(r"<((?:https?://|www\.)[^>\s]+)>")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]\"']+")
_INLINE_CODE = re.compile(r"`{1,3}([^`]+)`{1,3}")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_STAR_EM = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_UNDER_EM = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")
_STRIKE = re.compile(r"~~(.+?)~~")
_HTML = re.compile(r"</?[A-Za-z][^>]*>")
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U0001f1e6-\U0001f1ff"
    "☀-➿"
    "⬀-⯿"
    "️‍⃣"
    "]+"
)
_STRAY = re.compile(r"(?<!\w)[*_#`]+|[*_#`]+(?!\w)")
_WS = re.compile(r"\s+")
_SPACE_PUNCT = re.compile(r"\s+([.,!?;:])(?=\s|$)")
_EXPAND = [(re.compile(r"\be\.g\.(?=\s|,|$)", re.IGNORECASE), "for example"), (re.compile(r"\bi\.e\.(?=\s|,|$)", re.IGNORECASE), "that is")]


def _hostname(url: str) -> str:
    u = url if "://" in url else "http://" + url
    try:
        host = urlsplit(u).hostname or url
    except ValueError:
        return url
    return host.removeprefix("www.")


def _url_sub(m: re.Match[str]) -> str:
    url = m.group(0)
    trail = ""
    while url and url[-1] in ".,;:!?":
        trail = url[-1] + trail
        url = url[:-1]
    return _hostname(url) + trail


def clean_inline(text: str) -> str:
    """Strip inline markdown/HTML/emoji, URLs -> hostnames, collapse whitespace."""
    t = _IMAGE.sub(r"\1", text)
    t = _LINK.sub(lambda m: m.group(1), t)
    t = _AUTOLINK.sub(lambda m: m.group(1), t)
    t = _INLINE_CODE.sub(r"\1", t)  # before URL so `https://x` still becomes a host
    t = _URL.sub(_url_sub, t)
    t = _BOLD.sub(r"\2", t)
    t = _STRIKE.sub(r"\1", t)
    t = _STAR_EM.sub(r"\1", t)
    t = _UNDER_EM.sub(r"\1", t)
    t = _HTML.sub(" ", t)
    t = _EMOJI.sub("", t)
    t = _STRAY.sub("", t)
    for rx, rep in _EXPAND:
        t = rx.sub(rep, t)
    t = _WS.sub(" ", t).strip()
    return _SPACE_PUNCT.sub(r"\1", t)


def _is_boundary(text: str, m: re.Match[str]) -> bool:
    punct, nxt = m.group(1), m.group(4)
    if punct[-1] in "!?" or punct in ("...", "…") and nxt[:1].isupper():
        return True
    if punct in ("...", "…"):
        return False
    if nxt.islower():
        return False
    tok_m = _TOKEN_BEFORE.search(text[: m.start()])
    if tok_m:
        tok = tok_m.group(1).rstrip(".")
        low = tok.lower()
        if low in _NEVER_END or _ACRONYM.match(tok):
            return False
        if len(tok) == 1 and tok.isupper():  # initials: "J. Smith"
            return False
        if nxt.isdigit() and low in _BEFORE_DIGIT:
            return False
    return True


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (abbreviation/number safe). End of text is a boundary."""
    out: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        if _is_boundary(text, m):
            s = text[start : m.end(2)].strip()
            if s:
                out.append(s)
            start = m.end(3)
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _last_boundary(text: str) -> int:
    """Index just after the last confirmed sentence boundary in ``text`` (0 if none)."""
    cut = 0
    for m in _BOUNDARY.finditer(text):
        if _is_boundary(text, m):
            cut = m.end(3)
    return cut


def _terminate(s: str) -> str:
    s = s.rstrip()
    if s and s[-1] not in ".!?:;…\"'”’)":
        s += "."
    return s


def _line_to_text(line: str) -> str | None:
    """Convert one complete markdown line (outside code). None = skip the line."""
    if not line.strip() or _HRULE.match(line) or _TABLE.match(line) or _TABLE_SEP.match(line):
        return None
    if m := _HEADER.match(line):
        return _terminate(clean_inline(m.group(1)))
    if m := _BULLET.match(line):
        return _terminate(clean_inline(m.group(1)))
    if m := _QUOTE.match(line):
        line = m.group(1)
    return _terminate(clean_inline(line))


def speakable(text: str, code_placeholder: str = CODE_PLACEHOLDER) -> str:
    """Whole-text conversion of markdown to a single speakable string."""
    seg = SentenceSegmenter(min_chars=0, code_placeholder=code_placeholder)
    return " ".join(seg.feed(text) + seg.flush())


class SentenceSegmenter:
    """Streaming segmenter. Not thread-safe; use one per turn (or call ``reset``)."""

    def __init__(self, min_chars: int = MIN_CHARS, code_placeholder: str = CODE_PLACEHOLDER) -> None:
        self.min_chars = min_chars
        self.code_placeholder = code_placeholder
        self.reset()

    def reset(self) -> None:
        self._raw = ""
        self._held = ""
        self._in_code = False
        self._said_code = False
        self._mid_line = False  # part of the current line has already been consumed

    def feed(self, delta: str) -> list[str]:
        out: list[str] = []
        self._raw += delta.replace("\r\n", "\n").replace("\r", "\n")
        while "\n" in self._raw:
            line, self._raw = self._raw.split("\n", 1)
            self._consume_line(line, out)
        if self._raw and not self._in_code and (self._mid_line or not _MAYBE_STRUCTURAL.match(self._raw)):
            cut = _last_boundary(self._raw)
            if cut:
                chunk, self._raw = self._raw[:cut], self._raw[cut:]
                self._mid_line = True
                self._emit(split_sentences(clean_inline(chunk)), out)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        if self._raw:
            line, self._raw = self._raw, ""
            self._consume_line(line, out)
        if self._held:
            out.append(self._held)
        self._held = ""
        self._in_code = False
        self._mid_line = False
        return out

    # ------------------------------------------------------------------ internals

    def _consume_line(self, line: str, out: list[str]) -> None:
        mid, self._mid_line = self._mid_line, False
        if not mid and _FENCE.match(line):
            if not self._in_code and not self._said_code:
                self._said_code = True
                self._emit([self.code_placeholder], out)
            self._in_code = not self._in_code
            return
        if self._in_code:
            return
        text = _terminate(clean_inline(line)) if mid else _line_to_text(line)
        if text:
            self._emit(split_sentences(text), out)

    def _emit(self, sentences: list[str], out: list[str]) -> None:
        for s in sentences:
            s = s.strip()
            if not s or not any(ch.isalnum() for ch in s):
                continue
            if self._held:
                s = f"{self._held} {s}"
                self._held = ""
            if len(s) < self.min_chars:
                self._held = s
            else:
                out.append(s)
