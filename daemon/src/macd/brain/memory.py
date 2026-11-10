"""Structured long-term memory (memory.md): sections, grounding, dedupe, forget, consolidation.

Layout (always all sections, in this order):

    # Memory
    ## About you
    - ...
    ## Preferences
    ...
    ## Recent context
    - (session handoff lines, replaced on every handoff)

A line ending in `` [kept]`` was an explicit "remember …" from the user; consolidation must
preserve it verbatim. Writes are capability-gated by the PreToolUse hook: a fact is written only
when the user explicitly asked, or when it is *grounded* in what the user recently said.

Writers (F11, WhatsApp ARCHITECTURE D8 and W3.4): one `MemoryFile` instance is meant to be shared
by every brain of a process (agentd injects its single instance). Each read-modify-write runs under
an in-process lock and, while the voice brain can still run in macd (`brain.mode = local`), under a
transitional `flock` on the sidecar ``memory.md.lock`` so a second process never loses a fact.
The flock goes away together with local mode (pass ``flock=False``).
"""

from __future__ import annotations

import contextlib
import difflib
import fcntl
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS: tuple[str, ...] = (
    "About you", "Preferences", "Projects & work", "People", "Recurring tasks", "Recent context",
)
KINDS: dict[str, str] = {
    "about": "About you", "preference": "Preferences", "preferences": "Preferences",
    "project": "Projects & work", "work": "Projects & work", "person": "People", "people": "People",
    "task": "Recurring tasks", "recurring": "Recurring tasks", "context": "Recent context",
}
KEPT = " [kept]"
DEDUPE_RATIO = 0.85
FORGET_RATIO = 0.6
GROUNDING_RATIO = 0.6
RECENT_CONTEXT_LIMIT = 1024
CONSOLIDATE_ABOVE = 6 * 1024
CONSOLIDATE_TARGET = 4 * 1024
FLOCK_TIMEOUT_S = 5.0  # a write never waits longer than this for another process

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(["about", "above", "after", "again", "against", "also", "always", "answer", "answers", "anything", "around", "because", "been", "before", "being", "between", "both", "could", "does", "doing", "done", "during", "each", "either", "every", "from", "have", "having", "here", "into", "just", "like", "mine", "more", "most", "much", "must", "never", "only", "other", "over", "please", "prefer", "prefers", "quite", "rather", "really", "same", "shall", "should", "since", "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "thing", "things", "this", "those", "through", "under", "until", "very", "want", "wants", "were", "what", "when", "where", "which", "while", "will", "with", "would", "your", "yours"])
# Three-letter words count too ("tea", "son", "gym"); the common short function words do not.
_STOPWORDS = _STOPWORDS | frozenset(["the", "and", "for", "you", "are", "was", "his", "her", "she", "him", "has",
                                     "had", "not", "but", "our", "its", "who", "how", "why", "can", "did", "get",
                                     "got", "use", "new", "now", "all", "any", "may", "one", "too", "out", "own",
                                     "yes", "let", "say", "see", "way", "off", "per", "via", "user"])
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _norm(text: str) -> str:
    """Lowercase words of any script (\\w is Unicode-aware), punctuation and [kept] removed."""
    return " ".join(re.sub(r"[^\w ]+|_", " ", text.replace(KEPT.strip(), "").casefold()).split())


def _stem(word: str) -> str:
    w = re.sub(r"'s?$", "", word.lower().strip("'-"))
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def content_words(text: str) -> set[str]:
    stems = {_stem(w) for w in _WORD.findall(text) if w.lower() not in _STOPWORDS}
    return stems - _STOPWORDS  # "likes" stems to the stopword "like" (the user's "like" is dropped too)


def _ngrams(text: str, n: int = 3) -> set[str]:
    s = _norm(text).replace(" ", "")
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def grounded(fact: str, recent_user_text: str, ratio: float = GROUNDING_RATIO) -> bool:
    """True if ≥ `ratio` of the fact's content words occur in what the user recently said.

    Content words are English (stopwords, light stemming). For a fact in another script, or one
    with no English content words, the deterministic fallback (AG12) is character 3-gram
    overlap: ≥ `ratio` of the fact's 3-grams (letters only, spaces and punctuation dropped,
    casefolded) must occur in what the user said."""
    words = content_words(fact)
    if words and fact.isascii():
        said = content_words(recent_user_text)
        return len(words & said) / len(words) >= ratio
    grams = _ngrams(fact)
    if len(grams) < 2:
        return False
    return len(grams & _ngrams(recent_user_text)) / len(grams) >= ratio


def section_for(kind: str | None) -> str:
    return KINDS.get((kind or "").strip().lower(), "About you")


@dataclass
class Memory:
    sections: dict[str, list[str]] = field(default_factory=lambda: {s: [] for s in SECTIONS})

    # -- text <-> structure ------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> Memory:
        mem = cls()
        current: str | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                title = line[3:].strip()
                current = next((s for s in SECTIONS if s.lower() == title.lower()), None)
                if current is None:
                    current = "About you"
                continue
            if not line or line.startswith("#"):
                continue
            item = line[1:].strip() if line[0] in "-*" else line
            item = re.sub(r"^\d{4}-\d{2}-\d{2}:\s*", "", item)  # legacy dated bullets
            if item and current is not None:
                mem.sections[current].append(item)
            elif item:
                mem.sections["About you"].append(item)
        return mem

    def render(self) -> str:
        out = ["# Memory", ""]
        for name in SECTIONS:
            out.append(f"## {name}")
            out.extend(f"- {line}" for line in self.sections[name])
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def lines(self) -> list[tuple[str, str]]:
        return [(s, line) for s in SECTIONS for line in self.sections[s]]

    def kept_lines(self) -> list[tuple[str, str]]:
        return [(s, line) for s, line in self.lines() if line.endswith(KEPT)]

    # -- edits -------------------------------------------------------------------------
    def remember(self, fact: str, kind: str | None = None, kept: bool = False) -> str:
        """Add `fact` to its section, replacing a near-duplicate anywhere. Returns 'added'|'updated'."""
        fact = " ".join(fact.split()).rstrip(".")
        if kept and not fact.endswith(KEPT):
            fact += KEPT
        target = section_for(kind)
        best: tuple[float, str, int] | None = None
        for s in SECTIONS:
            for i, line in enumerate(self.sections[s]):
                r = difflib.SequenceMatcher(None, _norm(line), _norm(fact)).ratio()
                if r >= DEDUPE_RATIO and (best is None or r > best[0]):
                    best = (r, s, i)
        if best is not None:
            _, s, i = best
            old = self.sections[s][i]
            if old.endswith(KEPT) and not fact.endswith(KEPT):
                fact += KEPT
            self.sections[s][i] = fact
            return "updated"
        self.sections[target].append(fact)
        return "added"

    def forget(self, text: str) -> int:
        """Remove lines matching `text` (substring, fuzzy, or most of its content words). Returns count."""
        needle = _norm(text)
        words = content_words(text)
        removed = 0
        for s in SECTIONS:
            keep = []
            for line in self.sections[s]:
                hay = _norm(line)
                overlap = len(words & content_words(line)) / len(words) if words else 0.0
                hit = bool(needle) and (needle in hay or overlap >= FORGET_RATIO
                                        or difflib.SequenceMatcher(None, hay, needle).ratio() >= FORGET_RATIO)
                if hit:
                    removed += 1
                else:
                    keep.append(line)
            self.sections[s] = keep
        return removed

    def set_recent_context(self, lines: list[str], limit: int = RECENT_CONTEXT_LIMIT) -> None:
        out: list[str] = []
        used = 0
        for raw in lines:
            line = " ".join(raw.split())
            if not line:
                continue
            if used + len(line) + 3 > limit:
                break
            out.append(line)
            used += len(line) + 3
        self.sections["Recent context"] = out

    def consolidate(self, rewritten: str) -> Memory:
        """Accept a model-rewritten memory, re-adding any [kept] line it dropped."""
        new = Memory.parse(rewritten)
        present = {_norm(line) for _, line in new.lines()}
        for section, line in self.kept_lines():
            if _norm(line) not in present:
                new.sections[section].append(line)
        return new


class MemoryFile:
    """memory.md on disk; every operation re-reads and rewrites atomically.

    Thread-safe (calls run in `asyncio.to_thread`). Mutations hold `_lock` and, with ``flock``,
    an exclusive flock on ``<name>.lock`` next to the file (transitional, local mode only).
    """

    def __init__(self, path: Path, *, flock: bool = True, lock_path: Path | None = None,
                 flock_timeout_s: float = FLOCK_TIMEOUT_S) -> None:
        self.path = Path(path)
        self.flock = flock
        self.lock_path = lock_path or self.path.with_name(self.path.name + ".lock")
        self.flock_timeout_s = flock_timeout_s
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def _write_lock(self) -> Iterator[None]:
        with self._lock:
            if not self.flock:
                yield
                return
            fd = self._open_flock()
            try:
                yield
            finally:
                if fd is not None:
                    os.close(fd)  # closing the descriptor drops the flock

    def _open_flock(self) -> int | None:
        """Exclusive flock on the sidecar, polled (bounded). None: proceed without it, a broken
        lock file must not stop memory writes (the in-process lock still serialises)."""
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            log.warning("memory: cannot open %s; writing without the cross-process lock", self.lock_path)
            return None
        deadline = time.monotonic() + self.flock_timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError("memory file is busy in another process") from None
                time.sleep(0.02)
            except OSError:
                log.warning("memory: flock on %s failed; writing without it", self.lock_path)
                return fd

    def _mutate(self, fn: Callable[[Memory], bool]) -> None:
        """Load, apply `fn` (returns whether to save), save: one locked read-modify-write."""
        with self._write_lock():
            mem = self.load()
            if fn(mem):
                self.save(mem)

    def load(self) -> Memory:
        try:
            return Memory.parse(self.path.read_text(encoding="utf-8"))
        except OSError:
            return Memory()

    def save(self, mem: Memory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".memory.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(mem.render())
        os.replace(tmp, self.path)

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def remember(self, fact: str, kind: str | None = None, kept: bool = False) -> str:
        result = ""

        def apply(mem: Memory) -> bool:
            nonlocal result
            result = mem.remember(fact, kind, kept)
            return True
        self._mutate(apply)
        return result

    def forget(self, text: str) -> int:
        removed = 0

        def apply(mem: Memory) -> bool:
            nonlocal removed
            removed = mem.forget(text)
            return removed > 0
        self._mutate(apply)
        return removed

    def set_recent_context(self, lines: list[str]) -> None:
        def apply(mem: Memory) -> bool:
            mem.set_recent_context(lines)
            return True
        self._mutate(apply)

    def recent_context(self) -> list[str]:
        return list(self.load().sections["Recent context"])

    def add_recent_context(self, line: str) -> None:
        def apply(mem: Memory) -> bool:
            mem.set_recent_context([*mem.sections["Recent context"], line])
            return True
        self._mutate(apply)

    def consolidate(self, rewritten: str) -> None:
        with self._write_lock():
            self.save(self.load().consolidate(rewritten))

    def needs_consolidation(self) -> bool:
        return self.size() > CONSOLIDATE_ABOVE
