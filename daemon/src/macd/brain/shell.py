"""A small, conservative POSIX-shell lexer for the safety gate.

It never executes anything. It turns a command string into a list of pipelines of simple
commands, extracting command/process substitutions, heredocs and here-strings so that the
gate can classify every piece. When in doubt it errs towards *seeing more* commands (a
false positive costs a confirmation, a false negative can cost a disk).
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field

SUBST = "\x00SUBST\x00"  # placeholder for a $(...) / `...` whose value is unknown

_OPERATORS = (
    "&>>", "<<<", "<<-", "&&", "||", ";;", "|&", ">>", "<<", "<&", ">&", "&>", ">|", "<>",
    ";", "&", "|", "(", ")", "<", ">",
)
_OP_CHARS = set(";&|()<>")
_SEPARATORS = {";", "&", "&&", "||", ";;", "(", ")"}
_PIPES = {"|", "|&"}


@dataclass
class Word:
    text: str
    glob: bool = False  # contains an unquoted * ? or [


@dataclass
class Token:
    kind: str  # "word" | "op"
    word: Word | None = None
    op: str = ""


@dataclass
class SimpleCommand:
    words: list[Word] = field(default_factory=list)
    redirects_out: list[Word] = field(default_factory=list)
    redirects_in: list[Word] = field(default_factory=list)
    stdin_text: str | None = None  # heredoc / here-string body


@dataclass
class Script:
    pipelines: list[list[SimpleCommand]]
    substitutions: list[str]  # inner scripts of $(...), `...`, <(...), >(...)


class _Lexer:
    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0
        self.tokens: list[Token] = []
        self.subs: list[str] = []
        self.heredocs: list[str] = []
        self._pending_heredocs: list[tuple[int, str, bool]] = []  # (index, delim, strip_tabs)
        self._buf: list[str] = []
        self._glob = False
        self._in_word = False

    # -- word buffer -------------------------------------------------------------------
    def _flush(self) -> None:
        if self._in_word:
            self.tokens.append(Token("word", Word("".join(self._buf), self._glob)))
        self._buf, self._glob, self._in_word = [], False, False

    def _add(self, text: str) -> None:
        self._buf.append(text)
        self._in_word = True

    # -- scanning helpers --------------------------------------------------------------
    def _match_paren(self, start: int) -> int:
        """Index just after the ')' matching the '(' at s[start-1]. Quote-aware."""
        depth, i, s = 1, start, self.s
        while i < len(s):
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == "'":
                j = s.find("'", i + 1)
                i = len(s) if j < 0 else j + 1
                continue
            if c == '"':
                i = self._skip_double(i + 1)
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(s)

    def _skip_double(self, i: int) -> int:
        s = self.s
        while i < len(s):
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == '"':
                return i + 1
            i += 1
        return len(s)

    def _backtick(self, start: int) -> tuple[str, int]:
        s, i, out = self.s, start, []
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            if s[i] == "`":
                return "".join(out), i + 1
            out.append(s[i])
            i += 1
        return "".join(out), len(s)

    def _try_subst(self) -> bool:
        """Handle $( ... ), $(( ... )) and `...` at self.i. Returns True if consumed."""
        s, i = self.s, self.i
        if s.startswith("$(", i):
            end = self._match_paren(i + 2)
            inner = s[i + 2 : end - 1]
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]  # arithmetic: still scan it
        elif s[i] == "`":
            inner, end = self._backtick(i + 1)
        else:
            return False
        self.subs.append(inner)
        self._add(_echo_value(inner) or SUBST)
        self.i = end
        return True

    def _read_double(self) -> None:
        s = self.s
        self.i += 1
        while self.i < len(s):
            c = s[self.i]
            if c == '"':
                self.i += 1
                return
            if c == "\\" and self.i + 1 < len(s):
                nxt = s[self.i + 1]
                if nxt in '"\\$`':
                    self._add(nxt)
                elif nxt == "\n":
                    pass
                else:
                    self._add("\\" + nxt)
                self.i += 2
                continue
            if c in "$`" and self._try_subst():
                continue
            self._add(c)
            self.i += 1

    def _read_ansi_c(self) -> None:
        s = self.s
        j = self.i + 2
        out = []
        while j < len(s) and s[j] != "'":
            if s[j] == "\\" and j + 1 < len(s):
                out.append(s[j : j + 2])
                j += 2
                continue
            out.append(s[j])
            j += 1
        raw = "".join(out)
        try:
            text = codecs.decode(raw.encode("latin-1", "backslashreplace"), "unicode_escape")
        except Exception:  # noqa: BLE001 - malformed escapes: keep raw text
            text = raw
        self._add(text)
        self.i = j + 1

    def _read_heredoc_bodies(self) -> None:
        """Called at an unquoted newline: consume the bodies of pending heredocs."""
        s = self.s
        pos = self.i + 1
        for idx, delim, strip_tabs in self._pending_heredocs:
            lines = []
            while pos <= len(s):
                nl = s.find("\n", pos)
                line = s[pos:] if nl < 0 else s[pos:nl]
                pos = len(s) + 1 if nl < 0 else nl + 1
                check = line.lstrip("\t") if strip_tabs else line
                if check == delim:
                    break
                lines.append(check)
            self.heredocs[idx] = "\n".join(lines)
        self._pending_heredocs.clear()
        self.i = min(pos, len(s)) - 1  # main loop will step over the newline

    def _read_operator(self) -> str:
        for op in _OPERATORS:
            if self.s.startswith(op, self.i):
                self.i += len(op)
                return op
        self.i += 1
        return self.s[self.i - 1]

    # -- main loop ---------------------------------------------------------------------
    def run(self) -> None:
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c == "\\":
                if self.i + 1 < len(s) and s[self.i + 1] == "\n":
                    self.i += 2
                    continue
                if self.i + 1 < len(s):
                    self._add(s[self.i + 1])
                self.i += 2
                continue
            if c == "'":
                j = s.find("'", self.i + 1)
                j = len(s) if j < 0 else j
                self._add(s[self.i + 1 : j])
                self.i = j + 1
                continue
            if c == "$" and s.startswith("$'", self.i):
                self._read_ansi_c()
                continue
            if c == '"':
                self._in_word = True
                self._read_double()
                continue
            if c in "$`" and self._try_subst():
                continue
            if c in "<>" and s.startswith("(", self.i + 1):
                end = self._match_paren(self.i + 2)
                self.subs.append(s[self.i + 2 : end - 1])
                self._add(SUBST)
                self.i = end
                continue
            if c == "#" and not self._in_word:
                nl = s.find("\n", self.i)
                self.i = len(s) if nl < 0 else nl
                continue
            if c == "\n":
                self._flush()
                self.tokens.append(Token("op", op=";"))
                if self._pending_heredocs:
                    self._read_heredoc_bodies()
                self.i += 1
                continue
            if c in " \t\r":
                self._flush()
                self.i += 1
                continue
            if c in _OP_CHARS:
                # `2>` / `2>&1`: a pure-digit word right before a redirect is an fd number.
                if c in "<>" and self._in_word and "".join(self._buf).isdigit():
                    self._buf, self._in_word = [], False
                self._flush()
                op = self._read_operator()
                self.tokens.append(Token("op", op=op))
                if op in ("<<", "<<-"):
                    self._lex_heredoc_delim(strip_tabs=op == "<<-")
                continue
            if c in "*?[":
                self._glob = True
            self._add(c)
            self.i += 1
        self._flush()
        if self._pending_heredocs:  # unterminated heredoc: body is the rest (already lexed)
            for idx, _, _ in self._pending_heredocs:
                self.heredocs[idx] = ""

    def _lex_heredoc_delim(self, strip_tabs: bool) -> None:
        s = self.s
        while self.i < len(s) and s[self.i] in " \t":
            self.i += 1
        m = re.match(r"""(['"]?)([^\s;&|<>()'"]+)\1""", s[self.i :])
        delim = m.group(2) if m else ""
        if m:
            self.i += m.end()
        idx = len(self.heredocs)
        self.heredocs.append("")
        self._pending_heredocs.append((idx, delim, strip_tabs))
        self.tokens.append(Token("word", Word(f"\x00HEREDOC{idx}\x00")))


_HEREDOC_RE = re.compile(r"\x00HEREDOC(\d+)\x00")
_ECHO_RE = re.compile(r"\s*(?:echo|printf)\s+(?:-[A-Za-z]+\s+|'%s'\s+|\"%s\"\s+|%s\s+)*(.+?)\s*", re.DOTALL)


def _echo_value(inner: str) -> str | None:
    """`$(echo ~/.ssh/id_rsa)` → the literal text, so the outer word is still a path."""
    m = _ECHO_RE.fullmatch(inner)
    if not m or any(ch in m.group(1) for ch in "$`|;&<>("):
        return None
    return m.group(1).replace("'", "").replace('"', "")


def parse(src: str) -> Script:
    """Lex `src` into pipelines of simple commands (never raises)."""
    lx = _Lexer(src)
    try:
        lx.run()
    except Exception:  # noqa: BLE001 - lexer bug: fall back to whitespace split
        lx.tokens = [Token("word", Word(w, any(ch in w for ch in "*?["))) for w in src.split()]
    pipelines: list[list[SimpleCommand]] = []
    pipeline: list[SimpleCommand] = []
    cmd = SimpleCommand()
    toks = lx.tokens
    k = 0

    def end_cmd() -> None:
        nonlocal cmd
        if cmd.words or cmd.redirects_in or cmd.redirects_out:
            pipeline.append(cmd)
        cmd = SimpleCommand()

    def end_pipeline() -> None:
        nonlocal pipeline
        end_cmd()
        if pipeline:
            pipelines.append(pipeline)
        pipeline = []

    while k < len(toks):
        t = toks[k]
        if t.kind == "word":
            assert t.word is not None
            cmd.words.append(t.word)
        elif t.op in _PIPES:
            end_cmd()
        elif t.op in _SEPARATORS:
            end_pipeline()
        else:  # redirect: next word is the target
            target = toks[k + 1].word if k + 1 < len(toks) and toks[k + 1].kind == "word" else None
            if target is not None:
                k += 1
                hd = _HEREDOC_RE.fullmatch(target.text)
                if hd:
                    cmd.stdin_text = lx.heredocs[int(hd.group(1))]
                elif t.op == "<<<":
                    cmd.stdin_text = target.text
                elif ">" in t.op:
                    if not (t.op.endswith("&") and target.text.isdigit()) and target.text != "-":
                        cmd.redirects_out.append(target)
                else:
                    if not (t.op.endswith("&") and target.text.isdigit()):
                        cmd.redirects_in.append(target)
        k += 1
    end_pipeline()
    return Script(pipelines=pipelines, substitutions=lx.subs)


def brace_expand(text: str, limit: int = 64) -> list[str]:
    """Expand simple `{a,b}` and `{1..3}` braces (bounded)."""
    m = re.search(r"\{([^{}]*)\}", text)
    if not m:
        return [text]
    body = m.group(1)
    if "," in body:
        alts = body.split(",")
    else:
        rng = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", body)
        if not rng:
            return [text]
        a, b = int(rng.group(1)), int(rng.group(2))
        step = 1 if b >= a else -1
        alts = [str(x) for x in range(a, b + step, step)][:limit]
    out: list[str] = []
    for alt in alts:
        for e in brace_expand(text[: m.start()] + alt + text[m.end() :], limit):
            out.append(e)
            if len(out) >= limit:
                return out
    return out
