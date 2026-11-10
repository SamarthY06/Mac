"""Routing a chat request to a folder and a session (W3.5, D12). Voice never routes.

0. follow-up: the last answer was under 30 min ago and no `new_session` -> same folder and session
1. an exact folder name ("in Resume", `new_session{folder}`) -> that folder; an unknown name goes
   to Spotlight (escaped, N-a); several hits -> ask by path tail
2. one call to the warm Haiku client; the reply's LAST JSON object is parsed; at or above the
   threshold it is taken
3. otherwise one short `route` question
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentd.workspaces import (
    WorkspaceStore,
    is_skipped,
    resolve_user_path,
    spotlight_find,
    under,
)

log = logging.getLogger(__name__)

FOLLOWUP_S = 30 * 60
# P0.5: 10/10 picks right with confidences 0.7-0.95; the one right pick at 0.7 must not ask.
CONFIDENCE = 0.7
RECENT_DAYS = 30
MAX_RECENT = 15
SESSIONS_PER_FOLDER = 3
ROUTER_SYSTEM = (
    "You route a user's request to a folder on their Mac. Answer with ONE line of JSON and nothing "
    'else: {"folder": "<name from the list or general>", "session": "continue:<id>" or "new", '
    '"confidence": <0..1>}. Continue a session only for a follow-up or repeat of that work; '
    "a totally new task in the same folder is new. Use general when no folder fits.")
_NAMED = re.compile(r"\b(?:in|inside)\s+(?:the\s+|my\s+)?(?P<n>[~\w./-]+?)\s+(?:folder|project|workspace)\b"
                    r"|\b(?:folder|project|workspace)\s+(?P<m>[~\w./-]+)", re.IGNORECASE)


@dataclass
class Candidate:
    name: str
    path: Path
    about: str = ""
    sessions: list[tuple[str, str, str]] = field(default_factory=list)  # (id, title, date)


@dataclass
class Option:
    label: str
    folder: Path | None
    session: str  # "same" | "new" | "continue:<id>"
    keys: list[str] = field(default_factory=list)  # words in a reply that pick this option


@dataclass
class Route:
    folder: Path | None  # None = general
    session: str = "new"  # "same" | "new" | "continue:<id>"
    question: str = ""  # non-empty: ask this `route` question with `options`
    options: list[Option] = field(default_factory=list)


class RouterModel(Protocol):
    async def complete(self, prompt: str) -> str: ...


def parse_last_json(text: str) -> dict[str, Any] | None:
    """The last JSON object in `text` (the model sometimes writes a sentence first)."""
    dec = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found = obj
        i = text.find("{", end)
    return found


class Router:
    def __init__(self, workspaces: WorkspaceStore, model: RouterModel | None, *,
                 list_sessions: Callable[..., list[Any]] | None = None,
                 mdfind: Callable[[list[str]], str] | None = None,
                 clock: Callable[[], float] = time.time, threshold: float = CONFIDENCE) -> None:
        self.ws = workspaces
        self.home = workspaces.home
        self.model = model
        self._list_sessions = list_sessions or _sdk_list_sessions
        self._mdfind = mdfind
        self._clock = clock
        self.threshold = threshold

    async def route(self, text: str, *, folder: Path | None, last_answer_at: float,
                    fresh: bool, named: str | None) -> Route:
        if not fresh and last_answer_at and self._clock() - last_answer_at < FOLLOWUP_S:
            return Route(folder, "same")  # 0. follow-up
        cands = await asyncio.to_thread(self.candidates)
        name = named or _named(text, cands)
        if name:  # 1. exact name
            hits = await self._by_name(name, cands)
            if len(hits) == 1:
                return Route(hits[0], "same" if hits[0] == folder and not fresh else "new")
            if len(hits) > 1:
                opts = [Option(_tail(p, self.home), p, "new", [_tail(p, self.home).lower(), str(i + 1)])
                        for i, p in enumerate(hits[:4])]
                return Route(None, question=f"Which {name} do you mean: "
                             f"{', '.join(o.label for o in opts)}?", options=opts)
        pick = await self._ask_model(text, cands)  # 2. warm Haiku
        if pick is not None:
            cand, session, conf = pick
            if fresh:
                session = "new"
            if cand is None or conf >= self.threshold:
                return Route(cand.path if cand else None, session)
            return self._question(cand, session)  # 3. ask
        cur = next((c for c in cands if c.path == folder), None)
        return self._question(cur, "new") if cur else Route(None, "new")

    # -- steps ---------------------------------------------------------------------------
    async def _by_name(self, name: str, cands: list[Candidate]) -> list[Path]:
        p = resolve_user_path(name, self.home) if name.startswith(("~", "/")) else None
        if p is not None:
            ok = await asyncio.to_thread(p.is_dir)
            return [p] if ok and not is_skipped(p, self.home) else []
        exact = [c.path for c in cands if c.name.lower() == name.lower()]
        aliases = [p for p, al in self.ws.entries() if name.lower() in (a.lower() for a in al)]
        if exact or aliases:
            return sorted(set(exact + aliases))
        if self._mdfind is not None:
            return await asyncio.to_thread(spotlight_find, name, self.home, self._mdfind)
        return await asyncio.to_thread(spotlight_find, name, self.home)

    async def _ask_model(self, text: str, cands: list[Candidate]) -> tuple[Candidate | None, str, float] | None:
        if self.model is None:
            return None
        try:
            reply = await self.model.complete(_prompt(text, cands, self._clock()))
        except Exception:
            log.warning("router: model call failed", exc_info=True)
            return None
        obj = parse_last_json(reply)
        if not obj:
            return None
        by_name = {c.name.lower(): c for c in cands}
        fname = str(obj.get("folder", "")).strip().lower()
        if fname == "general":
            cand = None
        elif fname in by_name:
            cand = by_name[fname]
        else:
            return None
        session = str(obj.get("session", "new"))
        if session.startswith("continue:"):
            sid = session.split(":", 1)[1]
            if cand is None or sid not in {x[0] for x in cand.sessions}:
                return None
        elif session != "new":
            return None
        try:
            conf = float(obj.get("confidence", 0))
        except (TypeError, ValueError):
            return None
        return cand, session, conf

    def _question(self, cand: Candidate, session: str) -> Route:
        if session.startswith("continue:"):
            sid = session.split(":", 1)[1]
            title, date = next(((t, d) for i, t, d in cand.sessions if i == sid), ("that", ""))
            when = f" from {date}" if date else ""
            return Route(None, question=f'Continue "{title}"{when}, or start fresh?', options=[
                Option("continue", cand.path, session, ["continue", "yes", "same", "pick up"]),
                Option("fresh", cand.path, "new", ["fresh", "new", "no", "start over"])])
        return Route(None, question=f"Should I do this in {cand.name}, or just generally?", options=[
            Option(cand.name, cand.path, "new", [cand.name.lower(), "yes", "there"]),
            Option("generally", None, "new", ["general", "generally", "no", "anywhere"])])

    # -- candidates (blocking; called in a thread) --------------------------------------
    def candidates(self) -> list[Candidate]:
        out: dict[Path, Candidate] = {}
        for path, aliases in self.ws.entries():
            if path.is_dir() and not is_skipped(path, self.home):
                about = _about(path)
                if aliases:
                    about = f"also called {', '.join(aliases)}. {about}"
                out[path] = Candidate(path.name, path, about)
        cutoff = (self._clock() - RECENT_DAYS * 86400) * 1000
        try:
            infos = self._list_sessions(limit=200)
        except Exception:
            log.warning("router: list_sessions failed", exc_info=True)
            infos = []
        recent = 0
        for info in sorted(infos, key=lambda i: -(getattr(i, "last_modified", 0) or 0)):
            cwd = getattr(info, "cwd", None)
            if recent >= MAX_RECENT or not cwd or (getattr(info, "last_modified", 0) or 0) < cutoff:
                continue
            p = Path(cwd)
            if p not in out and under(p, self.home) and p != self.home and not is_skipped(p, self.home) \
                    and p.is_dir():
                out[p] = Candidate(p.name, p)
                recent += 1
        for c in out.values():
            try:
                infos = self._list_sessions(directory=str(c.path), limit=SESSIONS_PER_FOLDER,
                                            include_worktrees=False)
            except Exception:  # noqa: BLE001 - metadata only; a folder without sessions is fine
                infos = []
            c.sessions = [(i.session_id, (i.custom_title or i.summary or "")[:60],
                           time.strftime("%b %-d", time.localtime((i.last_modified or 0) / 1000)))
                          for i in infos[:SESSIONS_PER_FOLDER]]
        return list(out.values())


def _sdk_list_sessions(**kw: Any) -> list[Any]:
    from claude_agent_sdk import list_sessions

    return list_sessions(**kw)


def _named(text: str, cands: list[Candidate]) -> str | None:
    m = _NAMED.search(text)
    if m:
        return (m.group("n") or m.group("m")).rstrip(".,")
    low = text.lower()
    for c in sorted(cands, key=lambda c: -len(c.name)):
        if re.search(rf"\bin\s+(?:the\s+|my\s+)?{re.escape(c.name.lower())}(?![\w-])", low):
            return c.name
    return None


def _tail(p: Path, home: Path) -> str:
    return "/".join(p.relative_to(home).parts[-2:]) if under(p, home) else p.name


def _about(path: Path) -> str:
    bits = []
    with contextlib.suppress(OSError):
        first = (path / "CLAUDE.md").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if first:
            bits.append(first[0].lstrip("# ")[:120])
    skills = path / ".claude" / "skills"
    if skills.is_dir():
        for sk in sorted(skills.iterdir())[:10]:
            desc = ""
            with contextlib.suppress(OSError):
                m = re.search(r"^description:\s*(.+)$", (sk / "SKILL.md").read_text(errors="replace"), re.MULTILINE)
                desc = m.group(1)[:80] if m else ""
            bits.append(f"skill {sk.name}: {desc}".rstrip(": "))
    return ". ".join(bits)


def _prompt(text: str, cands: list[Candidate], now: float) -> str:
    lines = []
    for c in cands:
        sess = ", ".join(f"{d} '{t}' (id {i})" for i, t, d in c.sessions) or "none"
        lines.append(f"- {c.name}: {c.about or 'a folder'}. Sessions: {sess}")
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    return "Folders:\n" + "\n".join(lines) + f"\n\nToday: {today}.\nRequest: {text}"


class WarmHaiku:
    """The warm router client (P0.5): one long-lived ClaudeSDKClient(model="haiku", tools=[],
    setting_sources=[]); a fresh session every 20 calls or after 10 min idle."""

    def __init__(self, cli_path: str | None = None, *, refresh_calls: int = 20, idle_s: float = 600.0,
                 timeout_s: float = 15.0) -> None:
        self.cli_path = cli_path
        self.refresh_calls = refresh_calls
        self.idle_s = idle_s
        self.timeout_s = timeout_s
        self._client: Any = None
        self._calls = 0
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def complete(self, prompt: str) -> str:
        async with self._lock:
            return await asyncio.wait_for(self._complete(prompt), self.timeout_s)

    async def _complete(self, prompt: str) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import AssistantMessage, ResultMessage

        now = time.monotonic()
        if self._client is not None and (self._calls >= self.refresh_calls or now - self._last > self.idle_s):
            await self.close()
        if self._client is None:
            opts = ClaudeAgentOptions(model="haiku", tools=[], setting_sources=[], max_turns=1,
                                      system_prompt=ROUTER_SYSTEM, cli_path=self.cli_path)
            client = ClaudeSDKClient(opts)
            await client.connect()
            self._client, self._calls = client, 0
        self._calls += 1
        self._last = now
        try:
            await self._client.query(prompt)
            out = ""
            async for m in self._client.receive_response():
                if isinstance(m, AssistantMessage):
                    out += "".join(getattr(b, "text", "") for b in m.content)
                if isinstance(m, ResultMessage):
                    break
            return out
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.disconnect()
