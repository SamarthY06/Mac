"""ClaudeBrain: contracts.Brain on top of claude-agent-sdk (ARCHITECTURE §5).

Owner-task model: ONE long-lived task owns the ClaudeSDKClient (connect and disconnect happen
in that task). Everything else talks to it through an inbox queue of `_Cmd`s; a small reader
task pumps SDK messages into the same inbox, so commands and messages are handled strictly in
order without locks.

The CLI process is lazy: start() only starts the owner task; the client is connected on the
first ask() (or warm(), called on the wake word) and shut down again by sleep() or after
cfg.brain.idle_shutdown_min minutes without work, keeping the session id (state.json) so the
next ask() resumes the conversation unless it is older than session_idle_rotate_min.

Event contract (shared with FakeBrain):
- every BrainEvent carries the turn_id of the ask that produced it;
- each turn ends with exactly one terminal event: DONE (``data["interrupted"]`` tells whether
  it was cut short) or ERROR;
- after interrupt(turn_id) nothing is emitted for that turn except the final
  DONE(interrupted=True); the stale tail is drained until its ResultMessage, and the next
  Ask is only sent after the drain (bounded; otherwise the client is reconnected).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from macd import config
from macd.brain import errors, mac_tools, memory, prompts
from macd.brain.gate import ConfirmationGate, GateContext
from macd.brain.hooks import CONFIRM_TOOL, MEMORY_TOOLS, HookDecision, PreToolHook
from macd.brain.mac_tools import MacTools
from macd.brain.narration import describe_tool
from macd.contracts import BrainErrorKind, BrainEvent, BrainEventKind

log = logging.getLogger(__name__)

SAY_TOOL = "mcp__mac__say"
_SILENT_TOOLS = {SAY_TOOL, CONFIRM_TOOL}
HANDOFF_TURN = -1  # turn id of the silent session-handoff query (never emitted)
PROBE_TURN = -2  # turn id of the silent connect_check probe query (never emitted)
PROBE_PROMPT = "Reply with exactly the word OK and nothing else. Do not use any tools."
CONNECT_CHECK_TIMEOUT_S = 25.0
CHROME_HINT = ("Chrome isn't running, the Claude in Chrome extension isn't installed, or Chrome is "
               "signed in to a different claude.ai account/org than Claude Code.")
HANDOFF_TIMEOUT_S = 60.0
HANDOFF_PROMPT = (
    "[Session handoff: internal, not spoken.] This session is about to end. Save what your next "
    "session must know: call mcp__mac__remember with kind \"context\" for the 3 to 8 most useful "
    "durable facts, open tasks and decisions from this conversation (one short line each, newest "
    "first; the user may ask \"what were we doing?\" after a restart), and with other kinds for "
    "durable facts about the user you learned. Call no other tools; reply with no text."
)
CONSOLIDATE_PROMPT = (
    "\n\nAlso: the memory file is too long ({kb:.1f} KB). Call mcp__mac__rewrite_memory ONCE with a "
    "rewritten version of at most 4 KB: the same section headers, duplicates merged, stale items "
    "dropped, and every line ending in [kept] preserved verbatim. Current memory:\n```\n{memory}\n```"
)
_MEMORY_WORDS = re.compile(r"\b(remember|note that|make a note|save (this|that)|memori[sz]e|"
                           r"don'?t forget|keep in mind)\b", re.IGNORECASE)
_CHROME_SERVER = "claude-in-chrome"
_CHROME = "mcp__claude-in-chrome__"
LOCKED_NOTE = (
    "[System note: the Mac's screen is LOCKED. Voice, Chrome (Claude in Chrome), shell, files and "
    "AppleScript automation of apps (Mail, Calendar, Notes, Messages 'send') all work. Accessibility "
    "clicks and typing (mcp__mac__ui_click/ui_type/ui_key) and screenshots will NOT reach apps behind "
    "the lock screen, don't attempt them; prefer Chrome/AppleScript routes, or tell the user to unlock "
    "for that step. Never try to unlock the screen.]\n"
)
CHROME_DOWN_NOTE = ("[System note: the Claude in Chrome browser extension is not connected. "
                    "Don't use mcp__claude-in-chrome tools.]\n")

DRAIN_TIMEOUT_S = 10.0
QUIET_AFTER_STEER_S = 4.0  # a result arrived mid-steer: finish the turn if nothing follows
CHROME_RECHECK_S = 60.0
CONNECT_TIMEOUT_S = 60.0
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_S = 30.0
_SETTING_SOURCES = ("user", "project", "local")  # SDK SettingSource literal


@dataclass
class _Cmd:
    kind: str  # ask | steer | interrupt | reset | warm | sleep | close | msg | reader_done
    turn_id: int = 0
    text: str = ""
    gen: int = 0
    payload: Any = None


@dataclass
class _Turn:
    turn_id: int
    deadline: float
    texts: list[str] = field(default_factory=list)
    interrupted: bool = False
    timed_out: bool = False
    drain_deadline: float | None = None
    quiet_deadline: float | None = None
    last_result: ResultMessage | None = None
    tools: dict[str, str] = field(default_factory=dict)
    saw_delta: bool = False
    emitted_text: bool = False
    assistant_error: str | None = None
    queries: int = 1
    handoff: bool = False  # silent, memory-tools-only query (session handoff)
    probe: bool = False  # silent, tool-less connect_check query
    after: Callable[[], Any] | None = None  # deferred action once a handoff turn ends
    future: asyncio.Future[bool] | None = None

    @property
    def draining(self) -> bool:
        return self.interrupted or self.timed_out


def default_cli_path(cfg: config.Config) -> str | None:
    if cfg.brain.cli_path:
        return cfg.brain.cli_path
    found = shutil.which("claude")
    if found:
        return found
    local = Path.home() / ".local/bin/claude"
    return str(local) if local.exists() else None


def _chrome_running() -> bool:
    try:
        return mac_tools._find_app("Google Chrome") is not None
    except Exception:  # noqa: BLE001
        return False


def _frontmost_app() -> str:
    try:
        return mac_tools._frontmost_name()
    except Exception:  # noqa: BLE001
        return ""


def _kill_process_tree(pid: int) -> None:
    """SIGKILL `pid` and its descendants (the CLI's MCP children) if still alive."""
    pids = [pid]
    for p in pids:
        out = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True, check=False).stdout
        pids.extend(int(x) for x in out.split() if x.isdigit())
    for p in reversed(pids):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, signal.SIGKILL)


class ClaudeBrain:
    """contracts.Brain backed by the Claude Code CLI via claude-agent-sdk."""

    def __init__(self, cfg: config.Config, gate: ConfirmationGate,
                 chrome_status_cb: Callable[[bool], None] | None = None, *,
                 on_notify: Callable[[str, str, str | None], Any] | None = None,
                 cwd: Path | None = None, memory_path: Path | None = None,
                 client_factory: Callable[[ClaudeAgentOptions], Any] | None = None,
                 recent_user_text: Callable[[], str | list[str]] | None = None,
                 chrome_running: Callable[[], bool] = _chrome_running,
                 frontmost: Callable[[], str] = _frontmost_app,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.gate = gate
        self.chrome_status_cb = chrome_status_cb
        self.cwd = Path(cwd) if cwd else config.WORKSPACE_DIR
        self.memory_path = memory_path or (self.cwd / "memory.md" if cwd else config.MEMORY_PATH)
        self.results_dir = Path(getattr(cfg, "results_dir", "~/Documents/Mac")).expanduser()
        self._client_factory = client_factory or ClaudeSDKClient
        self._recent_user_text = recent_user_text  # B2 hook: last ~10 user utterances, newest last
        self._memory = memory.MemoryFile(self.memory_path)
        self._chrome_running = chrome_running
        self._frontmost = frontmost
        self._clock = clock
        self._wall = wall_clock

        self._inbox: asyncio.Queue[_Cmd] = asyncio.Queue()
        self._out: asyncio.Queue[BrainEvent | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._client: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._gen = 0
        self._active: _Turn | None = None
        self._pending: deque[_Cmd] = deque()
        self._cancelled: set[int] = set()
        self._stderr: deque[str] = deque(maxlen=200)
        self._failures = 0
        self._breaker_until = 0.0
        self._last_error: errors.ClassifiedError | None = None
        self._rate_reset_at: float | None = None
        self._session_id: str | None = None
        self._resume_next: str | None = None
        self._last_activity = self._wall()
        self._idle_at = self._clock()  # when the client last became idle (monotonic)
        self._init_at = 0.0  # when the last init message (Chrome probe) arrived
        self._sleep_requested = False  # F3: sleep() arrived mid-turn; honoured at the next idle
        self._session_started = self._wall()  # wall time the current session was created
        self._handed_off = False  # a handoff ran for the current session
        self._turns_this_session = 0  # nothing to hand off when 0
        self._init_info: dict[str, Any] = {}  # last init message: mcp server statuses, CLI version
        self.last_check: dict[str, Any] | None = None
        self.locked = False  # set_context(): screen locked → every ask carries LOCKED_NOTE
        self._orphans: list[tuple[Any, asyncio.Task[None] | None]] = []  # F2: clients of a dead owner
        self.chrome_connected: bool | None = None

        self._handlers: dict[str, Callable[[_Cmd], Any]] = {
            "ask": self._on_ask, "steer": self._on_steer, "interrupt": self._on_interrupt,
            "reset": self._on_reset, "warm": self._on_warm, "sleep": self._on_sleep,
            "msg": self._on_msg, "reader_done": self._on_reader_done, "handoff": self._on_handoff,
            "connect_check": self._on_connect_check,
        }
        self.hook = PreToolHook(
            gate, self._current_turn, self.is_cancelled, self._memory_check,
            restrict=self._restrict, on_decision=self._on_hook_decision)
        self.tools = MacTools(gate=gate, current_turn=self._current_turn,
                              memory_check=self._memory_check, on_say=self._on_say,
                              on_confirm=self._on_confirm, pending_action=self.hook.pending_for,
                              on_notify=on_notify, workspace=self.cwd, memory_path=self.memory_path,
                              results_dir=self.results_dir)
        self.hook.context = GateContext(
            workspace=str(self.cwd), element_label=self.tools.registry.label,
            chrome_ref_label=self.tools.chrome_refs.label, chrome_tab_host=self.tools.chrome_tabs.host,
            current_app=self._current_app)

    # ------------------------------------------------------------------ Brain protocol
    async def start(self) -> None:
        if self._task is not None:
            return
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        claude_md = self.cwd / "CLAUDE.md"
        if not claude_md.exists():
            claude_md.write_text(prompts.WORKSPACE_CLAUDE_MD)
        self._resume_next = await asyncio.to_thread(self._load_session)
        self._spawn_owner()

    async def close(self) -> None:
        if self._task is None:
            return
        self._closing = True
        pid = self._cli_pid()
        self._inbox.put_nowait(_Cmd("close"))
        try:
            await asyncio.wait_for(asyncio.shield(self._task), 15)
        except TimeoutError:
            log.warning("brain owner did not stop in time; cancelling")
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        if pid:
            await asyncio.to_thread(_kill_process_tree, pid)
        self._task = None
        self._closing = False

    async def events(self) -> AsyncIterator[BrainEvent]:  # type: ignore[override]
        while True:
            ev = await self._out.get()
            if ev is None:
                return
            yield ev

    async def ask(self, turn_id: int, text: str) -> None:
        if self._breaker_open():
            wait = int(self._breaker_until - self._clock()) + 1
            last = self._last_error
            kind = last.kind if last else BrainErrorKind.INTERNAL
            msg = (last.message if last and kind in (BrainErrorKind.AUTH, BrainErrorKind.RATE_LIMIT)
                   else f"I'm having trouble reaching Claude. I'll try again in about {wait} seconds.")
            self._emit(BrainEventKind.ERROR, turn_id, text=msg, is_error=True, error_kind=kind,
                       data={"circuit_open": True, "retry_in_s": wait})
            return
        self._put(_Cmd("ask", turn_id, text))

    async def steer(self, turn_id: int, text: str) -> None:
        self._put(_Cmd("steer", turn_id, text))

    async def interrupt(self, turn_id: int) -> None:
        if turn_id:
            self._cancelled.add(turn_id)  # the hook denies tools from now on
        self._put(_Cmd("interrupt", turn_id))

    async def reset(self) -> None:
        self._put(_Cmd("reset"))

    async def warm(self) -> None:
        """Start the CLI now so the first ask() is fast. Idempotent; never raises."""
        if self._task is not None:
            self._put(_Cmd("warm"))

    async def sleep(self) -> None:
        """Shut the CLI down, keeping the session id for the next ask(). Idempotent.
        A running turn is never killed: sleep() mid-turn is honoured when the turn ends
        (unless a new ask arrives first)."""
        if self._task is not None:
            self._put(_Cmd("sleep"))

    async def handoff(self) -> bool:
        """Run the silent session-handoff query (memory tools only, ≤ 60 s) if the CLI is alive
        and idle. Returns True when it completed. B2 calls this before rotation/reset."""
        if self._task is None:
            return False
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._put(_Cmd("handoff", payload=fut))
        try:
            return await asyncio.wait_for(fut, HANDOFF_TIMEOUT_S + 15)
        except TimeoutError:
            return False

    async def connect_check(self) -> dict[str, Any]:
        """Connect (if needed), read the CLI's init message and run one silent tool-less probe
        query; report auth / Chrome / version for the app's Permissions screen (≤ 25 s)."""
        if self.cfg.brain.auth == "claude_cli" and default_cli_path(self.cfg) is None:
            return self._check_result(errors.ClassifiedError(
                BrainErrorKind.CLI_MISSING, errors.MESSAGES[BrainErrorKind.CLI_MISSING], "claude not found"))
        if self._task is None:
            return self._check_result(errors.ClassifiedError(BrainErrorKind.INTERNAL, "brain not started"))
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._put(_Cmd("connect_check", payload=fut))
        try:
            return await asyncio.wait_for(fut, CONNECT_CHECK_TIMEOUT_S)
        except TimeoutError:
            return self._check_result(errors.ClassifiedError(
                BrainErrorKind.TIMEOUT, "Claude did not answer within 25 seconds.", "connect_check timeout"))

    def _check_result(self, err: errors.ClassifiedError | None, session_ok: bool | None = None) -> dict[str, Any]:
        auth, auth_detail = "ok", ""
        if err is not None:
            auth_detail = err.message
            if err.kind is BrainErrorKind.AUTH:
                auth = "not_signed_in"
            elif err.kind is BrainErrorKind.CLI_MISSING:
                auth = "cli_missing"
            elif err.kind is BrainErrorKind.RATE_LIMIT:
                auth = "ok"  # signed in, just throttled
            else:
                auth = "error"
        if auth == "ok" and self.cfg.brain.auth == "api_key":
            auth = "api_key"
        servers = self._init_info.get("servers") or []
        status = next((x.get("status") for x in servers
                       if isinstance(x, dict) and x.get("name") == _CHROME_SERVER), None)
        if self.cfg.brain.auth == "api_key":
            chrome, chrome_detail = "unavailable_api_key", (
                "Claude in Chrome needs a claude.ai sign-in (Pro/Max/Team/Enterprise); it stays off "
                "with an API key.")
        elif not self.cfg.brain.chrome:
            chrome, chrome_detail = "unknown", "Chrome integration is disabled in config (brain.chrome)."
        elif status == "connected":
            chrome, chrome_detail = "connected", "Claude in Chrome extension connected."
        elif status is None and not self._init_info:
            chrome, chrome_detail = "unknown", "Claude Code has not reported its tools yet."
        else:
            chrome, chrome_detail = "not_connected", f"Status: {status or 'not listed'}. {CHROME_HINT}"
        result = {
            "auth": auth, "auth_detail": auth_detail, "chrome": chrome, "chrome_detail": chrome_detail,
            "cli_version": str(self._init_info.get("version") or ""),
            "session_ok": bool(session_ok) if session_ok is not None else err is None,
        }
        self.last_check = result
        return result

    async def _on_connect_check(self, cmd: _Cmd) -> None:
        fut: asyncio.Future[dict[str, Any]] = cmd.payload
        if self._active is not None:  # busy: answer from what we know rather than interrupt the user
            fut.set_result(self.last_check or self._check_result(
                errors.ClassifiedError(BrainErrorKind.INTERNAL, "Mac is busy with a request; try again.")))
            return
        err = await self._ensure_connected()
        if err is not None:
            fut.set_result(self._check_result(err))
            return
        self._active = _Turn(PROBE_TURN, self._clock() + CONNECT_CHECK_TIMEOUT_S - 3, probe=True, future=fut)  # type: ignore[arg-type]
        try:
            await self._client.query(PROBE_PROMPT)
        except Exception as e:  # noqa: BLE001
            await self._fail_active(errors.classify_exception(e, self.stderr_tail()), reconnect=True)

    def _end_probe(self, a: _Turn, err: errors.ClassifiedError | None) -> None:
        if a.future is not None and not a.future.done():
            a.future.set_result(self._check_result(err, session_ok=err is None))

    async def set_context(self, locked: bool) -> None:
        """Screen lock state from the app; while locked every ask carries LOCKED_NOTE."""
        if locked != self.locked:
            log.info("brain: screen %s", "locked" if locked else "unlocked")
        self.locked = bool(locked)

    async def set_name(self, name: str) -> None:
        """Rename the persona for the next connect; a live session is parked (id kept)."""
        self.cfg.name = name.strip() or "Mac"
        await self.sleep()

    async def set_language(self, lang: str) -> None:
        """Change the spoken language (TTS lang code) for the next connect; session id kept."""
        self.cfg.tts.lang = lang
        await self.sleep()

    @property
    def alive(self) -> bool:
        """True while the Claude CLI process is running (connected)."""
        return self._client is not None

    # ------------------------------------------------------------------ owner task
    def _spawn_owner(self) -> None:
        self._task = asyncio.create_task(self._owner(), name="claude-brain-owner")

    def _put(self, cmd: _Cmd) -> None:
        t = self._task
        if t is not None and t.done() and not self._closing:  # N4: self-heal a dead owner
            log.critical("brain owner task died (%s); restarting it", t.exception() if not t.cancelled() else "cancelled")
            if self._client is not None:
                self._orphans.append((self._client, self._reader))  # F2: reaped by the new owner
            self._client, self._reader, self._active = None, None, None
            self._gen += 1
            self._spawn_owner()
        self._inbox.put_nowait(cmd)

    async def _owner(self) -> None:
        try:
            while self._orphans:
                client, reader = self._orphans.pop()
                await self._close_client(client, reader)
            while True:
                try:
                    try:
                        cmd = await asyncio.wait_for(self._inbox.get(), self._next_wakeup())
                    except TimeoutError:
                        await self._on_tick()
                        continue
                    if cmd.kind == "close":
                        break
                    await self._handlers[cmd.kind](cmd)
                except Exception:  # N4: nothing may kill the owner silently
                    log.exception("brain owner: unexpected error; recovering")
                    await self._fail_active(errors.ClassifiedError(
                        BrainErrorKind.INTERNAL, errors.CONNECTION_LOST), reconnect=True)
        finally:
            a = self._active
            if a is not None and a.handoff:
                await self._end_handoff(a, ok=False)
            elif a is not None and a.probe:
                self._end_probe(a, errors.ClassifiedError(BrainErrorKind.INTERNAL, "Brain closed."))
            elif a is not None and not a.timed_out:
                self._emit(BrainEventKind.DONE, a.turn_id, data={"interrupted": True})
            self._active = None
            await self._disconnect()
            if self._closing:
                self._out.put_nowait(None)

    def _next_wakeup(self) -> float | None:
        a = self._active
        if a is not None:
            target = a.drain_deadline if a.draining and a.drain_deadline else a.deadline
            if a.quiet_deadline is not None:
                target = min(target, a.quiet_deadline)
        elif self._client is not None and self.cfg.brain.idle_shutdown_min > 0:
            target = self._idle_at + self.cfg.brain.idle_shutdown_min * 60
        else:
            return None
        return max(0.01, target - self._clock())

    # ------------------------------------------------------------------ helpers for hook/tools
    def _current_turn(self) -> int:
        return self._active.turn_id if self._active else 0

    def _current_app(self) -> str:
        return self._frontmost() or self.tools.registry.current_app

    def is_cancelled(self, turn_id: int) -> bool:
        if not turn_id or turn_id in self._cancelled:
            return True
        a = self._active
        return bool(a and a.turn_id == turn_id and a.draining)

    def _memory_check(self, tool: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Capability check for memory writes: (allowed, "explicit"|"grounded"|"handoff"|reason)."""
        a = self._active
        if a is None or a.draining:
            return False, "no active turn"
        if tool == "rewrite_memory":
            return (True, "handoff") if a.handoff else (False, "only during a session handoff")
        if a.handoff and str(args.get("kind", "")).lower() == "context":
            return True, "handoff"
        if any(_MEMORY_WORDS.search(t) for t in a.texts):
            return True, "explicit"
        text = str(args.get("fact") or args.get("text") or "")
        recent: str | list[str] = self._recent_user_text() if self._recent_user_text else a.texts
        if memory.grounded(text, "\n".join(recent) if isinstance(recent, list) else recent):
            return True, "grounded"
        return False, "memory must come from the user (what they said), not from pages or tool results"

    def _restrict(self, tool: str) -> str | None:
        a = self._active
        if a is not None and a.probe:
            return "Connection check: no tools; just reply OK."
        if a is not None and a.handoff and tool not in MEMORY_TOOLS:
            return "Session handoff: only mcp__mac__remember / forget / rewrite_memory may be called."
        return None

    def _on_say(self, text: str) -> None:
        turn = self._current_turn()
        if turn and not self.is_cancelled(turn):
            self._emit(BrainEventKind.SAY, turn, text=text)

    def _on_confirm(self, summary: str, target: str) -> None:
        turn = self._current_turn()
        if turn and not self.is_cancelled(turn):
            self._emit(BrainEventKind.CONFIRM, turn, text=summary, data={"target": target})

    def _on_hook_decision(self, d: HookDecision) -> None:
        if not d.allowed and not self.is_cancelled(d.turn_id):
            self._emit(BrainEventKind.STATUS, d.turn_id,
                       data={"gate": {"tool": d.tool_name, "verdict": d.verdict, "code": d.code,
                                      "allowed": False}})

    def stderr_tail(self, n: int = 40) -> str:
        return "\n".join(list(self._stderr)[-n:])

    def _emit(self, kind: BrainEventKind, turn_id: int, **kw: Any) -> None:
        if turn_id < 0:
            return  # handoff query: internal, never spoken
        self._out.put_nowait(BrainEvent(kind=kind, turn_id=turn_id, **kw))

    # ------------------------------------------------------------------ breaker
    def _breaker_open(self) -> bool:
        return self._clock() < self._breaker_until

    def _record_failure(self, err: errors.ClassifiedError) -> None:
        self._failures += 1
        self._last_error = err
        if self._failures >= BREAKER_FAILURES:
            self._breaker_until = self._clock() + BREAKER_COOLDOWN_S
            log.warning("brain circuit breaker open for %.0fs after %d failures",
                        BREAKER_COOLDOWN_S, self._failures)

    def _record_success(self) -> None:
        self._failures = 0
        self._breaker_until = 0.0

    # ------------------------------------------------------------------ session state
    def _load_session(self) -> str | None:
        brain = config.read_state().get("brain")
        if not isinstance(brain, dict):
            return None
        sid, ts = brain.get("session_id"), float(brain.get("updated_at", 0) or 0)
        if sid and self._wall() - ts < self.cfg.brain.session_idle_rotate_min * 60:
            self._session_id = str(sid)
            self._last_activity = ts
            self._session_started = float(brain.get("started_at", ts) or ts)
            log.info("brain: resuming session %s", sid)
            return str(sid)
        return None

    async def _save_session(self, session_id: str | None) -> None:
        try:
            await asyncio.to_thread(config.update_state, brain={
                "session_id": session_id, "updated_at": self._wall(),
                "started_at": self._session_started})
        except OSError:
            log.exception("brain: could not persist session state")

    # ------------------------------------------------------------------ options / connection
    def build_options(self, resume: str | None, api_key: str | None = None) -> ClaudeAgentOptions:
        """Pure except for reading memory.md; callers run it in a thread."""
        cfg = self.cfg.brain
        env = {"ANTHROPIC_API_KEY": api_key} if api_key else {}
        persona = prompts.persona_prompt(self.cfg, browser_available=self.chrome_connected is not False,
                                         memory_path=self.memory_path, results_dir=self.results_dir)
        opts = ClaudeAgentOptions(
            system_prompt={"type": "preset", "preset": "claude_code", "append": persona},
            permission_mode="bypassPermissions",
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[self.hook])]},
            mcp_servers={"mac": self.tools.server()},
            cwd=str(self.cwd),
            setting_sources=self._setting_sources(),
            include_partial_messages=True,
            extra_args={"chrome": None} if cfg.chrome else {},
            env=env,
            stderr=self._stderr.append,
            resume=resume,
            model=cfg.model or None,
        )
        if cfg.auth == "claude_cli":
            opts.cli_path = default_cli_path(self.cfg)
        return opts

    def _setting_sources(self) -> list[Any]:
        wanted = list(self.cfg.brain.setting_sources)
        bad = [x for x in wanted if x not in _SETTING_SOURCES]
        if bad:
            log.warning("brain: ignoring unknown setting_sources %s (valid: %s)", bad, _SETTING_SOURCES)
        return [x for x in wanted if x in _SETTING_SOURCES]

    def _cli_pid(self) -> int | None:
        proc = getattr(getattr(self._client, "_transport", None), "_process", None)
        pid = getattr(proc, "pid", None)
        return int(pid) if isinstance(pid, int) else None

    async def _connect(self, resume: str | None) -> errors.ClassifiedError | None:
        api_key = None
        if self.cfg.brain.auth == "api_key":
            api_key = await asyncio.to_thread(config.get_secret, "ANTHROPIC_API_KEY")
            if not api_key:
                return errors.ClassifiedError(BrainErrorKind.AUTH,
                                              "I don't have a Claude API key. Please add one in settings.",
                                              "ANTHROPIC_API_KEY missing")
        if self.cfg.brain.auth == "claude_cli" and default_cli_path(self.cfg) is None:
            return errors.ClassifiedError(BrainErrorKind.CLI_MISSING,
                                          errors.MESSAGES[BrainErrorKind.CLI_MISSING], "claude not found")
        err: errors.ClassifiedError | None = None
        for attempt in ([resume, resume, None] if resume else [None]):  # N9: retry with resume first
            client = self._client_factory(await asyncio.to_thread(self.build_options, attempt, api_key))
            try:
                await asyncio.wait_for(client.connect(), CONNECT_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                err = errors.classify_exception(e, self.stderr_tail())
                log.warning("brain connect failed (%s, resume=%s): %s", err.kind.value, attempt, e)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.disconnect(), 5)
                continue
            self._client = client
            self._gen += 1
            self._reader = asyncio.create_task(self._read(client, self._gen), name="claude-brain-reader")
            self._session_id = attempt
            if attempt is None:
                self._session_started = self._wall()
            self._handed_off = False
            self._turns_this_session = 0
            self._idle_at = self._clock()
            log.info("brain connected (gen %d, resume=%s)", self._gen, attempt)
            return None
        return err

    async def _read(self, client: Any, gen: int) -> None:
        try:
            async for msg in client.receive_messages():
                self._inbox.put_nowait(_Cmd("msg", gen=gen, payload=msg))
            self._inbox.put_nowait(_Cmd("reader_done", gen=gen))
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # forwarded to the owner  # noqa: BLE001
            self._inbox.put_nowait(_Cmd("reader_done", gen=gen, payload=e))

    async def _disconnect(self) -> None:
        reader, client = self._reader, self._client
        self._reader, self._client = None, None
        self._gen += 1  # anything still queued from the old reader is stale
        await self._close_client(client, reader)

    async def _close_client(self, client: Any, reader: asyncio.Task[None] | None) -> None:
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(BaseException):
                await reader
        if client is not None:
            pid = getattr(getattr(getattr(client, "_transport", None), "_process", None), "pid", None)
            try:
                await asyncio.wait_for(client.disconnect(), 10)
            except Exception:
                log.warning("brain disconnect failed; killing the CLI tree", exc_info=True)
                if isinstance(pid, int):
                    await asyncio.to_thread(_kill_process_tree, pid)

    # ------------------------------------------------------------------ commands
    async def _on_ask(self, cmd: _Cmd) -> None:
        self._sleep_requested = False  # new work cancels a pending sleep
        if cmd.turn_id in self._cancelled:
            self._emit(BrainEventKind.DONE, cmd.turn_id, data={"interrupted": True})
            self._cancelled.discard(cmd.turn_id)
            return
        a = self._active
        if a is not None:
            if not a.draining:
                log.info("brain: new ask %d supersedes running turn %d", cmd.turn_id, a.turn_id)
                await self._begin_interrupt(a)
            self._pending.append(cmd)
            return
        await self._start_turn(cmd)

    async def _ensure_connected(self) -> errors.ClassifiedError | None:
        """Lazy connect (ask/warm). Rotates the session if idle for too long."""
        idle = self._wall() - self._last_activity
        if idle > self.cfg.brain.session_idle_rotate_min * 60 and (self._client or self._resume_next):
            log.info("brain: rotating session after %.0f idle minutes", idle / 60)
            await self._disconnect()
            self._resume_next = None
        if self._client is not None:
            return None
        err = await self._connect(self._resume_next)
        self._resume_next = None
        if err is not None:
            self._record_failure(err)
        return err

    async def _on_warm(self, cmd: _Cmd) -> None:
        self._sleep_requested = False
        if self._client is not None or self._active is not None or self._breaker_open():
            return
        err = await self._ensure_connected()
        if err is not None:
            log.warning("brain: warm failed (%s); will retry on ask", err.kind.value)

    async def _on_sleep(self, cmd: _Cmd) -> None:
        if self._client is None:
            return
        if self._active is not None:
            log.info("brain: sleep requested mid-turn; will sleep when the turn ends")
            self._sleep_requested = True
            return
        self._sleep_requested = False
        await self._park_session()
        log.info("brain: asleep (session %s kept for resume)", self._resume_next)

    async def _on_handoff(self, cmd: _Cmd) -> None:
        fut: asyncio.Future[bool] | None = cmd.payload
        if (self._client is None or self._active is not None or self._handed_off
                or self._turns_this_session == 0):  # idempotent and free when there's nothing to save
            if fut is not None and not fut.done():
                fut.set_result(False)
            return
        await self._start_handoff(after=None, future=fut)

    async def _start_handoff(self, after: Callable[[], Any] | None,
                             future: asyncio.Future[bool] | None = None) -> None:
        """Begin the silent memory-tools-only handoff query; `after` runs when it ends."""
        prompt = HANDOFF_PROMPT
        if self._memory.needs_consolidation():
            text = await asyncio.to_thread(self._memory.text)
            prompt += CONSOLIDATE_PROMPT.format(kb=len(text.encode()) / 1024, memory=text[:12000])
        await asyncio.to_thread(self._memory.set_recent_context, [])
        now = self._clock()
        self._active = _Turn(HANDOFF_TURN, now + HANDOFF_TIMEOUT_S, texts=[], handoff=True,
                             after=after, future=future)
        self._handed_off = True
        log.info("brain: session handoff started")
        try:
            await self._client.query(prompt)
        except Exception as e:  # noqa: BLE001
            await self._fail_active(errors.classify_exception(e, self.stderr_tail()), reconnect=True)

    async def _end_handoff(self, a: _Turn, ok: bool) -> None:
        log.info("brain: session handoff %s", "done" if ok else "cut short")
        if a.future is not None and not a.future.done():
            a.future.set_result(ok)
        if a.after is not None:
            await a.after()

    def _session_age_s(self) -> float:
        return self._wall() - self._session_started

    async def _park_session(self) -> None:
        """Disconnect but keep the session id for the next connect."""
        sid = self._session_id
        await self._disconnect()
        self._resume_next = sid
        await self._save_session(sid)

    async def _start_turn(self, cmd: _Cmd) -> None:
        err = await self._ensure_connected()
        if cmd.turn_id in self._cancelled:  # N9: "stop" arrived while connecting
            self._cancelled.discard(cmd.turn_id)
            self._emit(BrainEventKind.DONE, cmd.turn_id, data={"interrupted": True})
            await self._next_pending()
            return
        if err is not None:
            self._emit(BrainEventKind.ERROR, cmd.turn_id, text=err.message, is_error=True,
                       error_kind=err.kind, data=err.as_data())
            return
        now = self._clock()
        self._active = _Turn(cmd.turn_id, now + self.cfg.brain.max_turn_s, texts=[cmd.text])
        self._last_activity = self._wall()
        self._turns_this_session += 1
        prompt = cmd.text
        if self.cfg.brain.chrome and self.chrome_connected is False and not self._chrome_recheck_due():
            prompt = CHROME_DOWN_NOTE + prompt
        if self.locked:
            prompt = LOCKED_NOTE + prompt
        try:
            await self._client.query(prompt)
        except Exception as e:  # noqa: BLE001
            await self._fail_active(errors.classify_exception(e, self.stderr_tail()), reconnect=True)

    def _chrome_recheck_due(self) -> bool:
        """UX#17: Chrome appeared after we were told it's missing → stop telling the model."""
        return self._clock() - self._init_at > CHROME_RECHECK_S and self._chrome_running()

    async def _on_steer(self, cmd: _Cmd) -> None:
        a = self._active
        if a is None or a.turn_id != cmd.turn_id:
            if a is None and cmd.turn_id not in self._cancelled:
                log.info("brain: steer for finished turn %d → new query", cmd.turn_id)
                await self._start_turn(cmd)
            else:
                log.info("brain: dropping steer for turn %d", cmd.turn_id)
            return
        if a.draining:
            return
        a.texts.append(cmd.text)
        a.queries += 1
        try:
            await self._client.query(cmd.text)
        except Exception as e:  # noqa: BLE001
            await self._fail_active(errors.classify_exception(e, self.stderr_tail()), reconnect=True)

    async def _on_interrupt(self, cmd: _Cmd) -> None:
        a = self._active
        if a is not None and a.turn_id == cmd.turn_id:
            if not a.draining:
                await self._begin_interrupt(a)
            return
        for queued in list(self._pending):
            if queued.turn_id == cmd.turn_id:
                self._pending.remove(queued)
                self._emit(BrainEventKind.DONE, cmd.turn_id, data={"interrupted": True})
        # Not running: the inbox is FIFO, so any earlier Ask for this turn was handled already.
        self._cancelled.discard(cmd.turn_id)

    async def _begin_interrupt(self, a: _Turn) -> None:
        a.interrupted = True
        a.drain_deadline = self._clock() + DRAIN_TIMEOUT_S
        self._cancelled.add(a.turn_id)
        if self.gate.pending and self.gate.pending[0] == a.turn_id:
            self.gate.resolve(False)
        if self._client is None:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            log.warning("brain: interrupt request failed", exc_info=True)

    async def _on_reset(self, cmd: _Cmd) -> None:
        a = self._active
        if a is None and self._client is not None and not self._handed_off and self._turns_this_session:
            await self._start_handoff(after=self._do_reset)  # save context, then start fresh
            return
        if a is not None:
            if not a.timed_out:
                self._emit(BrainEventKind.DONE, a.turn_id, data={"interrupted": True})
            self._active = None
        await self._do_reset()

    async def _do_reset(self) -> None:
        await self._disconnect()
        self._resume_next = None
        self._session_id = None
        await self._save_session(None)
        log.info("brain: reset, next ask starts a fresh session")
        await self._next_pending()

    # ------------------------------------------------------------------ timers
    async def _on_tick(self) -> None:
        a = self._active
        now = self._clock()
        if a is None:
            limit = self.cfg.brain.idle_shutdown_min * 60
            if self._client is not None and limit > 0 and now - self._idle_at >= limit:
                log.info("brain: idle for %d min; shutting the CLI down", self.cfg.brain.idle_shutdown_min)
                if not self._handed_off and self._session_age_s() > self.cfg.brain.session_idle_rotate_min * 60:
                    await self._start_handoff(after=lambda: self._on_sleep(_Cmd("sleep")))
                else:
                    await self._on_sleep(_Cmd("sleep"))
            return
        if a.quiet_deadline is not None and now >= a.quiet_deadline and a.last_result is not None:
            log.info("brain: no messages after the steer's result; finishing turn %d", a.turn_id)
            await self._finish_turn(a, a.last_result)
        elif a.draining and a.drain_deadline and now >= a.drain_deadline:
            log.warning("brain: drain of turn %d timed out; reconnecting", a.turn_id)
            if not a.timed_out:
                self._emit(BrainEventKind.DONE, a.turn_id, data={"interrupted": True})
            self._active = None
            self._cancelled.discard(a.turn_id)
            await self._park_session()
            if a.handoff:
                await self._end_handoff(a, ok=False)
            elif a.probe:
                self._end_probe(a, errors.ClassifiedError(BrainErrorKind.TIMEOUT, "Claude did not answer."))
            await self._next_pending()
        elif not a.draining and now >= a.deadline:
            log.warning("brain: turn %d exceeded max_turn_s", a.turn_id)
            err = errors.ClassifiedError(BrainErrorKind.TIMEOUT, errors.MESSAGES[BrainErrorKind.TIMEOUT])
            self._emit(BrainEventKind.ERROR, a.turn_id, text=err.message, is_error=True,
                       error_kind=err.kind)
            a.timed_out = True
            await self._begin_interrupt(a)

    # ------------------------------------------------------------------ failures
    async def _fail_active(self, err: errors.ClassifiedError, reconnect: bool = False) -> None:
        self._record_failure(err)
        a = self._active
        self._active = None
        if a is not None and a.handoff:
            await self._end_handoff(a, ok=False)
        elif a is not None and a.probe:
            self._end_probe(a, err)
        elif a is not None:
            self._cancelled.discard(a.turn_id)
            if a.interrupted and not a.timed_out:
                self._emit(BrainEventKind.DONE, a.turn_id, data={"interrupted": True})
            elif not a.timed_out:
                self._emit(BrainEventKind.ERROR, a.turn_id, text=err.message, is_error=True,
                           error_kind=err.kind, data=err.as_data())
        if reconnect:
            await self._park_session()
        await self._next_pending()

    async def _on_reader_done(self, cmd: _Cmd) -> None:
        if cmd.gen != self._gen:
            return
        exc = cmd.payload
        err = (errors.classify_exception(exc, self.stderr_tail()) if exc is not None
               else errors.classify_text(self.stderr_tail() or "Claude Code exited unexpectedly"))
        log.warning("brain: CLI stream ended (%s): %s", err.kind.value, exc or "EOF")
        await self._park_session()
        if self._active is None:
            self._record_failure(err)
            return
        await self._fail_active(err)

    async def _next_pending(self) -> None:
        while self._pending and self._active is None:
            await self._on_ask(self._pending.popleft())
        if self._active is None:
            self._idle_at = self._clock()
            if self._sleep_requested and self._client is not None:
                await self._on_sleep(_Cmd("sleep"))
            elif self._client is not None and self.chrome_connected is False and self._chrome_recheck_due():
                log.info("brain: Chrome appeared; reconnecting at idle to re-probe the extension")
                await self._park_session()

    # ------------------------------------------------------------------ messages
    async def _on_msg(self, cmd: _Cmd) -> None:
        if cmd.gen != self._gen:
            return
        msg = cmd.payload
        if isinstance(msg, SystemMessage):
            self._on_system(msg)
            return
        if isinstance(msg, RateLimitEvent):
            if msg.rate_limit_info.status == "rejected":
                self._rate_reset_at = msg.rate_limit_info.resets_at
            return
        a = self._active
        if a is None:
            log.debug("brain: dropping %s with no active turn", type(msg).__name__)
            return
        if isinstance(msg, ResultMessage):
            await self._on_result(a, msg)
            return
        a.quiet_deadline = None  # the turn is still producing after a steer result
        if a.draining:
            return
        if isinstance(msg, StreamEvent):
            self._on_stream_event(a, msg)
        elif isinstance(msg, AssistantMessage):
            self._on_assistant(a, msg)
        elif isinstance(msg, UserMessage):
            self._on_user(a, msg)

    def _on_system(self, msg: SystemMessage) -> None:
        if msg.subtype != "init":
            return
        data = msg.data or {}
        servers = data.get("mcp_servers") or []
        status = next((s.get("status") for s in servers
                       if isinstance(s, dict) and s.get("name") == _CHROME_SERVER), None)
        if data.get("session_id"):
            self._session_id = str(data["session_id"])
        self._init_at = self._clock()
        self._init_info = {"servers": servers, "version": data.get("claude_code_version", "")}
        connected: bool | None
        if not self.cfg.brain.chrome:
            connected = None
        elif status == "pending":
            connected = self.chrome_connected  # extension still connecting: keep last known
        else:
            connected = status == "connected"
        changed = connected != self.chrome_connected
        self.chrome_connected = connected
        turn = self._current_turn()
        if not turn or not self.is_cancelled(turn):
            self._emit(BrainEventKind.STATUS, turn, data={
                "chrome_connected": connected, "session_id": data.get("session_id"),
                "mcp_servers": servers, "model": data.get("model"), "init": True})
        if changed and self.chrome_status_cb and connected is not None:
            try:
                self.chrome_status_cb(connected)
            except Exception:
                log.exception("chrome_status_cb failed")

    def _on_stream_event(self, a: _Turn, msg: StreamEvent) -> None:
        if msg.parent_tool_use_id:
            return
        ev = msg.event or {}
        etype = ev.get("type")
        if etype == "message_start":
            a.saw_delta = False
        elif etype == "content_block_start":
            if (ev.get("content_block") or {}).get("type") == "text" and a.emitted_text:
                self._emit(BrainEventKind.TEXT, a.turn_id, text="\n", data={"block_start": True})
        elif etype == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                a.saw_delta = True
                a.emitted_text = True
                self._emit(BrainEventKind.TEXT, a.turn_id, text=delta["text"], data={"partial": True})

    def _on_assistant(self, a: _Turn, msg: AssistantMessage) -> None:
        if msg.parent_tool_use_id:
            return  # sub-agent internals are not spoken
        if msg.error:
            a.assistant_error = str(msg.error)
        streamed = a.saw_delta
        for block in msg.content:
            if isinstance(block, TextBlock):
                if not streamed and block.text:
                    if a.emitted_text:
                        self._emit(BrainEventKind.TEXT, a.turn_id, text="\n", data={"block_start": True})
                    a.emitted_text = True
                    self._emit(BrainEventKind.TEXT, a.turn_id, text=block.text)
            elif isinstance(block, ToolUseBlock):
                a.tools[block.id] = block.name
                inp = block.input or {}
                if block.name == f"{_CHROME}navigate":
                    self.tools.chrome_tabs.note_navigate(inp.get("tabId"), str(inp.get("url", "")))
                if block.name in _SILENT_TOOLS:
                    continue
                self._emit(BrainEventKind.TOOL_START, a.turn_id, tool_name=block.name,
                           tool_input=dict(inp), tool_use_id=block.id,
                           data={"narration": describe_tool(block.name, inp)})
        a.saw_delta = False

    def _on_user(self, a: _Turn, msg: UserMessage) -> None:
        if msg.parent_tool_use_id or isinstance(msg.content, str):
            return
        for block in msg.content:
            if not isinstance(block, ToolResultBlock):
                continue
            name = a.tools.get(block.tool_use_id, "")
            text = _result_text(block.content)
            if name in (f"{_CHROME}find", f"{_CHROME}read_page"):
                self.tools.chrome_refs.ingest(text)
            elif name in (f"{_CHROME}tabs_context_mcp", f"{_CHROME}tabs_create_mcp", f"{_CHROME}navigate"):
                self.tools.chrome_tabs.ingest(text)
            if name in _SILENT_TOOLS:
                continue
            self._emit(BrainEventKind.TOOL_END, a.turn_id, tool_name=name, tool_use_id=block.tool_use_id,
                       is_error=bool(block.is_error), text=text[:500])

    async def _on_result(self, a: _Turn, msg: ResultMessage) -> None:
        if msg.session_id:
            self._session_id = msg.session_id
            await self._save_session(msg.session_id)
        self._last_activity = self._wall()
        if a.queries > 1:
            # A steer was sent mid-turn. If the CLI answers each query with its own result, the
            # turn continues (or, when draining, the later results are swallowed here instead of
            # landing on the next turn); if it folded the steer into one result, nothing else
            # arrives and the quiescence timer finishes the turn (N3).
            a.queries -= 1
            a.last_result = msg
            a.quiet_deadline = self._clock() + QUIET_AFTER_STEER_S
            return
        await self._finish_turn(a, msg)

    async def _finish_turn(self, a: _Turn, msg: ResultMessage) -> None:
        self._active = None
        self._cancelled.discard(a.turn_id)
        if a.handoff:
            await self._end_handoff(a, ok=not a.draining)
            await self._next_pending()
            return
        if a.probe:
            err = errors.classify_result(msg, self.stderr_tail(10))
            if err is None and a.assistant_error:
                err = errors.classify_assistant_error(a.assistant_error, msg.result or "")
            if err is None and a.draining:
                err = errors.ClassifiedError(BrainErrorKind.INTERNAL, "The check was interrupted.")
            self._end_probe(a, err)
            if err is not None and err.kind is not BrainErrorKind.RATE_LIMIT:
                self._record_failure(err)
            await self._next_pending()
            return
        data = {"session_id": msg.session_id, "cost_usd": msg.total_cost_usd,
                "duration_ms": msg.duration_ms, "num_turns": msg.num_turns,
                "interrupted": a.draining}
        if a.timed_out:
            pass  # ERROR(timeout) was already emitted
        elif a.interrupted:
            self._emit(BrainEventKind.DONE, a.turn_id, data=data)
        else:
            err = errors.classify_result(msg, self.stderr_tail(10))
            if err is None and a.assistant_error:
                err = errors.classify_assistant_error(a.assistant_error, msg.result or "")
            if err is not None:
                if err.kind is BrainErrorKind.RATE_LIMIT and self._rate_reset_at:
                    err = errors.ClassifiedError(err.kind, errors.classify_text(
                        "usage limit", resets_at=self._rate_reset_at).message, err.detail)
                self._record_failure(err)
                self._emit(BrainEventKind.ERROR, a.turn_id, text=err.message, is_error=True,
                           error_kind=err.kind, data={**data, **err.as_data()})
            else:
                self._record_success()
                self._emit(BrainEventKind.DONE, a.turn_id, data=data)
        await self._next_pending()


def _result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image":
                parts.append("[image]")
    return "\n".join(parts)
