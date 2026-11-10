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

Profiles (WhatsApp ARCHITECTURE W3.1): every constructor argument after ``wall_clock`` defaults
to today's voice behaviour; `CHAT_PROFILE` holds the chat-channel values. They cover the session
store (M3), a shared memory file (F11), session policy (F9: no idle rotation, no handoff, a
``session_fallback`` status when a resume fails), ``steer_only`` (F12), the WhatsApp gate rules
(M1), the ``.env`` exemption (F10), ``mac.send_file`` (ATTACH events) and the desktop lease (F3),
which is released at turn teardown, while confirming and at idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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
from macd.brain.gate import MAC_SEND_FILE, ConfirmationGate, GateContext, analyze
from macd.brain.hooks import (
    CHAT_DESKTOP_WAIT_S,
    CONFIRM_TOOL,
    HOOK_TIMEOUT_S,
    MEMORY_TOOLS,
    REASON_DESKTOP_BUSY_CHAT,
    REASON_DESKTOP_BUSY_VOICE,
    VOICE_DESKTOP_WAIT_S,
    DesktopGuard,
    DesktopLock,
    HookDecision,
    PreToolHook,
    default_desktop_locks,
    is_desktop_tool,
)
from macd.brain.mac_tools import MacTools
from macd.brain.narration import describe_tool
from macd.contracts import BrainErrorKind, BrainEvent, BrainEventKind

log = logging.getLogger(__name__)

SAY_TOOL = "mcp__mac__say"
_SILENT_TOOLS = {SAY_TOOL, CONFIRM_TOOL}
_SWITCH_TOOL = "mcp__mac__switch_workspace"
HANDOFF_TURN = -1  # turn id of the silent session-handoff query (never emitted)
PROBE_TURN = -2  # turn id of the silent connect_check probe query (never emitted)
STEER_REPLY_TIMEOUT_S = 15.0
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
_FROM_CONFIG = "config"  # rotate_after_idle_min default: cfg.brain.session_idle_rotate_min
_POST_TOOL_MATCHER = "mcp__claude-in-chrome__.*|mcp__mac__ui_.*|mcp__mac__screenshot|mcp__mac__open_app"
SettingSourcesFn = Callable[[], "Sequence[str] | Awaitable[Sequence[str]]"]


# ----------------------------------------------------------------------------- sessions (M3)


@dataclass
class SessionInfo:
    session_id: str | None
    updated_at: float = 0.0  # wall time of the last use (0: unknown)
    started_at: float = 0.0  # wall time the session was created
    fork: bool = False  # resume as a fork: a session Mac did not create is never written to


class SessionStore(Protocol):
    """Where a brain keeps its session id. Voice (local mode): state.json; chat: agentd's
    conversation record. Both methods are async so a store may use its own asyncio lock."""

    async def load(self) -> SessionInfo | None: ...
    async def save(self, info: SessionInfo) -> None: ...


class StateJsonSessionStore:
    """Today's store: ``state.json["brain"]`` (voice in local mode, the default)."""

    async def load(self) -> SessionInfo | None:
        brain = (await asyncio.to_thread(config.read_state)).get("brain")
        if not isinstance(brain, dict) or not brain.get("session_id"):
            return None
        ts = float(brain.get("updated_at", 0) or 0)
        return SessionInfo(str(brain["session_id"]), ts, float(brain.get("started_at", ts) or ts))

    async def save(self, info: SessionInfo) -> None:
        await asyncio.to_thread(config.update_state, brain={
            "session_id": info.session_id, "updated_at": info.updated_at, "started_at": info.started_at})


# The chat column of the W3.1 profile table (agentd adds cwd, session_store, memory_file,
# setting_sources, never_chat_names and a cfg carrying [chat_brain]).
CHAT_PROFILE: dict[str, Any] = {
    "steer_only": True, "rotate_after_idle_min": None, "handoff_enabled": False,
    "on_resume_fallback": "status", "max_turn_s": 2700.0, "write_claude_md": False,
    "whatsapp_ui_block": True, "send_file": True, "switch_workspace": True,
    "channel": "chat", "desktop_wait_s": CHAT_DESKTOP_WAIT_S,
    "desktop_busy_reason": REASON_DESKTOP_BUSY_CHAT, "confirm_timeout_s": 600.0,
}


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
    switch: tuple[str, str] | None = None  # (folder, mode) the model asked to move to (AG1)
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
                 wall_clock: Callable[[], float] = time.time,
                 # -- profile (WhatsApp ARCHITECTURE W3.1); defaults = the voice profile ----------
                 session_store: SessionStore | None = None,
                 memory_file: memory.MemoryFile | None = None,
                 steer_only: bool = False,
                 rotate_after_idle_min: float | str | None = _FROM_CONFIG,
                 handoff_enabled: bool = True,
                 on_resume_fallback: Literal["silent", "status"] = "silent",
                 max_turn_s: float | None = None,
                 write_claude_md: bool = True,
                 setting_sources: Sequence[str] | SettingSourcesFn | None = None,
                 env_exempt_dir: Path | None = None,
                 whatsapp_ui_block: bool = False,
                 never_chat_names: Sequence[str] = (),
                 confirm_send_file: bool | Callable[[], bool] = False,
                 capabilities: frozenset[str] | Callable[[], frozenset[str]] | None = None,
                 chat_persona: str = "",
                 display_name: str = "",
                 chat_shared: bool = False,
                 snapshot: bool | None = None,
                 channel: Literal["voice", "chat"] = "voice",
                 channel_note: str = "",
                 send_file: bool = False,
                 test_tools: bool = False,
                 desktop_locks: Sequence[DesktopLock] | None = None,
                 desktop_wait_s: float = VOICE_DESKTOP_WAIT_S,
                 desktop_busy_reason: str = REASON_DESKTOP_BUSY_VOICE,
                 confirm_timeout_s: float = 45.0,
                 switch_workspace: bool = False,
                 on_rename: mac_tools.SettingCallback | None = None,
                 on_reset_voice: mac_tools.SettingCallback | None = None,
                 on_go_to_sleep: mac_tools.SettingCallback | None = None,
                 on_alive: Callable[[bool], None] | None = None,
                 on_set_persona: mac_tools.SettingCallback | None = None) -> None:
        self.cfg = cfg
        self.gate = gate
        self.chrome_status_cb = chrome_status_cb
        self.cwd = Path(cwd) if cwd else config.WORKSPACE_DIR
        if memory_file is not None:
            memory_path = memory_file.path
        self.memory_path = memory_path or (self.cwd / "memory.md" if cwd else config.MEMORY_PATH)
        self.results_dir = Path(getattr(cfg, "results_dir", "~/Documents/Mac")).expanduser()
        self._client_factory = client_factory or ClaudeSDKClient
        self._recent_user_text = recent_user_text  # B2 hook: last ~10 user utterances, newest last
        self._memory = memory_file if memory_file is not None else memory.MemoryFile(self.memory_path)
        self._store: SessionStore = session_store or StateJsonSessionStore()
        self._steer_only = steer_only
        self._rotate_arg = rotate_after_idle_min
        self._handoff_enabled = handoff_enabled
        self._announce_fallback = on_resume_fallback == "status"
        self._max_turn_arg = max_turn_s
        self._write_claude_md = write_claude_md
        self._sources_arg = setting_sources
        self._channel = channel
        self._on_alive = on_alive  # True on every connect, False on every disconnect (idle, crash, close)
        self._channel_note = channel_note
        self._chat_persona = chat_persona  # W9.4: rendered (escaped) by prompts after the safety rules
        self._display_name = display_name
        self._chat_shared = chat_shared
        # W10.7: False rebuilds the system prompt on every request, so a persona edit reaches a
        # resumed session (chat brains). None keeps the SDK default (the recorded prompt).
        self._snapshot = snapshot
        if channel == "chat" and (on_rename or on_reset_voice or on_go_to_sleep):
            log.warning("brain: voice-only tools are never given to a chat brain; ignoring their callbacks")
            on_rename = on_reset_voice = on_go_to_sleep = None
        if channel != "chat":
            on_set_persona = None  # set_persona exists only in chat profiles (W9.4)
        self._never_chat_names = tuple(never_chat_names)
        self._ingest_all_tabs = whatsapp_ui_block  # keep tab hosts fresh for the M1 block
        self._fork_next = False  # the next connect resumes `_resume_next` as a fork
        self._fork_live = False  # connected to a fork whose new session id is not known yet
        self._fallback_from: str | None = None  # F9: the session a failed resume dropped
        self._connected_sources: list[str] | None = None
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
        locks = default_desktop_locks() if desktop_locks is None else list(desktop_locks)
        self.desktop: DesktopGuard | None = DesktopGuard(
            locks, wait_s=desktop_wait_s, busy_reason=desktop_busy_reason,
            on_wait=self._on_desktop_wait) if locks else None
        self.hook = PreToolHook(
            gate, self._current_turn, self.is_cancelled, self._memory_check,
            restrict=self._restrict, on_decision=self._on_hook_decision, desktop=self.desktop)
        self.tools = MacTools(gate=gate, current_turn=self._current_turn,
                              memory_check=self._memory_check, on_say=self._on_say,
                              on_confirm=self._on_confirm, pending_action=self.hook.pending_for,
                              on_notify=on_notify, workspace=self.cwd, memory_path=self.memory_path,
                              results_dir=self.results_dir, confirm_timeout_s=confirm_timeout_s,
                              memory=self._memory, on_attach=self._on_attach if send_file else None,
                              test_tools=test_tools, on_rename=on_rename, on_reset_voice=on_reset_voice,
                              on_go_to_sleep=on_go_to_sleep, on_set_persona=on_set_persona,
                              on_switch_workspace=self._on_switch_workspace if switch_workspace else None)
        self.hook.context = GateContext(
            workspace=str(self.cwd), element_label=self.tools.registry.label,
            chrome_ref_label=self.tools.chrome_refs.label, chrome_tab_host=self.tools.chrome_tabs.host,
            current_app=self._current_app, whatsapp_ui_block=whatsapp_ui_block,
            never_chat_names=tuple(never_chat_names),
            env_exempt_dir=str(env_exempt_dir) if env_exempt_dir is not None else None,
            memory_path=str(self.memory_path), confirm_send_file=confirm_send_file,
            capabilities=capabilities, display_name=display_name)

    # ------------------------------------------------------------------ Brain protocol
    async def start(self) -> None:
        if self._task is not None:
            return
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        claude_md = self.cwd / "CLAUDE.md"
        if self._write_claude_md and not claude_md.exists():
            claude_md.write_text(prompts.WORKSPACE_CLAUDE_MD)
        self._resume_next = await self._load_session()
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
        self._release_desktop()
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
                       data=self._error_data(last, {"circuit_open": True, "retry_in_s": wait}))
            return
        self._put(_Cmd("ask", turn_id, text))

    async def steer(self, turn_id: int, text: str) -> bool | None:
        """Voice: fire and forget; a steer for a finished turn starts a new query (today's
        behaviour). ``steer_only`` (chat, F12): returns whether the text joined the RUNNING turn
        (``turn_id`` 0 = whichever turn runs); False means nothing was sent and the caller makes it
        a new ask under fresh leases."""
        if not self._steer_only:
            self._put(_Cmd("steer", turn_id, text))
            return None
        if self._task is None:
            return False
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._put(_Cmd("steer", turn_id, text, payload=fut))
        try:
            return await asyncio.wait_for(asyncio.shield(fut), STEER_REPLY_TIMEOUT_S)
        except TimeoutError:
            fut.cancel()  # the owner sees a done future and drops the text
            return False

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
        if self._task is None or not self._handoff_enabled:
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
                self._report_alive(False)
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
            self._release_desktop()
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

    # ------------------------------------------------------------------ profile values
    def _rotate_min(self) -> float | None:
        """Idle minutes after which the session is rotated; None: never (chat, F9)."""
        if self._rotate_arg == _FROM_CONFIG:
            return float(self.cfg.brain.session_idle_rotate_min)
        return None if self._rotate_arg is None else float(self._rotate_arg)

    def _max_turn_s(self) -> float:
        return float(self._max_turn_arg if self._max_turn_arg is not None else self.cfg.brain.max_turn_s)

    # ------------------------------------------------------------------ desktop lease (F3)
    def _release_desktop(self) -> None:
        if self.desktop is not None and self.desktop.held:
            self.desktop.release()

    def _on_desktop_wait(self, turn: int) -> None:
        if turn and not self.is_cancelled(turn):
            self._emit(BrainEventKind.STATUS, turn, data={"waiting_for": "desktop"})

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
        """Capability check for memory writes: (allowed, "grounded"|"handoff"|reason). Deterministic
        anti-injection guard: a fact must come from what the user said. Whether they ASKED to keep it
        is the model's call (remember's `explicit` argument, AG14)."""
        a = self._active
        if a is None or a.draining:
            return False, "no active turn"
        if tool == "rewrite_memory":
            return (True, "handoff") if a.handoff else (False, "only during a session handoff")
        if a.handoff and str(args.get("kind", "")).lower() == "context":
            return True, "handoff"
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
        if a is not None and a.switch is not None:
            return "A move to another folder is pending: stop now and call no more tools."
        return None

    def _on_say(self, text: str) -> None:
        turn = self._current_turn()
        if turn and not self.is_cancelled(turn):
            self._emit(BrainEventKind.SAY, turn, text=text)

    def _on_confirm(self, summary: str, target: str, question: str = "", detail: str = "") -> None:
        self._release_desktop()  # W3.4: never hold the screen while a human decides
        turn = self._current_turn()
        if turn and not self.is_cancelled(turn):
            # AG10: text = the model's summary; data.question = its checked question ("" = none given)
            data = {"target": target, "question": question}
            if detail:
                data["detail"] = detail  # set_persona: the full new text the yes applies to (W9.4)
            self._emit(BrainEventKind.CONFIRM, turn, text=summary, data=data)

    def clear_switch(self, turn_id: int | None = None) -> bool:
        """agentd refused the model's switch_workspace: tools are allowed again in this turn.
        Call it before steering the refusal in. `turn_id` (None = the running turn) must match.
        True when a pending switch was cleared."""
        a = self._active
        if a is None or a.switch is None or (turn_id is not None and a.turn_id != turn_id):
            return False
        a.switch = None
        return True

    def _on_switch_workspace(self, folder: str, mode: str) -> str | None:
        """AG1: the running model says this request belongs elsewhere. agentd validates the folder
        and session and owns the move (it stops this turn and re-runs it); the brain only reports
        it and refuses every further tool call of the turn."""
        a = self._active
        if a is None or a.turn_id <= 0 or self.is_cancelled(a.turn_id):
            return "The request was stopped."
        if a.switch is not None:
            return "A switch was already requested; stop now."
        a.switch = (folder, mode)
        work = [n for n in a.tools.values() if n not in _SILENT_TOOLS and n != _SWITCH_TOOL]
        self._emit(BrainEventKind.STATUS, a.turn_id, data={"switch_workspace": {
            "folder": folder, "mode": mode, "tools_run": len(work)}})
        return None

    def _on_attach(self, attach_id: str, path: str) -> str | None:
        """mac.send_file → ATTACH. The real path is re-checked (a symlink swapped in after the
        hook's check must not leak a secret)."""
        turn = self._current_turn()
        if turn <= 0 or self.is_cancelled(turn):
            return "The request was stopped; nothing was sent."
        d = analyze(MAC_SEND_FILE, {"path": path}, self.hook.context)
        if d.always_denied:
            return f"Not sent: {d.reason}."
        self._emit(BrainEventKind.ATTACH, turn, data={"attach_id": attach_id, "path": path})
        return None

    def _on_hook_decision(self, d: HookDecision) -> None:
        if not d.allowed and not self.is_cancelled(d.turn_id):
            self._emit(BrainEventKind.STATUS, d.turn_id,
                       data={"gate": {"tool": d.tool_name, "verdict": d.verdict, "code": d.code,
                                      "allowed": False}})

    def _error_data(self, err: errors.ClassifiedError | None, data: dict[str, Any]) -> dict[str, Any]:
        """ERROR data, plus ``reset_at`` (wall-clock epoch seconds) for a rate limit whose reset
        time is known: from the CLI's rate_limit_event, else an epoch in the error text."""
        if err is None or err.kind is not BrainErrorKind.RATE_LIMIT:
            return data
        reset_at: float | None = float(self._rate_reset_at) if self._rate_reset_at else None
        if reset_at is None:
            m = errors._EPOCH.search(f"{err.detail} {err.message}")
            reset_at = float(m.group(1)) if m else None
        return {**data, "reset_at": reset_at} if reset_at else data

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
    async def _load_session(self) -> str | None:
        try:
            info = await self._store.load()
        except Exception:
            log.exception("brain: could not load the session store")
            return None
        if info is None or not info.session_id:
            return None
        ts, rotate = info.updated_at, self._rotate_min()
        if rotate is not None and self._wall() - ts >= rotate * 60:
            return None
        self._session_id = str(info.session_id)
        self._last_activity = ts or self._wall()
        self._session_started = info.started_at or ts or self._wall()
        self._fork_next = info.fork
        log.info("brain: resuming session %s%s", info.session_id, " as a fork" if info.fork else "")
        return str(info.session_id)

    async def _save_session(self, session_id: str | None) -> None:
        try:
            await self._store.save(SessionInfo(session_id, self._wall(), self._session_started,
                                               fork=bool(session_id) and self._fork_live))
        except Exception:
            log.exception("brain: could not persist session state")

    # ------------------------------------------------------------------ options / connection
    def build_options(self, resume: str | None, api_key: str | None = None) -> ClaudeAgentOptions:
        """Pure except for reading memory.md; callers run it in a thread."""
        cfg = self.cfg.brain
        env = {"ANTHROPIC_API_KEY": api_key} if api_key else {}
        persona = prompts.persona_prompt(self.cfg, browser_available=self.chrome_connected is not False,
                                         memory_path=self.memory_path, results_dir=self.results_dir,
                                         channel=self._channel, channel_note=self._channel_note,
                                         never_chat_names=self._never_chat_names,
                                         tools={t.name for t in self.tools.tools()},
                                         chat_persona=self._chat_persona, display_name=self._display_name,
                                         shared=self._chat_shared)
        system_prompt: dict[str, Any] = {"type": "preset", "preset": "claude_code", "append": persona}
        if self._snapshot is not None:
            system_prompt["snapshot"] = self._snapshot
        opts = ClaudeAgentOptions(
            system_prompt=system_prompt,  # type: ignore[arg-type]
            permission_mode="bypassPermissions",
            hooks=self.hook_matchers(),
            mcp_servers={"mac": self.tools.server()},
            cwd=str(self.cwd),
            setting_sources=list(self._connected_sources) if self._connected_sources is not None
            else self._setting_sources(),
            include_partial_messages=True,
            extra_args={"chrome": None} if cfg.chrome else {},
            env=env,
            stderr=self._stderr.append,
            resume=resume,
            model=cfg.model or None,
        )
        if resume and self._fork_next:
            opts.fork_session = True  # never write into a session Mac did not create
        if cfg.auth == "claude_cli":
            opts.cli_path = default_cli_path(self.cfg)
        return opts

    def hook_matchers(self) -> dict[Any, list[HookMatcher]]:
        """F1: the gate fails CLOSED. A hook that raises or outlives HOOK_TIMEOUT_S denies the
        call (on_failure="block"; SPIKES P0.3), and every wait inside it is capped well below that.
        PostToolUse stamps the last desktop use (idle release)."""
        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[self.hook], timeout=HOOK_TIMEOUT_S,
                                       on_failure="block")],
            "PostToolUse": [HookMatcher(matcher=_POST_TOOL_MATCHER, hooks=[self.hook], timeout=10)],
        }

    async def _wanted_sources(self) -> list[str]:
        """Setting sources for the next connect: the profile's (fixed or re-checked at every turn
        start, M2) or cfg.brain.setting_sources."""
        arg = self._sources_arg
        if arg is None:
            return self._setting_sources()
        value: Any = arg() if callable(arg) else arg
        if inspect.isawaitable(value):
            value = await value
        return self._valid_sources(list(value))

    def _setting_sources(self) -> list[Any]:
        return self._valid_sources(list(self.cfg.brain.setting_sources))

    @staticmethod
    def _valid_sources(wanted: list[Any]) -> list[Any]:
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
        self._connected_sources = await self._wanted_sources()
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
            self._fork_live = bool(attempt) and self._fork_next
            self._fork_next = False
            if attempt is None:
                self._session_started = self._wall()
                if resume:  # F9: the stored session could not be reopened; a fresh one started
                    log.warning("brain: could not resume session %s; started a fresh one", resume)
                    self._fallback_from = resume
            self._handed_off = False
            self._turns_this_session = 0
            self._idle_at = self._clock()
            log.info("brain connected (gen %d, resume=%s)", self._gen, attempt)
            self._report_alive(True)
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
        if client is not None:
            self._report_alive(False)
        await self._close_client(client, reader)

    def _report_alive(self, alive: bool) -> None:
        if self._on_alive is None:
            return
        try:
            self._on_alive(alive)
        except Exception:
            log.exception("on_alive callback failed")

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
        """Lazy connect (ask/warm). Rotates the session if idle for too long (voice profile), and
        reconnects (same session) when the profile's setting sources changed (M2 re-check)."""
        idle = self._wall() - self._last_activity
        rotate = self._rotate_min()
        if rotate is not None and idle > rotate * 60 and (self._client or self._resume_next):
            log.info("brain: rotating session after %.0f idle minutes", idle / 60)
            await self._disconnect()
            self._resume_next = None
            self._fork_next = False
        if self._client is not None and self._sources_arg is not None and callable(self._sources_arg):
            wanted = await self._wanted_sources()
            if wanted != self._connected_sources:
                log.info("brain: setting sources changed to %s; reconnecting, same session", wanted)
                await self._park_session()
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
        self._recent_backup = await asyncio.to_thread(self._memory.recent_context)
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
        backup, self._recent_backup = getattr(self, "_recent_backup", []), []
        if backup and not await asyncio.to_thread(self._memory.recent_context):
            await asyncio.to_thread(self._memory.set_recent_context, backup)  # keep the old notes
        if a.future is not None and not a.future.done():
            a.future.set_result(ok)
        if a.after is not None:
            await a.after()

    def _session_age_s(self) -> float:
        return self._wall() - self._session_started

    async def _park_session(self) -> None:
        """Disconnect but keep the session id for the next connect."""
        sid = self._session_id
        fork = self._fork_live  # still on a fork whose own id never arrived: resume it as a fork
        await self._disconnect()
        self._resume_next = sid
        self._fork_next = fork and bool(sid)
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
                       error_kind=err.kind, data=self._error_data(err, err.as_data()))
            return
        now = self._clock()
        self._active = _Turn(cmd.turn_id, now + self._max_turn_s(), texts=[cmd.text])
        if self._fallback_from is not None:
            if self._announce_fallback:  # F9: the client says so instead of "Continuing ..."
                self._emit(BrainEventKind.STATUS, cmd.turn_id,
                           data={"session_fallback": {"session_id": self._fallback_from}})
            self._fallback_from = None
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
        if cmd.payload is not None:
            await self._on_steer_only(cmd)
            return
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

    async def _on_steer_only(self, cmd: _Cmd) -> None:
        """F12: join the RUNNING turn or report False; never start a query on its own."""
        fut: asyncio.Future[bool] = cmd.payload
        if fut.done():
            return  # the caller gave up waiting: drop the text, it will come back as an ask
        a = self._active
        target = cmd.turn_id or (a.turn_id if a is not None else 0)
        if a is None or target <= 0 or a.turn_id != target or a.draining or self._client is None:
            fut.set_result(False)
            return
        a.texts.append(cmd.text)
        a.queries += 1
        try:
            await self._client.query(cmd.text)
        except Exception as e:  # noqa: BLE001
            fut.set_result(False)
            await self._fail_active(errors.classify_exception(e, self.stderr_tail()), reconnect=True)
            return
        fut.set_result(True)

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
        if (a is None and self._client is not None and self._handoff_enabled and not self._handed_off
                and self._turns_this_session):
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
                rotate = self._rotate_min()
                if (self._handoff_enabled and not self._handed_off and rotate is not None
                        and self._session_age_s() > rotate * 60):
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
                           error_kind=err.kind, data=self._error_data(err, err.as_data()))
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
        if self._active is None:
            self._release_desktop()  # turn teardown (finish, drain, timeout, failure, reset)
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
        if msg.subtype == "compact_boundary":
            self._on_compacted(msg.data or {})
            return
        if msg.subtype != "init":
            return
        data = msg.data or {}
        servers = data.get("mcp_servers") or []
        status = next((s.get("status") for s in servers
                       if isinstance(s, dict) and s.get("name") == _CHROME_SERVER), None)
        if data.get("session_id"):
            self._note_session_id(str(data["session_id"]))
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

    def _on_compacted(self, data: dict[str, Any]) -> None:
        """W10.7: Claude compacted the session. STATUS {compacted: {trigger, pre_tokens, post_tokens?}}
        for agentd (logged and shown in the app, never a chat message)."""
        meta = data.get("compact_metadata") or {}
        info: dict[str, Any] = {"trigger": str(meta.get("trigger") or "auto"), "pre_tokens": meta.get("pre_tokens")}
        if meta.get("post_tokens") is not None:
            info["post_tokens"] = meta["post_tokens"]
        log.info("brain: session compacted %s", info)
        turn = self._current_turn()
        self._emit(BrainEventKind.STATUS, max(turn, 0), data={"compacted": info})

    def _on_stream_event(self, a: _Turn, msg: StreamEvent) -> None:
        if msg.parent_tool_use_id:
            return
        ev = msg.event or {}
        etype = ev.get("type")
        if etype == "message_start":
            a.saw_delta = False
        elif etype == "content_block_start":
            btype = (ev.get("content_block") or {}).get("type")
            if btype == "text" and a.emitted_text:
                self._emit(BrainEventKind.TEXT, a.turn_id, text="\n", data={"block_start": True})
            elif btype == "tool_use":  # the text before this was a step line, not the answer
                self._emit(BrainEventKind.TEXT, a.turn_id, text="", data={"tool_follows": True})
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
            elif name in (f"{_CHROME}tabs_context_mcp", f"{_CHROME}tabs_create_mcp", f"{_CHROME}navigate") \
                    or (self._ingest_all_tabs and name.startswith(_CHROME)):
                self.tools.chrome_tabs.ingest(text)
            if self.desktop is not None and is_desktop_tool(name):
                self.desktop.stamp(block.tool_use_id)  # also covers failed calls (no PostToolUse)
            if name in _SILENT_TOOLS:
                continue
            self._emit(BrainEventKind.TOOL_END, a.turn_id, tool_name=name, tool_use_id=block.tool_use_id,
                       is_error=bool(block.is_error), text=text[:500])

    def _note_session_id(self, sid: str) -> None:
        if self._fork_live and sid != self._session_id:
            self._fork_live = False  # the fork has its own id now: that one is ours to continue
        self._session_id = sid

    async def _on_result(self, a: _Turn, msg: ResultMessage) -> None:
        if msg.session_id:
            self._note_session_id(msg.session_id)
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
                           error_kind=err.kind, data=self._error_data(err, {**data, **err.as_data()}))
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
