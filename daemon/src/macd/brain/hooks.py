"""PreToolUse hook: the capability gate the model cannot bypass (ARCHITECTURE §6).

An outward action is allowed only with a confirmation token bound to that action's
fingerprint (tool, gate code, normalised target). When the hook denies an outward call it
records the fingerprint as the *pending action*; ``mcp__mac__request_confirmation`` binds the
token to it, so a "yes" for "send Priya a WhatsApp" cannot be spent on `curl -F` or `git push`.

Fail closed (F1): the hook is registered with ``on_failure="block"`` and a timeout of
``HOOK_TIMEOUT_S``, so a hook that raises or times out denies. The verdict is computed first;
only an ``allow`` for a desktop tool may then wait for the desktop lease, and that wait is
capped at ``MAX_DESKTOP_WAIT_S``, well under the hook timeout.

Desktop lease (WhatsApp ARCHITECTURE W3.4): `DesktopGuard` serialises the physical screen and
Chrome between brains. It polls a list of non-blocking `DesktopLock`s in order (agentd's
in-process lock, then the transitional ``~/.mac-agent/run/desktop.lock`` flock while voice can
run in macd), is owned by one turn at a time, is released 20 s after the last desktop call
(PostToolUse stamp), while a confirmation is open, and at turn teardown (ClaudeBrain).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import inspect
import logging
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from macd.brain.gate import (
    CAPABILITY,
    MEMORY_WRITE,
    ConfirmationGate,
    GateContext,
    GateDecision,
    analyze,
)

log = logging.getLogger(__name__)

CONFIRM_TOOL = "mcp__mac__request_confirmation"
MEMORY_TOOLS = {"mcp__mac__remember", "mcp__mac__forget", "mcp__mac__rewrite_memory"}
MemoryCheck = Callable[[str, dict[str, Any]], tuple[bool, str]]  # (tool, args) -> (allowed, how|reason)

REASON_CANCELLED = "The user cancelled this request. Stop now and do not call any more tools."
REASON_MEMORY_WRITE = ("Direct edits to CLAUDE.md or memory.md are not allowed. Use mcp__mac__remember "
                       "with a one-line fact, and only when the user asked you to remember something.")
HOOK_TIMEOUT_S = 90.0  # HookMatcher timeout; with on_failure="block" a timeout denies (F1)
MAX_DESKTOP_WAIT_S = 60.0  # any single hook wait stays at least 30 s under HOOK_TIMEOUT_S
VOICE_DESKTOP_WAIT_S = 15.0
CHAT_DESKTOP_WAIT_S = 45.0
DESKTOP_IDLE_RELEASE_S = 20.0
DESKTOP_POLL_S = 0.1
REASON_DESKTOP_BUSY_VOICE = (
    "The browser and screen are busy with a WhatsApp task. Tell the user they can say 'stop the "
    "other task' to free it, or try again in a minute.")
REASON_DESKTOP_BUSY_CHAT = ("The browser and screen are busy with your voice request. Call the same "
                            "tool again to keep waiting.")
DESKTOP_PREFIXES = ("mcp__claude-in-chrome__", "mcp__mac__ui_")
DESKTOP_TOOLS = {"mcp__mac__screenshot", "mcp__mac__open_app"}
REASON_OUTWARD = ("This action needs the user's spoken confirmation. First call "
                  f"{CONFIRM_TOOL} with a one-sentence summary (who or what it goes to, and the gist). "
                  "If it says the user approved, retry exactly the same action. If declined, don't.")


def _deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def is_desktop_tool(name: str) -> bool:
    """Tools that drive the shared screen or Chrome (matched on the tool name only)."""
    return name.startswith(DESKTOP_PREFIXES) or name in DESKTOP_TOOLS


# ----------------------------------------------------------------------------- desktop lease


class DesktopLock(Protocol):
    """A non-blocking lock. The guard polls it, unless it also offers the event-driven
    ``async acquire(timeout, cancelled) -> bool`` (agentd's DesktopHandle), which is awaited."""

    def try_acquire(self) -> bool: ...
    def release(self) -> None: ...


class FlockLock:
    """Transitional cross-process lock: a non-blocking flock on ``path`` (local mode only).

    Each instance has its own open file description, so two instances conflict even inside one
    process. The kernel drops the lock when the holder dies (kill -9 included)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def try_acquire(self) -> bool:
        if self._fd is not None:
            return True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            # Coordination, not safety: an unusable lock file must not take the desktop away
            # from voice. The gate verdict was already decided.
            log.warning("desktop: cannot open %s; continuing without the cross-process lock", self.path)
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError:
            log.warning("desktop: flock on %s failed; continuing without it", self.path)
            os.close(fd)
            return True
        self._fd = fd
        return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)


def default_desktop_locks() -> list[DesktopLock]:
    from macd import config  # resolved at call time (tests redirect RUN_DIR)

    return [FlockLock(config.RUN_DIR / "desktop.lock")]


class DesktopGuard:
    """The desktop lease of ONE brain. Owned by one turn; acquired lazily by the hook for an
    allowed desktop tool call; released at idle, while confirming and at turn teardown."""

    def __init__(self, locks: Sequence[DesktopLock], *, wait_s: float = VOICE_DESKTOP_WAIT_S,
                 busy_reason: str = REASON_DESKTOP_BUSY_VOICE,
                 idle_release_s: float = DESKTOP_IDLE_RELEASE_S, poll_s: float = DESKTOP_POLL_S,
                 on_wait: Callable[[int], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.locks = list(locks)
        self.wait_s = min(max(0.0, wait_s), MAX_DESKTOP_WAIT_S)
        self.busy_reason = busy_reason
        self.idle_release_s = idle_release_s
        self.poll_s = poll_s
        self.on_wait = on_wait  # (turn) -> e.g. emit status{waiting_for: desktop}
        self._clock = clock
        self._held: list[DesktopLock] = []
        self.owner: int | None = None  # turn id holding the lease
        self._in_flight: set[str] = set()
        self._idle_timer: asyncio.TimerHandle | None = None

    @property
    def held(self) -> bool:
        return self.owner is not None

    async def acquire(self, turn: int, tool_use_id: str, cancelled: Callable[[], bool]) -> str:
        """"ok", "busy" (bounded wait ran out) or "cancelled" (the turn stopped meanwhile).

        Locks are taken in order. A lock with an async ``acquire(timeout, cancelled)`` (agentd's
        handle) is awaited and wakes on the release itself; others (the transitional flock) are
        polled every ``poll_s``."""
        if cancelled():
            return "cancelled"
        if self.owner is not None and self.owner != turn:
            self.release()  # a previous turn of this brain left it: turns are sequential
        deadline = self._clock() + self.wait_s
        waited = False
        for lock in self.locks:
            if lock in self._held:
                continue
            while not lock.try_acquire():
                if cancelled():
                    self._drop_partial()
                    return "cancelled"
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self._drop_partial()
                    return "busy"
                if not waited and self.on_wait is not None:
                    waited = True
                    with contextlib.suppress(Exception):
                        self.on_wait(turn)
                event_acquire = getattr(lock, "acquire", None)
                if event_acquire is not None and inspect.iscoroutinefunction(event_acquire):
                    if await event_acquire(remaining, cancelled):
                        break  # granted by the release itself
                    if not cancelled() and self._clock() < deadline:
                        await asyncio.sleep(self.poll_s)  # spurious refusal: never spin
                else:
                    await asyncio.sleep(self.poll_s)
            self._held.append(lock)
        if cancelled():  # re-check after the wait: a stopped turn's waiter never keeps it (F3)
            if self.owner is None:
                self.release()
            return "cancelled"
        self.owner = turn
        self._cancel_idle()
        if tool_use_id:
            self._in_flight.add(tool_use_id)
        else:
            self._schedule_idle()
        return "ok"

    def _drop_partial(self) -> None:
        if self.owner is None:
            self.release()

    def stamp(self, tool_use_id: str) -> None:
        """PostToolUse of a desktop tool: the idle-release clock restarts."""
        self._in_flight.discard(tool_use_id)
        if self.owner is not None and not self._in_flight:
            self._schedule_idle()

    def _schedule_idle(self) -> None:
        self._cancel_idle()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_timer = loop.call_later(self.idle_release_s, self._idle_release)

    def _cancel_idle(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _idle_release(self) -> None:
        self._idle_timer = None
        if not self._in_flight:
            log.info("desktop: released after %.0f s idle", self.idle_release_s)
            self.release()

    def release(self) -> None:
        """Free every lock (turn teardown, confirmation open, idle). Idempotent."""
        self._cancel_idle()
        self._in_flight.clear()
        self.owner = None
        for lock in reversed(self._held):
            try:
                lock.release()
            except Exception:
                log.exception("desktop: lock release failed")
        self._held = []


@dataclass(frozen=True)
class ActionFingerprint:
    """What a confirmation token is bound to."""

    turn_id: int
    tool: str
    code: str
    target: str
    detail: str = ""  # shown with the confirmation (set_persona's full text); not in the key

    @property
    def key(self) -> str:
        return f"{self.tool}|{self.code}|{self.target.strip().lower()}"


@dataclass(frozen=True)
class HookDecision:
    t: float
    turn_id: int
    tool_name: str
    tool_use_id: str
    allowed: bool
    verdict: str
    code: str
    reason: str


class PreToolHook:
    """Callable SDK hook (`HookCallback`). `decisions` keeps recent outcomes for tests/diagnostics."""

    def __init__(self, gate: ConfirmationGate, current_turn: Callable[[], int],
                 is_cancelled: Callable[[int], bool], memory_check: MemoryCheck, *,
                 context: GateContext | None = None,
                 restrict: Callable[[str], str | None] | None = None,
                 on_decision: Callable[[HookDecision], None] | None = None,
                 desktop: DesktopGuard | None = None) -> None:
        self.gate = gate
        self.current_turn = current_turn
        self.is_cancelled = is_cancelled
        self.memory_check = memory_check
        self.restrict = restrict  # e.g. "memory tools only" during a session handoff
        self.context = context
        self.on_decision = on_decision
        self.desktop = desktop
        self.decisions: deque[HookDecision] = deque(maxlen=256)
        self.pending_action: ActionFingerprint | None = None  # last denied outward action

    def pending_for(self, turn_id: int) -> ActionFingerprint | None:
        p = self.pending_action
        return p if p is not None and p.turn_id == turn_id else None

    def _record(self, turn: int, name: str, tool_use_id: str, allowed: bool,
                decision: GateDecision, reason: str) -> None:
        rec = HookDecision(time.time(), turn, name, tool_use_id, allowed, decision.verdict,
                           decision.code, reason)
        self.decisions.append(rec)
        level = logging.INFO if allowed and decision.verdict == "allow" else logging.WARNING
        log.log(level, "gate %s turn=%d tool=%s verdict=%s code=%s target=%r %s",
                "ALLOW" if allowed else "DENY", turn, name, decision.verdict, decision.code,
                decision.target, reason)
        if self.on_decision:
            try:
                self.on_decision(rec)
            except Exception:
                log.exception("hook on_decision callback failed")

    def decide(self, name: str, tool_input: dict[str, Any], tool_use_id: str = "") -> tuple[bool, str]:
        turn = self.current_turn()
        if self.is_cancelled(turn):
            d = GateDecision("allow", "cancelled", REASON_CANCELLED)
            self._record(turn, name, tool_use_id, False, d, REASON_CANCELLED)
            return False, REASON_CANCELLED
        restricted = self.restrict(name) if self.restrict else None
        if restricted:
            d = GateDecision("allow", "restricted", restricted)
            self._record(turn, name, tool_use_id, False, d, restricted)
            return False, restricted
        if name in MEMORY_TOOLS:
            allowed, how = self.memory_check(name.removeprefix("mcp__mac__"), tool_input)
            if not allowed:
                reason = f"Memory not changed: {how}. Do not retry."
                d = GateDecision("sensitive_read", "memory_not_grounded", reason)
                self._record(turn, name, tool_use_id, False, d, reason)
                return False, reason
        d = analyze(name, tool_input, self.context)
        if d.verdict == "catastrophic":
            reason = (f"Blocked: {d.reason}. This is never allowed. Don't try another way; "
                      "tell the user you can't do that.")
        elif d.verdict == "sensitive_read" and d.code == MEMORY_WRITE:
            reason = REASON_MEMORY_WRITE
        elif d.code.startswith(f"{CAPABILITY}:"):  # W9.5: plain words the model relays once
            reason = (f"Not allowed: {d.reason}. Tell the user that in one short line, once, and carry on "
                      "without it. Don't try another way.")
        elif d.verdict == "sensitive_read":
            reason = (f"Blocked: {d.reason}. Reading or copying secrets, keys, keychains or browser "
                      "profiles is never allowed. Don't try another way; tell the user.")
        elif d.verdict == "outward":
            fp = ActionFingerprint(turn, name, d.code, d.target, d.detail)
            if self.gate.consume(turn, fp.key):
                self._record(turn, name, tool_use_id, True, d, "confirmed by the user")
                return True, ""
            self.pending_action = fp
            reason = f"{REASON_OUTWARD} (Reason: {d.reason}.)"
        else:
            self._record(turn, name, tool_use_id, True, d, "")
            return True, ""
        self._record(turn, name, tool_use_id, False, d, reason)
        return False, reason

    async def __call__(self, input_data: dict[str, Any], tool_use_id: str | None,
                       context: Any = None) -> dict[str, Any]:
        event = input_data.get("hook_event_name", "PreToolUse")
        if event in ("PostToolUse", "PostToolUseFailure"):
            return self.post_tool_use(input_data, tool_use_id)
        if event != "PreToolUse":
            return {}
        name = str(input_data.get("tool_name", ""))
        tool_input = input_data.get("tool_input") or {}
        tuid = tool_use_id or str(input_data.get("tool_use_id", ""))
        try:
            turn = self.current_turn()
            allowed, reason = self.decide(name, tool_input, tuid)  # F1: the verdict comes first
            if allowed and self.desktop is not None and is_desktop_tool(name):
                allowed, reason = await self._wait_desktop(turn, name, tuid)
        except Exception:  # fail closed (and the matcher's on_failure="block" backs this up)
            log.exception("pre-tool hook failed for %s", name)
            return _deny("Safety check failed for this action; it was blocked.")
        return {} if allowed else _deny(reason)

    async def _wait_desktop(self, turn: int, name: str, tool_use_id: str) -> tuple[bool, str]:
        assert self.desktop is not None
        outcome = await self.desktop.acquire(turn, tool_use_id, lambda: self.is_cancelled(turn))
        if outcome == "ok":
            return True, ""
        reason = REASON_CANCELLED if outcome == "cancelled" else self.desktop.busy_reason
        self._record(turn, name, tool_use_id, False, GateDecision("allow", f"desktop_{outcome}", reason), reason)
        return False, reason

    def post_tool_use(self, input_data: dict[str, Any], tool_use_id: str | None) -> dict[str, Any]:
        """PostToolUse / PostToolUseFailure: stamp the last desktop use. Never raises."""
        try:
            name = str(input_data.get("tool_name", ""))
            if self.desktop is not None and is_desktop_tool(name):
                self.desktop.stamp(tool_use_id or str(input_data.get("tool_use_id", "")))
        except Exception:
            log.exception("post-tool hook failed")
        return {}
