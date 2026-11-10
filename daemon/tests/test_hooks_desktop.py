"""F1 (fail-closed hook, verdict before any wait) and the desktop lease (F3, W3.4), including the
transitional desktop.lock flock across two real processes (kill -9 included)."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk.types import _hooks_to_internal_format

from macd import config
from macd.brain import hooks as hooks_mod
from macd.brain.claude_brain import CHAT_PROFILE, ClaudeBrain
from macd.brain.gate import ConfirmationGate, GateContext
from macd.brain.hooks import (
    HOOK_TIMEOUT_S,
    MAX_DESKTOP_WAIT_S,
    REASON_CANCELLED,
    REASON_DESKTOP_BUSY_CHAT,
    REASON_DESKTOP_BUSY_VOICE,
    DesktopGuard,
    FlockLock,
    PreToolHook,
    is_desktop_tool,
)

HOME = "/Users/alice"
WS = f"{HOME}/.mac-agent/workspace"
CLICK = {"action": "left_click", "coordinate": [1, 2], "tabId": 7}


def pre(tool: str, inp: dict[str, Any], tuid: str = "toolu_1") -> dict[str, Any]:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp, "tool_use_id": tuid}


def post(tool: str, tuid: str = "toolu_1") -> dict[str, Any]:
    return {"hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": {}, "tool_use_id": tuid}


def denied(out: dict[str, Any]) -> bool:
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def reason(out: dict[str, Any]) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


class BusyLock:
    """A DesktopLock someone else holds until `free` is set."""

    def __init__(self) -> None:
        self.busy = True
        self.tries = 0
        self.released = 0

    def try_acquire(self) -> bool:
        self.tries += 1
        return not self.busy

    def release(self) -> None:
        self.released += 1


def make_hook(lock: Any, *, wait_s: float = 0.3, turn: int = 5, cancelled: set[int] | None = None,
              busy_reason: str = REASON_DESKTOP_BUSY_VOICE, host: str = "") -> tuple[PreToolHook, DesktopGuard]:
    cancelled = cancelled if cancelled is not None else set()
    guard = DesktopGuard([lock], wait_s=wait_s, busy_reason=busy_reason, poll_s=0.01)
    hook = PreToolHook(ConfirmationGate(), lambda: turn, lambda t: t in cancelled,
                       lambda tool, args: (False, "no"), desktop=guard,
                       context=GateContext(home=HOME, workspace=WS, user="alice",
                                           chrome_tab_host=lambda tab: host or None))
    return hook, guard


# ----------------------------------------------------------------------------- F1


def test_hook_matchers_fail_closed_with_timeout_above_every_wait(tmp_path: Path) -> None:
    brain = ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "ws")
    internal = _hooks_to_internal_format(brain.build_options(None).hooks)
    (pre_m,) = internal["PreToolUse"]
    assert pre_m["on_failure"] == "block"  # a raising hook denies (SPIKES P0.3)
    assert pre_m["timeout"] == HOOK_TIMEOUT_S == 90
    assert pre_m["matcher"] is None and pre_m["hooks"] == [brain.hook]
    # every bounded wait inside the hook stays at least 30 s under the hook timeout
    assert MAX_DESKTOP_WAIT_S <= HOOK_TIMEOUT_S - 30
    chat = ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "c", **CHAT_PROFILE)
    assert chat.desktop is not None and chat.desktop.wait_s == 45 < HOOK_TIMEOUT_S - 30
    assert brain.desktop is not None and brain.desktop.wait_s == 15
    huge = DesktopGuard([], wait_s=10_000)
    assert huge.wait_s == MAX_DESKTOP_WAIT_S
    assert internal["PostToolUse"][0]["hooks"] == [brain.hook]


async def test_hook_that_raises_denies() -> None:
    def boom() -> int:
        raise RuntimeError("bug")
    hook = PreToolHook(ConfirmationGate(), boom, lambda t: False, lambda tool, args: (True, ""))
    out = await hook(pre("Bash", {"command": "ls"}), "t")
    assert denied(out) and "blocked" in reason(out)


async def test_desktop_wait_that_raises_denies() -> None:
    class Broken:
        def try_acquire(self) -> bool:
            raise OSError("kaput")

        def release(self) -> None:
            pass
    hook, _ = make_hook(Broken())
    assert denied(await hook(pre("mcp__mac__screenshot", {}), "t"))


@pytest.mark.parametrize("tool,inp,verdict", [
    ("mcp__claude-in-chrome__computer", {"action": "left_click", "tabId": 7, "coordinate": [1, 2]}, "outward"),
    ("Bash", {"command": "rm -rf ~"}, "catastrophic"),
    ("mcp__claude-in-chrome__computer", {"action": "left_click", "tabId": 7, "action_summary": "Send"},
     "outward"),
])
async def test_verdict_comes_before_any_wait(tool: str, inp: dict[str, Any], verdict: str) -> None:
    lock = BusyLock()  # the desktop is busy for the whole test
    hook, _ = make_hook(lock, wait_s=5, host="mail.google.com")
    t0 = time.monotonic()
    out = await hook(pre(tool, inp), "t")
    assert denied(out)
    assert time.monotonic() - t0 < 0.5  # never waited for the lease
    assert lock.tries == 0
    assert hook.decisions[-1].verdict == verdict


# ----------------------------------------------------------------------------- lease


def test_desktop_tools_are_matched_on_names_only() -> None:
    for name in ("mcp__claude-in-chrome__navigate", "mcp__claude-in-chrome__get_page_text",
                 "mcp__mac__ui_click", "mcp__mac__ui_tree", "mcp__mac__screenshot", "mcp__mac__open_app"):
        assert is_desktop_tool(name)
    for name in ("Bash", "Read", "mcp__mac__say", "mcp__mac__notify", "mcp__mac__send_file", "WebFetch"):
        assert not is_desktop_tool(name)


async def test_allowed_desktop_call_waits_bounded_then_denies_busy() -> None:
    lock = BusyLock()
    hook, guard = make_hook(lock, wait_s=0.2, busy_reason=REASON_DESKTOP_BUSY_CHAT)
    t0 = time.monotonic()
    out = await hook(pre("mcp__claude-in-chrome__get_page_text", {"tabId": 7}), "t")
    assert 0.15 <= time.monotonic() - t0 < 1.0
    assert denied(out) and reason(out) == REASON_DESKTOP_BUSY_CHAT
    assert "voice request" in reason(out) and "—" not in reason(out)
    assert not guard.held and hook.decisions[-1].code == "desktop_busy"


async def test_non_desktop_tools_never_touch_the_lease() -> None:
    lock = BusyLock()
    hook, _ = make_hook(lock)
    assert await hook(pre("Bash", {"command": "ls"}), "t") == {}
    assert lock.tries == 0


async def test_waiter_gets_the_lease_when_freed() -> None:
    lock = BusyLock()
    waits: list[int] = []
    hook, guard = make_hook(lock, wait_s=2)
    guard.on_wait = waits.append
    asyncio.get_running_loop().call_later(0.1, setattr, lock, "busy", False)
    assert await hook(pre("mcp__mac__screenshot", {}), "t") == {}
    assert guard.held and guard.owner == 5 and waits == [5]


async def test_stopped_turns_waiter_never_acquires() -> None:
    lock = BusyLock()
    cancelled: set[int] = set()
    hook, guard = make_hook(lock, wait_s=5, cancelled=cancelled)
    task = asyncio.create_task(hook(pre("mcp__mac__screenshot", {}), "t"))
    await asyncio.sleep(0.05)
    cancelled.add(5)
    lock.busy = False  # freed at the same moment: the stopped waiter must still not take it
    out = await asyncio.wait_for(task, 1)
    assert denied(out) and reason(out) == REASON_CANCELLED
    assert not guard.held


async def test_idle_release_after_last_desktop_use() -> None:
    lock = BusyLock()
    lock.busy = False
    hook, guard = make_hook(lock)
    guard.idle_release_s = 0.1
    assert await hook(pre("mcp__mac__ui_click", {"x": 1, "y": 2}, "u1"), "u1") == {}
    await asyncio.sleep(0.2)
    assert guard.held  # in flight: never released under a running tool call
    assert await hook(post("mcp__mac__ui_click", "u1"), "u1") == {}  # PostToolUse stamp
    await asyncio.sleep(0.05)
    assert guard.held
    await asyncio.sleep(0.15)
    assert not guard.held and lock.released == 1


async def test_flock_lease_is_shared_by_two_brains_in_one_process(tmp_path: Path) -> None:
    path = tmp_path / "desktop.lock"
    a, b = FlockLock(path), FlockLock(path)
    ga = DesktopGuard([a], wait_s=0.1, poll_s=0.01)
    gb = DesktopGuard([b], wait_s=0.1, poll_s=0.01)
    assert await ga.acquire(1, "x", lambda: False) == "ok"
    assert await gb.acquire(9, "y", lambda: False) == "busy"
    ga.release()
    assert await gb.acquire(9, "y", lambda: False) == "ok"
    gb.release()


def test_flock_open_failure_does_not_take_the_desktop_away(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    lock = FlockLock(blocker / "desktop.lock")  # parent is a file: open fails
    assert lock.try_acquire() is True and not lock.held


def test_default_lock_lives_in_the_run_dir() -> None:
    (lock,) = hooks_mod.default_desktop_locks()
    assert isinstance(lock, FlockLock) and lock.path == config.RUN_DIR / "desktop.lock"


# ----------------------------------------------------------------------------- two processes

_HOLDER = """
import sys, time
from macd.brain.hooks import FlockLock
lock = FlockLock(sys.argv[1])
assert lock.try_acquire()
print("held", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _spawn_holder(path: Path, hold_s: float) -> subprocess.Popen[str]:
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(path), str(hold_s)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
    return proc


async def test_flock_two_processes_waiter_gets_it_after_release(tmp_path: Path) -> None:
    path = tmp_path / "run" / "desktop.lock"
    proc = _spawn_holder(path, 0.4)
    try:
        guard = DesktopGuard([FlockLock(path)], wait_s=0.1, poll_s=0.02)
        assert await guard.acquire(1, "t", lambda: False) == "busy"  # bounded, never blocks
        guard.wait_s = 5
        t0 = time.monotonic()
        assert await guard.acquire(1, "t", lambda: False) == "ok"  # holder exits -> free
        assert time.monotonic() - t0 < 3
        guard.release()
    finally:
        proc.kill()
        proc.wait()


async def test_flock_two_processes_kill_9_frees_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "desktop.lock"
    proc = _spawn_holder(path, 60)
    try:
        guard = DesktopGuard([FlockLock(path)], wait_s=0.1, poll_s=0.02)
        assert await guard.acquire(1, "t", lambda: False) == "busy"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        guard.wait_s = 2
        assert await guard.acquire(1, "t", lambda: False) == "ok"
        guard.release()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
