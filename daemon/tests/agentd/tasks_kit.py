"""Helpers for the scheduled-task scenario tests: a settable clock, origins, the invariant check
of DESIGN 12.4 (asserted after every step) and small waits."""

from __future__ import annotations

import asyncio
import collections
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from agentd_kit import CHAT, Agent, make_agent, stop_agent

from agentd import schema as s
from agentd import tasks as tk
from agentd.service import conv_kind
from macd.brain import fake_brain as fb
from macd.contracts import BrainEventKind

T0 = datetime(2026, 11, 13, 10, 0).timestamp()  # Fri 13 Nov 2026 10:00 local
VOICE = tk.Origin(conv=s.VOICE_CONV, kind="voice")
APP = tk.Origin(conv="app", kind="app")
IN_CHAT = tk.Origin(conv=CHAT, kind="chat")


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def done(session: str = "s1", tokens: int = 1000, window: int = 200000) -> fb.Step:
    """A DONE step carrying what ClaudeBrain's DONE data carries (D5): the kit's GateBrain saves
    the session id like the real brain does."""
    return fb.Step(BrainEventKind.DONE, data={"session_id": session, "context_tokens": tokens,
                                              "context_window": window, "interrupted": False})


def report(text: str = "Found 3 jobs.\nDetails in ~/Documents/Mac/jobs.md", **kw: Any) -> list[fb.Step]:
    return [fb.text(text), done(**kw)]


async def no_osascript(title: str, body: str) -> None:
    """Tests never run the real osascript (a notification on the developer's Mac)."""


async def agent(root: Any, clock: Clock | None = None, **kw: Any) -> Agent:
    """`make_agent` for the task tests: a settable clock, no real osascript, and `_finalize`
    counted per run id so `check` can assert I7 (exactly once per run)."""
    a = await make_agent(root, clock=clock or Clock(), **kw)
    a.tasks.osascript = no_osascript
    a._finalized = collections.Counter()  # type: ignore[attr-defined]
    orig = a.tasks._finalize

    async def counted(task: tk.TaskRecord, run: tk.RunRecord, *args: Any, **kwargs: Any) -> None:
        if not run.ended:
            a._finalized[(task.id, run.run_id)] += 1  # type: ignore[attr-defined]
        await orig(task, run, *args, **kwargs)

    a.tasks._finalize = counted  # type: ignore[method-assign]
    return a


async def stop(a: Agent) -> None:
    """`stop_agent`, then I9: nothing the manager spawned is left pending."""
    await stop_agent(a)
    mgr = a.tasks
    assert mgr._loop_task is None
    assert all(t.done() for t in [*mgr._runs.values(), *mgr._helpers]), "orphan task after close"
    for c in a.svc.convs.values():
        assert all(t.done() for t in c.tasks), c.conv


async def create(a: Agent, origin: tk.Origin = VOICE, *, title: str = "Job search", description: str = "find 20 jobs",
                 schedule: dict[str, Any] | None = None, kind: str = "task") -> tk.TaskRecord:
    data = await a.tasks.call(origin, "task_create", {"title": title, "description": description, "kind": kind,
                                                       "schedule": schedule or {"at": ["08:00"]}})
    check(a)
    return a.tasks.tasks[data["task"]["id"]]


async def until(pred: Callable[[], Any], timeout: float = 5.0) -> None:
    async def wait() -> None:
        while not pred():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)


async def settle(a: Agent, ticks: int = 1) -> None:
    """Drive the scheduler: tick, then let spawned runs and saves make progress."""
    for _ in range(ticks):
        a.tasks.tick()
        check(a)
        for _ in range(5):
            await asyncio.sleep(0.005)
    check(a)


async def finished(a: Agent, task: tk.TaskRecord, timeout: float = 5.0) -> None:
    await until(lambda: task.current is None or task.state == "waiting_limit", timeout)
    await history(a, task.id)
    check(a)


async def history(a: Agent, task_id: str, timeout: float = 5.0) -> list[dict[str, Any]]:
    """The history rows once every pending append or amend has landed (no bare sleeps)."""
    mgr = a.tasks

    def quiet() -> bool:
        return not any(t.get_name().startswith(f"agentd:tasks:{task_id}:") and not t.done() for t in mgr._helpers) \
            and not (mgr._lock(task_id).locked())

    await until(quiet, timeout)
    return runs_file(a, task_id)


def runs_file(a: Agent, task_id: str) -> list[dict[str, Any]]:
    return tk._read_lines(a.tasks.task_dir(task_id) / "runs.jsonl")


def tasks_json(a: Agent) -> dict[str, Any]:
    return json.loads((a.paths.state_dir / "tasks.json").read_text())


def convs_json(a: Agent) -> dict[str, Any]:
    return json.loads((a.paths.state_dir / "conversations.json").read_text())["conversations"]


def check(a: Agent) -> None:
    """DESIGN 12.4 invariants I1 to I5 (and rev monotonic) on the live state."""
    mgr = a.tasks
    last_rev = getattr(a, "_last_rev", 0)  # on the agent itself: a recycled id() must not carry a rev over
    assert mgr.rev >= last_rev, (mgr.rev, last_rev)
    a._last_rev = mgr.rev  # type: ignore[attr-defined]
    slots = 0
    for t in mgr.tasks.values():
        cur = t.current
        if t.state == "queued":  # I1
            assert cur is not None and cur.status == "queued", (t.id, t.state, cur)
        elif t.state == "waiting_limit":
            assert cur is not None and cur.status == "limit", (t.id, t.state, cur)
        elif t.state == "running":
            assert cur is not None and (cur.status == "running" or (cur.ended and cur.status not in ("queued", "limit"))), \
                (t.id, cur)
        else:
            assert t.state in tk.IDLE_STATES and cur is None, (t.id, t.state, cur)
        if cur is not None and cur.status == "queued":
            assert t.state == "queued"
        if cur is not None and cur.status == "running":
            assert t.state == "running"
        if cur is not None and cur.status == "limit":
            assert t.state == "waiting_limit"
        ids = [r["run_id"] for r in runs_file(a, t.id)]  # I2
        assert len(ids) == len(set(ids)), ids
        assert t.runs >= max((r.get("n", 0) for r in runs_file(a, t.id)), default=0)
        if mgr._holds_slot(t):
            slots += 1
        assert mgr.by_conv.get(t.conv) == t.id  # I4
        assert t.bound == (t.conv == t.report_to and conv_kind(t.conv) == "chat")
    assert slots <= mgr.settings.max_concurrent, slots  # I3
    assert len(mgr.by_conv) == len(mgr.tasks), (mgr.by_conv, list(mgr.tasks))
    counts = getattr(a, "_finalized", {})
    assert all(n == 1 for n in counts.values()), counts  # I7: _finalize once per run id
    for c in a.svc.convs.values():  # I5
        live = [x for x in c.turns.values() if not x.finished and not x.stopping]
        assert len(live) <= 1, (c.conv, [x.uid for x in live])
        assert sum(1 for x in c.turns.values() if x.in_brain and not x.finished) <= 1, c.conv
        if c.kind == "task" and c.brain is not None:
            assert getattr(c.brain, "_active", None) is None or c.running() is not None, c.conv


def app_notices(client: Any, ev: str) -> list[Any]:
    return [n for n in client.notices if n.EV == ev]
