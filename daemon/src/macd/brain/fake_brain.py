"""FakeBrain: a scripted Brain with the same turn/interrupt/drain semantics as ClaudeBrain.

Event contract shared with ClaudeBrain (orchestrator tests rely on it):
- every event carries the turn_id of the ask that produced it;
- each turn ends with exactly one terminal event: DONE (data may contain
  ``{"interrupted": True}``) or ERROR;
- after ``interrupt(turn_id)`` nothing but that one DONE(interrupted) is emitted for the
  turn; the remaining steps are drained silently, and the next ``ask`` starts only after
  the drain completes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from macd.contracts import BrainErrorKind, BrainEvent, BrainEventKind

log = logging.getLogger(__name__)


@dataclass
class Step:
    """One scripted action. `delay` is slept *before* the step runs."""

    kind: BrainEventKind
    text: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    delay: float = 0.0
    is_error: bool = False
    error_kind: BrainErrorKind | None = None
    data: dict[str, Any] = field(default_factory=dict)


def text(t: str, delay: float = 0.0) -> Step:
    return Step(BrainEventKind.TEXT, text=t, delay=delay)


def say(t: str, delay: float = 0.0) -> Step:
    return Step(BrainEventKind.SAY, text=t, delay=delay)


def tool(name: str, tool_input: dict[str, Any] | None = None, duration: float = 0.0,
         result: str = "ok") -> list[Step]:
    return [Step(BrainEventKind.TOOL_START, tool_name=name, tool_input=tool_input or {}),
            Step(BrainEventKind.TOOL_END, tool_name=name, text=result, delay=duration)]


def error(kind: BrainErrorKind, message: str, delay: float = 0.0) -> Step:
    return Step(BrainEventKind.ERROR, text=message, error_kind=kind, is_error=True, delay=delay)


Script = Callable[[str], list[Step] | None]  # prompt -> steps (None/[] = default reply)


@dataclass
class _Cmd:
    kind: str  # ask | steer | interrupt | reset | close
    turn_id: int = 0
    text: str = ""


class FakeBrain:
    """Scripted Brain. `script(text)` returns the steps for an ask (pass `d.get` for a dict);
    prompts it doesn't know answer with ``default_reply``. `steer_script` handles steers."""

    def __init__(self, script: Script | None = None, *, default_reply: str = "Okay.",
                 steer_script: Callable[[str], list[Step]] | None = None,
                 drain_s: float = 0.0) -> None:
        self._script = script
        self.default_reply = default_reply
        self._steer_script = steer_script
        self.drain_s = drain_s  # simulated time for the stale tail to drain
        self._inbox: asyncio.Queue[_Cmd] = asyncio.Queue()
        self._out: asyncio.Queue[BrainEvent | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._cancelled: set[int] = set()
        self._active: int | None = None
        self._finished: set[int] = set()
        self._extra: list[Step] = []
        self.asks: list[tuple[int, str]] = []
        self.steers: list[tuple[int, str]] = []
        self.interrupts: list[int] = []
        self.resets = 0
        self.warms = 0
        self.sleeps = 0
        self.handoffs = 0
        self.alive = False
        self.name = "Mac"
        self.lang = "en-us"
        self.closed = False

    # -- Brain protocol ----------------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._owner(), name="fake-brain-owner")

    async def close(self) -> None:
        if self._task is None:
            return
        await self._inbox.put(_Cmd("close"))
        try:
            await asyncio.wait_for(self._task, 5)
        except TimeoutError:
            self._task.cancel()
        self._task = None
        self.closed = True
        await self._out.put(None)

    async def events(self) -> AsyncIterator[BrainEvent]:  # type: ignore[override]
        while True:
            ev = await self._out.get()
            if ev is None:
                return
            yield ev

    async def ask(self, turn_id: int, text: str) -> None:
        self.asks.append((turn_id, text))
        await self._inbox.put(_Cmd("ask", turn_id, text))

    async def steer(self, turn_id: int, text: str) -> None:
        self.steers.append((turn_id, text))
        await self._inbox.put(_Cmd("steer", turn_id, text))

    async def interrupt(self, turn_id: int) -> None:
        self.interrupts.append(turn_id)
        if turn_id and turn_id not in self._finished:
            self._cancelled.add(turn_id)

    async def reset(self) -> None:
        self.resets += 1
        await self._inbox.put(_Cmd("reset"))

    async def warm(self) -> None:
        self.warms += 1
        self.alive = True

    async def sleep(self) -> None:
        self.sleeps += 1
        self.alive = False

    async def handoff(self) -> bool:
        self.handoffs += 1
        return True

    async def set_name(self, name: str) -> None:
        self.name = name

    async def set_language(self, lang: str) -> None:
        self.lang = lang

    def is_cancelled(self, turn_id: int) -> bool:
        return turn_id in self._cancelled

    # -- owner task --------------------------------------------------------------------
    def _steps_for(self, text: str) -> list[Step]:
        steps = self._script(text) if self._script else None
        return list(steps) if steps else [Step(BrainEventKind.TEXT, text=self.default_reply)]

    async def _owner(self) -> None:
        while True:
            cmd = await self._inbox.get()
            if cmd.kind == "close":
                return
            if cmd.kind == "ask":
                await self._run_turn(cmd.turn_id, self._steps_for(cmd.text))
            elif cmd.kind == "steer":
                # Steer after the turn finished behaves like a new query on that turn.
                self._finished.discard(cmd.turn_id)
                steps = self._steer_script(cmd.text) if self._steer_script else []
                await self._run_turn(cmd.turn_id, steps or self._steps_for(cmd.text))

    async def _drain_steer_inbox(self, turn_id: int) -> None:
        """Steers for the running turn are appended to it (not queued as new turns)."""
        keep: list[_Cmd] = []
        while not self._inbox.empty():
            c = self._inbox.get_nowait()
            if c.kind == "steer" and c.turn_id == turn_id:
                if self._steer_script:
                    self._extra.extend(self._steer_script(c.text))
            else:
                keep.append(c)
        for c in keep:
            self._inbox.put_nowait(c)

    async def _run_turn(self, turn_id: int, steps: list[Step]) -> None:
        self._active = turn_id
        self._extra = []
        try:
            queue = list(steps)
            while queue or self._extra:
                if self._extra:
                    queue.extend(self._extra)
                    self._extra = []
                step = queue.pop(0)
                if step.delay:
                    await self._sleep_or_interrupt(turn_id, step.delay)
                await self._drain_steer_inbox(turn_id)
                if turn_id in self._cancelled:
                    break
                ev = BrainEvent(kind=step.kind, turn_id=turn_id, text=step.text,
                                tool_name=step.tool_name, tool_input=dict(step.tool_input),
                                is_error=step.is_error, error_kind=step.error_kind,
                                data=dict(step.data))
                await self._out.put(ev)
                if step.kind in (BrainEventKind.ERROR, BrainEventKind.DONE):
                    return
            if turn_id in self._cancelled:
                if self.drain_s:
                    await asyncio.sleep(self.drain_s)
                await self._out.put(BrainEvent(BrainEventKind.DONE, turn_id, data={"interrupted": True}))
            else:
                await self._out.put(BrainEvent(BrainEventKind.DONE, turn_id, data={"interrupted": False}))
        finally:
            self._active = None
            self._finished.add(turn_id)
            self._cancelled.discard(turn_id)

    async def _sleep_or_interrupt(self, turn_id: int, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while loop.time() < end and turn_id not in self._cancelled:
            await asyncio.sleep(min(0.01, max(0.0, end - loop.time())))
