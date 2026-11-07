"""The two ways macd runs background work: a supervised long-lived subsystem, or a tracked task."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

log = logging.getLogger(__name__)


async def supervise(
    name: str,
    coro_factory: Callable[[], Awaitable[Any]],
    backoff: tuple[float, float] = (0.5, 30.0),
    *,
    on_error: Callable[[str, BaseException], None] | None = None,
    restart_on_exit: bool = False,
    healthy_after_s: float = 60.0,
) -> None:
    """Run `coro_factory()` under a crash barrier: exceptions are logged, reported through
    `on_error`, and the subsystem is restarted with exponential backoff (reset after it has
    run `healthy_after_s`). A normal return ends supervision unless `restart_on_exit`.
    Only cancellation propagates."""
    delay = backoff[0]
    while True:
        t0 = time.monotonic()
        try:
            await coro_factory()
            if not restart_on_exit:
                return
            log.warning("subsystem %s exited; restarting in %.1fs", name, delay)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("subsystem %s crashed; restarting in %.1fs", name, delay)
            if on_error is not None:
                with contextlib.suppress(Exception):
                    on_error(name, e)
        if time.monotonic() - t0 > healthy_after_s:
            delay = backoff[0]
        await asyncio.sleep(delay)
        delay = min(delay * 2, backoff[1])


class TaskSet:
    """A set of tracked tasks: failures are logged (never lost), `cancel()` stops them all."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=f"{self.label}:{name}")
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("task %s failed", task.get_name(), exc_info=task.exception())

    def cancel(self) -> None:
        me = asyncio.current_task()
        for t in list(self._tasks):
            if t is not me:
                t.cancel()

    def __len__(self) -> int:
        return len(self._tasks)
