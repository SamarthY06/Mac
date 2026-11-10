"""CPU-bound blocking work in a worker PROCESS (V7).

A thread is not enough for Python-heavy work such as the settings-hash walk or `list_sessions`:
the worker thread holds the GIL, and every syscall the event loop makes (each socket write) then
waits up to the 5 ms switch interval to get it back. Under chat load that showed as 35-57 ms
loop-lag spikes. One long-lived spawn-context process does that work instead; if it cannot be
used, the call falls back to a thread.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")

_pool: concurrent.futures.ProcessPoolExecutor | None = None
MAX_BROKEN = 3  # worker starts that may fail before every call stays in a thread (circuit breaker)
_broken = 0


def _get_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = concurrent.futures.ProcessPoolExecutor(max_workers=1,
                                                       mp_context=multiprocessing.get_context("spawn"))
    return _pool


def shutdown() -> None:
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(shutdown)


async def run[T](fn: Callable[..., T], *args: Any) -> T:
    """`fn(*args)` in the worker process (fn and args must pickle), else in a thread."""
    global _broken
    if "<" in getattr(fn, "__qualname__", "<") or _broken >= MAX_BROKEN:  # a lambda or closure cannot pickle
        return await asyncio.to_thread(fn, *args)
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(_get_pool(), fn, *args)
    except (BrokenProcessPool, OSError, pickle.PicklingError, TypeError) as e:
        # TypeError: "cannot pickle ..." from the arguments; fn re-runs in a thread (pure work only)
        if isinstance(e, BrokenProcessPool | OSError):
            _broken += 1
            shutdown()
            if _broken >= MAX_BROKEN:
                log.error("offload: the worker process failed %d times; using threads from now on", _broken)
        log.warning("offload: worker process unusable (%s); using a thread", type(e).__name__)
        return await asyncio.to_thread(fn, *args)


def _noop() -> None:
    return None


async def warm() -> None:
    """Start the worker process now (agentd start), not under the first chat turn."""
    await run(_noop)
