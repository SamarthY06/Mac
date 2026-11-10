"""The desktop lock (W3.4, F3): one in-process lock over the shared screen and Chrome.

Holders are turns: `TurnToken(conv, turn_id, cancelled)`. The safeguards:

- bounded waits: `acquire(token, timeout)` gives up after `timeout` (15 s voice, 45 s chat);
- turn token: after any wait the token is re-checked, and a stopped turn never keeps the lock;
- idle release: the lock frees itself `idle_release_s` (20 s) after the last `touch`;
- confirmation release: `release_conv(conv)` while that conversation's confirmation is open;
- teardown: `cancel_turn(conv)` fails that conversation's waiters and frees its hold.

`handle(conv, label, what)` adapts the lock to the brain's polled `DesktopLock` protocol
(`try_acquire()` / `release()`, macd/brain/hooks.py), so every ClaudeBrain in agentd shares it. A
handle holder is governed by the brain's own DesktopGuard (wait, idle, in-flight tools); agentd's
`release_conv` and `cancel_turn` still free it.
"""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from agentd import schema as s

log = logging.getLogger(__name__)

IDLE_RELEASE_S = 20.0
VOICE_WAIT_S = 15.0
CHAT_WAIT_S = 45.0


@dataclass(eq=False)
class TurnToken:
    conv: str
    turn_id: int
    label: str = ""  # "voice" | "chat": who `status.desktop.holder` names
    what: str = ""  # the running request, for `status.desktop.what`
    cancelled: Callable[[], bool] = field(default=lambda: False)


class DesktopLock:
    def __init__(self, *, idle_release_s: float = IDLE_RELEASE_S) -> None:
        self.idle_release_s = idle_release_s
        self.holder: TurnToken | None = None
        self._idle_timer: asyncio.TimerHandle | None = None
        self._idle_managed = True  # False while a brain's own guard governs the hold
        self._waiters: collections.deque[tuple[TurnToken, asyncio.Future[bool]]] = collections.deque()

    # -- async API ---------------------------------------------------------------------
    async def acquire(self, token: TurnToken, timeout: float) -> bool:
        """True once `token` holds the lock; False on timeout or if its turn stopped."""
        if token.cancelled():
            return False
        self._drop_dead_holder()
        if self.holder is None or self.holder is token:
            self._grant(token)
            return True
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._waiters.append((token, fut))
        try:
            await asyncio.wait({fut}, timeout=max(0.0, timeout))
        except BaseException:  # the waiting task itself was cancelled
            if fut.done() and fut.result():
                self.release(token)
            self._settle(fut)
            raise
        self._settle(fut)
        granted = fut.result()
        if granted and token.cancelled():  # F3 (a): a stopped turn's waiter never keeps it
            self.release(token)
            return False
        return granted

    def touch(self, token: TurnToken) -> None:
        """A desktop tool finished (PostToolUse): restart the idle clock."""
        if self.holder is token:
            self._schedule_idle()

    def release(self, token: TurnToken) -> None:
        if self.holder is token:
            self._free()

    def release_conv(self, conv: str) -> None:
        """Confirmation open in `conv`: free its hold (the next desktop tool acquires again)."""
        if self.holder is not None and self.holder.conv == conv:
            self._free()

    def cancel_turn(self, conv: str) -> None:
        """Turn teardown in `conv`: its waiters fail and its hold is freed."""
        for token, fut in list(self._waiters):
            if token.conv == conv and not fut.done():
                fut.set_result(False)
                self._remove_waiter(fut)
        self.release_conv(conv)

    def status(self) -> s.DesktopStatus | None:
        h = self.holder
        return s.DesktopStatus(holder=h.label or h.conv, what=h.what) if h else None

    # -- polled protocol for the brain's DesktopGuard --------------------------------------
    def handle(self, conv: str, label: str, what: Callable[[], str] = lambda: "") -> DesktopHandle:
        return DesktopHandle(self, conv, label, what)

    def try_acquire(self, token: TurnToken, *, idle_managed: bool = True) -> bool:
        self._drop_dead_holder()
        if self.holder is token:
            return True
        if self.holder is None and not self._waiters:
            self._grant(token, idle_managed=idle_managed)
            return True
        return False

    # -- internals -----------------------------------------------------------------------
    def _drop_dead_holder(self) -> None:
        if self.holder is not None and self.holder.cancelled():
            log.info("desktop: freeing the hold of a stopped turn in %s", self.holder.conv)
            self._free()

    def _grant(self, token: TurnToken, *, idle_managed: bool = True) -> None:
        self.holder = token
        self._idle_managed = idle_managed
        self._schedule_idle()

    def _free(self) -> None:
        self._cancel_idle()
        self.holder = None
        while self._waiters:
            token, fut = self._waiters.popleft()
            if fut.done():
                continue
            if token.cancelled():
                fut.set_result(False)
                continue
            self._grant(token)
            fut.set_result(True)
            return

    def _settle(self, fut: asyncio.Future[bool]) -> None:
        if not fut.done():
            fut.set_result(False)
            self._remove_waiter(fut)

    def _remove_waiter(self, fut: asyncio.Future[bool]) -> None:
        self._waiters = collections.deque((t, f) for t, f in self._waiters if f is not fut)

    def _schedule_idle(self) -> None:
        self._cancel_idle()
        if not self._idle_managed:
            return
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
        if self.holder is not None:
            log.info("desktop: released after %.0f s idle", self.idle_release_s)
            self._free()


class DesktopHandle:
    """One brain's view of the shared lock, as the polled `DesktopLock` protocol of hooks.py."""

    def __init__(self, lock: DesktopLock, conv: str, label: str, what: Callable[[], str]) -> None:
        self.lock = lock
        self.token = TurnToken(conv, 0, label)
        self._what = what

    def try_acquire(self) -> bool:
        self.token.what = self._what()
        return self.lock.try_acquire(self.token, idle_managed=False)

    def release(self) -> None:
        self.lock.release(self.token)
