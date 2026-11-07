"""SpeechQueue: the ONLY path to an AudioOutput (ARCHITECTURE §8).

- Ordered items `(turn_id, kind, text | pcm, output)`; one consumer task (`run()`).
- Pipelining: while item N plays, TTS for the next text item is prefetched.
- Generations: `set_current_turn(tid)` makes every item of an older turn stale. Stale
  items are dropped when dequeued, and synthesis results that come back for a stale turn
  are discarded (we never rely on cancelling the C inference call).
- `cancel_turn(tid)` flushes the output synchronously (sub-ms) and drops queued items.
- turn_id 0 is "system" speech (announcements, "Okay, stopped."): never stale, only
  removed by `flush_all()`.
- `drain(tid)` waits until everything queued for `tid` has played or been dropped.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from macd.contracts import TTS, AudioOutput

log = logging.getLogger(__name__)

SpeechKind = Literal["say", "answer", "narration", "earcon", "filler", "confirm"]
SYSTEM_TURN = 0


@dataclass(eq=False)
class SpeechItem:
    turn_id: int
    kind: SpeechKind
    text: str = ""
    pcm: np.ndarray | None = None
    sample_rate: int = 0
    output: str | None = None  # None = default output
    priority: bool = False  # jumps the queue (steer ack, wake earcon, "Okay, stopped.")
    on_play: Callable[[], None] | None = None  # called when audio for this item starts
    enqueued_at: float = field(default_factory=time.monotonic)
    _synth: asyncio.Task[np.ndarray] | None = field(default=None, repr=False)


def _retrieve(task: asyncio.Task[Any]) -> None:
    """Mark a (possibly discarded) synthesis task's exception as retrieved."""
    if not task.cancelled():
        task.exception()


class SpeechQueue:
    def __init__(
        self,
        tts: TTS | None,
        outputs: dict[str, AudioOutput],
        *,
        default_output: str = "local",
        maxsize: int = 128,
        on_start: Callable[[str, str], None] | None = None,  # (text, kind) when audio starts
        on_idle: Callable[[], None] | None = None,
        on_drop: Callable[[SpeechItem, str], None] | None = None,
    ) -> None:
        if default_output not in outputs and outputs:
            default_output = next(iter(outputs))
        self.tts = tts
        self.outputs: dict[str, AudioOutput] = dict(outputs)
        self.default_output = default_output
        self.maxsize = maxsize
        self.on_start = on_start
        self.on_idle = on_idle
        self.on_drop = on_drop
        self._q: collections.deque[SpeechItem] = collections.deque()
        self._wake = asyncio.Event()
        self._current_turn = 0
        self._cancelled: set[int] = set()
        self._playing: SpeechItem | None = None
        self._play_task: asyncio.Task[None] | None = None
        self._render_task: asyncio.Task[np.ndarray] | None = None
        self._pending: collections.Counter[int] = collections.Counter()
        self._drain_waiters: dict[int, list[asyncio.Future[None]]] = {}
        self.speaking = False
        self.dropped = 0
        self.played = 0
        self.last_activity = time.monotonic()

    # ------------------------------------------------------------------ outputs

    def add_output(self, name: str, output: AudioOutput) -> None:
        self.outputs[name] = output

    def remove_output(self, name: str) -> None:
        out = self.outputs.pop(name, None)
        if out is not None and self._playing is not None and self._out_name(self._playing) == name:
            out.flush()
            self._cancel_play()

    def has_output(self, name: str) -> bool:
        return name in self.outputs

    def _out_name(self, item: SpeechItem) -> str:
        name = item.output or self.default_output
        return name if name in self.outputs else self.default_output

    # ------------------------------------------------------------------ generations

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def is_stale(self, turn_id: int) -> bool:
        if turn_id == SYSTEM_TURN:
            return False
        return turn_id < self._current_turn or turn_id in self._cancelled

    def set_current_turn(self, turn_id: int) -> None:
        """Advance the generation: drop all queued/in-flight speech of older turns."""
        if turn_id < self._current_turn:
            return
        self._current_turn = turn_id
        self._cancelled = {t for t in self._cancelled if t >= turn_id}
        self._purge(lambda it: self.is_stale(it.turn_id), "stale")
        if self._playing is not None and self.is_stale(self._playing.turn_id):
            self._stop_current()

    def cancel_turn(self, turn_id: int) -> None:
        """Stop `turn_id`'s speech NOW (sync) and drop its queued items."""
        if turn_id == SYSTEM_TURN:
            return
        self._cancelled.add(turn_id)
        self._purge(lambda it: it.turn_id == turn_id or self.is_stale(it.turn_id), "cancelled")
        if self._playing is not None and self._playing.turn_id == turn_id:
            self._stop_current()

    def flush_all(self) -> None:
        """Stop everything, including system speech."""
        self._purge(lambda it: True, "flushed")
        if self._playing is not None:
            self._stop_current()
        for out in self.outputs.values():
            with contextlib.suppress(Exception):
                out.flush()

    def _stop_current(self) -> None:
        item = self._playing
        if item is not None:
            out = self.outputs.get(self._out_name(item))
            if out is not None:
                with contextlib.suppress(Exception):
                    out.flush()
        self._cancel_play()

    def _cancel_play(self) -> None:
        if self._play_task is not None and not self._play_task.done():
            self._play_task.cancel()
        if self._render_task is not None and not self._render_task.done():
            self._render_task.cancel()

    def _purge(self, pred: Callable[[SpeechItem], bool], reason: str) -> None:
        keep: collections.deque[SpeechItem] = collections.deque()
        for it in self._q:
            if pred(it):
                self._drop(it, reason)
            else:
                keep.append(it)
        self._q = keep

    # ------------------------------------------------------------------ producer API

    def enqueue(self, item: SpeechItem) -> bool:
        """Non-blocking. Returns False if dropped (stale or queue full)."""
        if self.is_stale(item.turn_id):
            self.dropped += 1
            return False
        if item.kind != "earcon" and not item.text.strip() and item.pcm is None:
            return False
        if len(self._q) >= self.maxsize:
            # Shed the least important queued item first; never drop answers silently
            # unless the whole queue is answers.
            victim = next((it for it in self._q if it.kind in ("narration", "filler", "earcon")), None)
            if victim is None:
                victim = self._q[0]
            self._q.remove(victim)
            self._drop(victim, "overflow")
        if item.priority:
            self._q.appendleft(item)
        else:
            self._q.append(item)
        self._pending[item.turn_id] += 1
        self._wake.set()
        return True

    def say(
        self,
        turn_id: int,
        text: str,
        kind: SpeechKind = "say",
        *,
        output: str | None = None,
        priority: bool = False,
        on_play: Callable[[], None] | None = None,
    ) -> bool:
        return self.enqueue(
            SpeechItem(turn_id=turn_id, kind=kind, text=text, output=output, priority=priority, on_play=on_play)
        )

    def play_pcm(
        self,
        turn_id: int,
        pcm: np.ndarray,
        sample_rate: int,
        kind: SpeechKind = "earcon",
        *,
        output: str | None = None,
        priority: bool = False,
        on_play: Callable[[], None] | None = None,
    ) -> bool:
        return self.enqueue(
            SpeechItem(turn_id=turn_id, kind=kind, pcm=pcm, sample_rate=sample_rate, output=output,
                       priority=priority, on_play=on_play)
        )

    def pending(self, turn_id: int) -> int:
        return self._pending.get(turn_id, 0)

    @property
    def queued(self) -> int:
        return len(self._q)

    async def drain(self, turn_id: int, timeout: float | None = None) -> bool:
        """Wait until all speech queued for `turn_id` so far has played (or was dropped).
        Returns False if `timeout` elapsed first (the output may be stalled)."""
        if self._pending.get(turn_id, 0) <= 0:
            return True
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._drain_waiters.setdefault(turn_id, []).append(fut)
        try:
            await asyncio.wait_for(fut, timeout)
            return True
        except TimeoutError:
            log.warning("speech for turn %d did not drain within %.0fs", turn_id, timeout or 0)
            return False
        finally:
            waiters = self._drain_waiters.get(turn_id)
            if waiters and fut in waiters:
                waiters.remove(fut)
                if not waiters:
                    del self._drain_waiters[turn_id]

    # ------------------------------------------------------------------ accounting

    def _drop(self, item: SpeechItem, reason: str, *, finish: bool = True) -> None:
        """Account for a dropped item. `finish=False` when the consumer's `run()` loop will
        call `_finish` itself (it does so for every dequeued item)."""
        self.dropped += 1
        self._cancel_synth(item)
        if finish:
            self._finish(item)
        if self.on_drop is not None:
            with contextlib.suppress(Exception):
                self.on_drop(item, reason)

    def _finish(self, item: SpeechItem) -> None:
        n = self._pending.get(item.turn_id, 0) - 1
        if n <= 0:
            self._pending.pop(item.turn_id, None)
            for fut in self._drain_waiters.pop(item.turn_id, []):
                if not fut.done():
                    fut.set_result(None)
        else:
            self._pending[item.turn_id] = n

    @staticmethod
    def _cancel_synth(item: SpeechItem) -> None:
        t = item._synth
        if t is not None and not t.done():
            t.cancel()
        item._synth = None

    # ------------------------------------------------------------------ consumer

    def _synthesize(self, item: SpeechItem) -> asyncio.Task[np.ndarray]:
        if item._synth is None:
            task = asyncio.create_task(self._tts(item.text), name=f"tts:{item.turn_id}")
            task.add_done_callback(_retrieve)
            item._synth = task
        return item._synth

    async def _tts(self, text: str) -> np.ndarray:
        if self.tts is None:
            raise RuntimeError("no TTS configured")
        return np.asarray(await self.tts.synthesize(text), dtype=np.float32)

    def _prefetch_next(self) -> None:
        for it in self._q:
            if self.is_stale(it.turn_id):
                continue
            if it.pcm is None and it._synth is None and it.text:
                self._synthesize(it)
            return

    async def _next_item(self) -> SpeechItem:
        while not self._q:
            if self.speaking:
                self.speaking = False
                if self.on_idle is not None:
                    with contextlib.suppress(Exception):
                        self.on_idle()
            self._wake.clear()
            await self._wake.wait()
        return self._q.popleft()

    async def run(self) -> None:
        """The single consumer task. Never returns unless cancelled."""
        try:
            while True:
                item = await self._next_item()
                try:
                    await self._process(item)
                finally:
                    self._finish(item)
        finally:
            for it in list(self._q):
                self._cancel_synth(it)
            self._playing = None
            if self.speaking:
                self.speaking = False

    async def _process(self, item: SpeechItem) -> None:
        if self.is_stale(item.turn_id):
            self._drop(item, finish=False, reason="stale")
            return
        out_name = self._out_name(item)
        out = self.outputs.get(out_name)
        if out is None:
            self._drop(item, finish=False, reason="no_output")
            return
        self._playing = item
        try:
            if item.pcm is not None:
                pcm, sr = item.pcm, item.sample_rate or (self.tts.sample_rate if self.tts else 16000)
            else:
                self._render_task = self._synthesize(item)
                # The task could be cancelled by cancel_turn(); asyncio.wait never raises
                # for the awaited task, so the consumer itself is not cancelled.
                await asyncio.wait({self._render_task})
                task, self._render_task = self._render_task, None
                if task.cancelled():
                    self._drop(item, finish=False, reason="cancelled")
                    return
                exc = task.exception()
                if exc is not None:
                    log.warning("TTS failed for %r: %s", item.text[:60], exc)
                    self._drop(item, finish=False, reason="tts_error")
                    return
                pcm = task.result()
                sr = self.tts.sample_rate if self.tts else 24000
            if self.is_stale(item.turn_id):  # turn changed while synthesizing
                self._drop(item, finish=False, reason="stale")
                return
            if pcm.size == 0:
                return
            self._prefetch_next()
            if not self.speaking:
                self.speaking = True
            if self.on_start is not None:
                with contextlib.suppress(Exception):
                    self.on_start(item.text, item.kind)
            if item.on_play is not None:
                with contextlib.suppress(Exception):
                    item.on_play()
            self.last_activity = time.monotonic()
            self._play_task = asyncio.create_task(out.play(pcm, sr), name=f"play:{out_name}")
            await asyncio.wait({self._play_task})
            task, self._play_task = self._play_task, None
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    log.warning("output %s play failed: %s", out_name, exc)
                else:
                    self.played += 1
            self.last_activity = time.monotonic()
        finally:
            self._playing = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "speaking": self.speaking,
            "queued": len(self._q),
            "current_turn": self._current_turn,
            "played": self.played,
            "dropped": self.dropped,
            "outputs": sorted(self.outputs),
        }
