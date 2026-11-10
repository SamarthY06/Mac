"""Per-conversation event streams (api §5.4): `(boot, seq)`, a 2,000-event replay buffer, sinks.

An event is encoded once. Each sink gets `(conv, line, epoch)`; the epoch is the client epoch of a
voice turn event (None otherwise), which the connection uses for the voice-only filter (B1).
"""

from __future__ import annotations

import collections
import time
from collections.abc import Iterable
from typing import Any, Protocol

from agentd import schema as s


class Sink(Protocol):
    def deliver(self, conv: str, line: bytes, epoch: str | None) -> None: ...


class GapError(Exception):
    def __init__(self, boot: str, oldest_seq: int) -> None:
        super().__init__("gap")
        self.data = s.GapData(boot=boot, oldest_seq=oldest_seq)


class Stream:
    def __init__(self, conv: str, boot: str, maxlen: int = s.REPLAY_BUFFER_EVENTS) -> None:
        self.conv = conv
        self.boot = boot
        self.seq = 0
        self._buf: collections.deque[tuple[int, bytes, str | None]] = collections.deque(maxlen=maxlen)
        self._sinks: dict[int, Sink] = {}

    @property
    def oldest_seq(self) -> int:
        return self._buf[0][0] if self._buf else self.seq + 1

    def emit(self, cls: type[s.Event], *, epoch: str | None = None, **fields: Any) -> s.Event:
        self.seq += 1
        ev = cls(conv=self.conv, boot=self.boot, seq=self.seq, **fields)
        line = s.encode_line(s.encode_event(ev))
        self._buf.append((self.seq, line, epoch))
        for sink in list(self._sinks.values()):
            sink.deliver(self.conv, line, epoch)
        return ev

    def check_after(self, after: s.Position | None) -> None:
        """Raise GapError unless the events after `after` are all still buffered."""
        if after is None:
            return
        if after.boot != self.boot or after.seq < self.oldest_seq - 1 or after.seq > self.seq:
            raise GapError(self.boot, self.oldest_seq)

    def attach(self, sink: Sink, after: s.Position | None) -> None:
        """Replay the events after `after` into `sink`, then deliver live ones. Synchronous, so no
        event can slip between the replay and the registration."""
        if after is not None:
            for seq, line, epoch in self._buf:
                if seq > after.seq:
                    sink.deliver(self.conv, line, epoch)
        self._sinks[id(sink)] = sink

    def detach(self, sink: Sink) -> None:
        self._sinks.pop(id(sink), None)

    def lines(self) -> Iterable[bytes]:  # tests
        return [line for _, line, _ in self._buf]


class LagMonitor:
    """Event-loop lag (V7): a sleeper's overshoot every `interval_s`; p99 and max of recent samples."""

    def __init__(self, interval_s: float = 0.1, window: int = 600) -> None:
        self.interval_s = interval_s
        self.samples: collections.deque[float] = collections.deque(maxlen=window)
        self.max_ms = 0.0

    async def run(self) -> None:
        import asyncio

        while True:
            t = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            lag = max(0.0, (time.perf_counter() - t - self.interval_s) * 1000)
            self.samples.append(lag)
            self.max_ms = max(self.max_ms, lag)

    def p99(self) -> float:
        if not self.samples:
            return 0.0
        xs = sorted(self.samples)
        return round(xs[min(len(xs) - 1, int(0.99 * (len(xs) - 1) + 0.5))], 2)

    def reset(self) -> None:
        self.samples.clear()
        self.max_ms = 0.0
