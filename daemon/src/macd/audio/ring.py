"""Preallocated float32 ring buffer with absolute sample indexing (ARCHITECTURE §3).

Threading model (single producer, single consumer):
- The producer (PortAudio callback thread, socket reader, or the loop itself) calls
  ``append``. It writes samples into the array *first* and only then publishes the new
  ``written`` count (a single attribute store, atomic under the GIL). A consumer that
  reads ``written`` therefore never sees an index for data that isn't in place yet.
- The consumer (one asyncio task) calls ``read_since``/``last``/``slice``. Because the
  producer may lap the consumer, a reader copies the region and then re-reads
  ``written`` (seqlock style): anything overwritten during the copy is trimmed and
  counted as an overrun, so callers never receive torn data.
- Wakeups: ``append`` schedules ``event.set`` on the loop via ``call_soon_threadsafe``
  only if no wakeup is already pending (``_signalled``), so a burst of N appends costs
  one loop callback, not N. The consumer clears the flag before re-checking for data,
  which closes the lost-wakeup race (see ``wait``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

log = logging.getLogger(__name__)


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._written = 0  # absolute index of the next sample to be written
        self.overruns = 0  # consumer reads that found data already overwritten
        self.lost_samples = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self._signalled = False
        self._closed = False

    # ------------------------------------------------------------------ producer side

    def bind(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Bind the wakeup event to an event loop (call from the consumer's loop)."""
        self._loop = loop or asyncio.get_running_loop()
        self._event = asyncio.Event()

    @property
    def bound(self) -> bool:
        return self._event is not None

    @property
    def written(self) -> int:
        return self._written

    def append(self, samples: np.ndarray) -> None:
        """Append samples (any length). Safe to call from one foreign thread."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return
        cap = self.capacity
        if n >= cap:
            x = x[-cap:]
            start = self._written + n - cap
            m = cap
        else:
            start = self._written
            m = n
        pos = start % cap
        first = min(m, cap - pos)
        self._buf[pos : pos + first] = x[:first]
        if first < m:
            self._buf[: m - first] = x[first:]
        self._written = self._written + n  # publish after data is in place
        self._notify()

    def close(self) -> None:
        """Mark closed and wake the consumer (it should then stop)."""
        self._closed = True
        self._notify(force=True)

    @property
    def closed(self) -> bool:
        return self._closed

    def _notify(self, force: bool = False) -> None:
        loop, ev = self._loop, self._event
        if loop is None or ev is None:
            return
        if self._signalled and not force:
            return
        self._signalled = True
        try:
            loop.call_soon_threadsafe(ev.set)
        except RuntimeError:  # loop closed during shutdown
            log.debug("ring wakeup dropped: loop closed")

    # ------------------------------------------------------------------ consumer side

    async def wait(self, since: int, timeout: float | None = None) -> bool:
        """Wait until ``written > since`` (or closed). Returns False on timeout."""
        if self._event is None:
            self.bind()
        assert self._event is not None
        while True:
            # Clear first, then re-check: an append racing with us either is visible in
            # the check or sees _signalled False and schedules a fresh set().
            self._event.clear()
            self._signalled = False
            if self._written > since or self._closed:
                return True
            try:
                if timeout is None:
                    await self._event.wait()
                else:
                    await asyncio.wait_for(self._event.wait(), timeout)
            except TimeoutError:
                return self._written > since or self._closed

    def read_since(self, abs_index: int) -> tuple[np.ndarray, int]:
        """Return (samples from abs_index up to now, next index).

        If the producer has already overwritten part of that range, the result starts at
        the oldest valid sample and the overrun counter is incremented.
        """
        end = self._written
        if abs_index >= end:
            return np.zeros(0, dtype=np.float32), max(abs_index, end)
        data, start = self._copy(abs_index, end)
        if start > abs_index:
            self.overruns += 1
            self.lost_samples += start - abs_index
        return data, end

    def slice(self, start: int, end: int) -> np.ndarray:
        """Samples in [start, end) still retained (clamped to the available window)."""
        end = min(end, self._written)
        if start >= end:
            return np.zeros(0, dtype=np.float32)
        data, _ = self._copy(start, end)
        return data

    @property
    def oldest(self) -> int:
        return max(0, self._written - self.capacity)

    def _copy(self, start: int, end: int) -> tuple[np.ndarray, int]:
        cap = self.capacity
        start = max(start, end - cap, 0)
        n = end - start
        pos = start % cap
        first = min(n, cap - pos)
        out = np.empty(n, dtype=np.float32)
        out[:first] = self._buf[pos : pos + first]
        if first < n:
            out[first:] = self._buf[: n - first]
        # Seqlock check: anything the producer overwrote during the copy is invalid.
        valid_from = self._written - cap
        if valid_from > start:
            cut = min(valid_from - start, n)
            out = out[cut:]
            start += cut
        return out, start


async def ring_frames(ring: RingBuffer, frame_samples: int, poll_s: float = 0.25) -> AsyncIterator[np.ndarray]:
    """Async-iterate fixed-size frames out of ``ring`` (from "now") until it is closed.

    Must be the ring's only consumer. Yields copies, so callers may keep them.
    """
    if not ring.bound:
        ring.bind()
    idx = ring.written
    pending = np.zeros(0, dtype=np.float32)
    while not ring.closed:
        await ring.wait(idx, timeout=poll_s)
        data, idx = ring.read_since(idx)
        if data.size == 0:
            continue
        pending = data if pending.size == 0 else np.concatenate([pending, data])
        n = pending.size // frame_samples
        for k in range(n):
            yield pending[k * frame_samples : (k + 1) * frame_samples]
        pending = pending[n * frame_samples :]
