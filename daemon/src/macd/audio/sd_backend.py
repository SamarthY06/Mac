"""sounddevice (PortAudio) backends for standalone mode (no Swift app).

Threading: PortAudio callbacks run on a PortAudio thread.
- Input: the callback resamples to 16 kHz and appends to a RingBuffer (single producer;
  see ring.py for the lock-free handoff). ``_last_cb`` is a float written by the callback
  and read by the watchdog task (atomic under the GIL; staleness of one read is harmless).
- Output: a small deque of pending items shared between the loop (enqueue / cancel) and
  the callback (consume). Guarded by a threading.Lock held only for O(1) work, so the
  realtime thread never blocks meaningfully. Completion is signalled back with
  ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from macd.audio.resample import StreamResampler, resample
from macd.audio.ring import RingBuffer, ring_frames
from macd.config import Config
from macd.contracts import FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger(__name__)

_BLUETOOTH_HINTS = ("airpods", "bluetooth", "beats", "headset", "hands-free", "handsfree", "bose", "sony wh", "jabra", "buds")
_BUILTIN_HINTS = ("macbook", "built-in", "builtin", "internal")
WATCHDOG_S = 1.0
DRAIN_SLACK_S = 1.5  # play() waits expected duration + this; longer = stalled output callback


def _is_bluetooth(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _BLUETOOTH_HINTS)


def _is_builtin(name: str) -> bool:
    n = name.lower()
    return "microphone" in n and any(h in n for h in _BUILTIN_HINTS)


def choose_input_device(devices: list[dict[str, Any]], default_index: int | None, requested: str = "", prefer_builtin: bool = True) -> int:
    """Pick an input device index. Never auto-picks a Bluetooth mic (forces HFP)."""
    inputs = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    if not inputs:
        raise RuntimeError("no audio input devices")
    if requested:
        for i, d in inputs:
            if requested.lower() in str(d["name"]).lower():
                return i
        raise RuntimeError(f"input device {requested!r} not found")
    if prefer_builtin:
        for i, d in inputs:
            if _is_builtin(str(d["name"])):
                return i
    if default_index is not None and default_index >= 0:
        for i, d in inputs:
            if i == default_index and not _is_bluetooth(str(d["name"])):
                return i
    for i, d in inputs:
        if not _is_bluetooth(str(d["name"])):
            return i
    raise RuntimeError("only Bluetooth inputs available; set audio.input_device explicitly to use one")


class SoundDeviceInput:
    """AudioInput from a PortAudio device, 16 kHz 320-sample frames, with a watchdog."""

    def __init__(self, cfg: Config | None = None, name: str = "local") -> None:
        self.cfg = cfg or Config()
        self.name = name
        self.requires_wake_word = True
        self.ring = RingBuffer(SAMPLE_RATE * 10)
        self.reopens = 0
        self.status_flags = 0
        self.device_name = ""
        self._stream: Any = None
        self._resampler: StreamResampler | None = None
        self._last_cb = 0.0
        self._watchdog: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def aec(self) -> bool:
        return False

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # PortAudio thread: no allocation-heavy or blocking work beyond the resample.
        if status:
            self.status_flags += 1
        self._last_cb = time.monotonic()
        rs = self._resampler
        if rs is not None:
            self.ring.append(rs.process(indata[:, 0]))

    def _open(self) -> None:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
        default_in = sd.default.device[0] if sd.default.device else None
        idx = choose_input_device(devices, default_in, self.cfg.audio.input_device, self.cfg.audio.prefer_builtin_mic)
        dev = devices[idx]
        sr = int(dev.get("default_samplerate") or 48000)
        self.device_name = str(dev["name"])
        self._resampler = StreamResampler(sr, SAMPLE_RATE)
        stream = sd.InputStream(device=idx, samplerate=sr, channels=1, dtype="float32", blocksize=int(sr * 0.02), callback=self._callback)
        stream.start()
        self._stream = stream
        self._last_cb = time.monotonic()
        log.info("mic opened: %s @ %d Hz", self.device_name, sr)

    def _close_stream(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # PortAudio errors on a dead device are expected
                log.warning("error closing input stream", exc_info=True)

    async def start(self) -> None:
        self.ring.bind()
        await asyncio.to_thread(self._open)
        self._watchdog = asyncio.create_task(self._watch(), name="sd-input-watchdog")

    async def _watch(self) -> None:
        while not self._closed:
            await asyncio.sleep(0.5)
            if time.monotonic() - self._last_cb > WATCHDOG_S:
                self.reopens += 1
                log.warning("no mic frames for > %.1fs; reopening input stream", WATCHDOG_S)
                try:
                    await asyncio.to_thread(self._close_stream)
                    await asyncio.to_thread(self._open)
                except Exception:
                    log.exception("reopening input stream failed; retrying")
                    self._last_cb = time.monotonic()  # back off one watchdog period

    async def close(self) -> None:
        self._closed = True
        if self._watchdog is not None:
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog
        await asyncio.to_thread(self._close_stream)
        self.ring.close()

    def frames(self) -> AsyncIterator[np.ndarray]:
        return ring_frames(self.ring, FRAME_SAMPLES)


@dataclass
class _Item:
    pcm: np.ndarray
    fut: asyncio.Future[None]
    pos: int = 0
    cancelled: bool = False


class SoundDeviceOutput:
    """AudioOutput on the default PortAudio output device (one persistent stream)."""

    def __init__(self, cfg: Config | None = None, device: int | str | None = None, sample_rate: int = 48_000) -> None:
        self.name = "sounddevice"
        self.cfg = cfg or Config()
        self.device = device
        self.sample_rate = sample_rate
        self._stream: Any = None
        self._items: deque[_Item] = deque()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.reopens = 0

    @property
    def aec(self) -> bool:
        return False

    def _done(self, item: _Item) -> None:
        loop = self._loop
        if loop is not None and not item.fut.done():
            loop.call_soon_threadsafe(lambda: item.fut.done() or item.fut.set_result(None))

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        out = outdata[:, 0]
        filled = 0
        finished: list[_Item] = []
        with self._lock:
            while filled < frames and self._items:
                it = self._items[0]
                if it.cancelled:
                    self._items.popleft()
                    finished.append(it)
                    continue
                n = min(frames - filled, it.pcm.shape[0] - it.pos)
                out[filled : filled + n] = it.pcm[it.pos : it.pos + n]
                it.pos += n
                filled += n
                if it.pos >= it.pcm.shape[0]:
                    self._items.popleft()
                    finished.append(it)
        out[filled:] = 0.0
        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]
        for it in finished:
            self._done(it)

    def _open(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(device=self.device, samplerate=self.sample_rate, channels=1, dtype="float32", latency="low", callback=self._callback)
        self._stream.start()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._stream is None:
            await asyncio.to_thread(self._open)

    def _close_stream(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:  # PortAudio errors on a dead device are expected
                log.warning("error closing output stream", exc_info=True)

    async def close(self) -> None:
        self.flush()
        await asyncio.to_thread(self._close_stream)

    async def _reopen(self) -> None:
        """Recover from a stalled output callback (device gone/sleeping): drop the queue, reopen."""
        self.reopens += 1
        self.flush()
        await asyncio.to_thread(self._close_stream)
        try:
            await asyncio.to_thread(self._open)
        except Exception:
            log.exception("reopening output stream failed; will retry on next play()")

    def _drop(self, item: _Item) -> None:
        with self._lock:
            item.cancelled = True
            with contextlib.suppress(ValueError):
                self._items.remove(item)

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if self._stream is None:
            await self.start()
        x = resample(np.asarray(pcm, dtype=np.float32), sample_rate, self.sample_rate)
        if x.size == 0:
            return
        item = _Item(np.ascontiguousarray(x), asyncio.get_running_loop().create_future())
        with self._lock:
            self._items.append(item)
        timeout = x.size / self.sample_rate + DRAIN_SLACK_S
        try:
            await asyncio.wait_for(asyncio.shield(item.fut), timeout)
        except TimeoutError:
            log.warning("output stalled: %.2fs of audio not finished after %.1fs; reopening stream", x.size / self.sample_rate, timeout)
            self._drop(item)
            await self._reopen()
        except asyncio.CancelledError:
            self._drop(item)
            raise

    def flush(self) -> None:
        with self._lock:
            items = list(self._items)
            self._items.clear()
        for it in items:
            it.cancelled = True
            if not it.fut.done():
                it.fut.set_result(None)
