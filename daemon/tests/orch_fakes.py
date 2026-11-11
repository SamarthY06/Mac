"""Minimal fakes for orchestrator / speech-queue tests (no models, no audio hardware).

The brain is the real scripted one: macd.brain.fake_brain.FakeBrain.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import numpy as np

from macd.contracts import (
    ListenerEvent,
    ListenerEventKind,
    Utterance,
    WakeHit,
    WindowReason,
)


class FakeTTS:
    """Encodes the text index in the PCM so outputs can decode what was 'spoken'."""

    sample_rate = 16_000

    def __init__(self, delay: float = 0.0, samples: int = 160) -> None:
        self.delay = delay
        self.samples = samples  # PCM length per text (16 kHz): set it to the output's play_s for barge-in tests
        self.texts: list[str] = []
        self.started: list[tuple[str, float]] = []
        self.finished: list[tuple[str, float]] = []

    def encode(self, text: str) -> np.ndarray:
        self.texts.append(text)
        return np.full(self.samples, float(len(self.texts) - 1), dtype=np.float32)

    def decode(self, pcm: np.ndarray) -> str:
        i = int(pcm[0])
        return self.texts[i] if 0 <= i < len(self.texts) else f"<pcm:{len(pcm)}>"

    async def synthesize(self, text: str) -> np.ndarray:
        self.started.append((text, time.monotonic()))
        if self.delay:
            await asyncio.sleep(self.delay)
        self.finished.append((text, time.monotonic()))
        return self.encode(text)


class FakeOutput:
    def __init__(self, tts: FakeTTS, name: str = "local", play_s: float = 0.005, aec: bool = True) -> None:
        self.name = name
        self.tts = tts
        self.play_s = play_s
        self._aec = aec
        self.spoken: list[str] = []
        self.play_started: list[tuple[str, float]] = []
        self.cancelled_at: list[float] = []
        self.flushes = 0

    @property
    def aec(self) -> bool:
        return self._aec

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        text = self.tts.decode(pcm) if sample_rate == self.tts.sample_rate else f"<earcon@{sample_rate}>"
        self.play_started.append((text, time.monotonic()))
        try:
            await asyncio.sleep(self.play_s)
        except asyncio.CancelledError:
            self.cancelled_at.append(time.monotonic())
            raise
        self.spoken.append(text)

    def flush(self) -> None:
        self.flushes += 1


class FakeListener:
    def __init__(self, source: str = "local") -> None:
        self.source_name = source
        self.speaking = False
        self.muted = False
        self.windows: list[tuple[float, WindowReason]] = []
        self.closed_windows = 0
        self._q: asyncio.Queue[ListenerEvent] = asyncio.Queue()

    async def run(self) -> None:
        await asyncio.Event().wait()

    async def events(self) -> AsyncIterator[ListenerEvent]:
        while True:
            yield await self._q.get()

    def open_window(self, seconds: float, reason: WindowReason) -> None:
        self.windows.append((seconds, reason))

    def close_window(self) -> None:
        self.closed_windows += 1

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def set_speaking(self, speaking: bool, current_text: str = "") -> None:
        self.speaking = speaking

    # test helpers
    def push(self, ev: ListenerEvent) -> None:
        self._q.put_nowait(ev)

    def wake(self) -> None:
        self.push(ListenerEvent(ListenerEventKind.WAKE, self.source_name, wake=WakeHit("hey mac", 0)))

    def say(self, text: str, woke: bool = False, *, window: WindowReason | None = None,
            continuation: bool = False, same_breath: bool = False) -> None:
        """Deliver one utterance; `woke=True` also emits the preceding WAKE event."""
        now = time.monotonic()
        if woke:
            self.wake()
        utt = Utterance(text=text, source=self.source_name, woke=woke, started_at=now - 1, ended_at=now, audio_s=1.0,
                        window=window, same_breath=same_breath, continuation=continuation)
        self.push(ListenerEvent(ListenerEventKind.UTTERANCE, self.source_name, utterance=utt))

    def reasons(self) -> list[WindowReason]:
        return [r for _, r in self.windows]


class FakeGate:
    def __init__(self) -> None:
        self.resolved: list[bool] = []
        self.revokes = 0

    def resolve(self, approved: bool) -> None:
        self.resolved.append(approved)

    def revoke_all(self) -> None:
        self.revokes += 1


async def wait_until(cond: Callable[[], Any], timeout: float = 2.0, msg: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for condition {msg}")
        await asyncio.sleep(0.002)
