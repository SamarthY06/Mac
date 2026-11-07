"""Test backends: FileInput (wav / numpy array) and CaptureOutput (records playback)."""

from __future__ import annotations

import asyncio
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from macd.audio.resample import float_to_int16, int16_to_float, resample
from macd.contracts import FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger(__name__)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an int16 PCM wav (any channels) -> (float32 mono, sample_rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = int16_to_float(data)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    x = float_to_int16(pcm).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(x.tobytes())


class FileInput:
    """AudioInput that plays a file/array as 16 kHz frames.

    realtime=False yields frames as fast as the consumer takes them (yielding to the loop
    every frame); realtime=True paces at 20 ms per frame. ``trailing_silence_s`` of zeros
    is appended so endpointing can complete. ``frames()`` ends after the last frame.
    """

    def __init__(
        self,
        source: str | Path | np.ndarray,
        realtime: bool = False,
        trailing_silence_s: float = 1.5,
        sample_rate: int = SAMPLE_RATE,
        name: str = "file",
        requires_wake_word: bool = True,
    ) -> None:
        if isinstance(source, np.ndarray):
            pcm, sr = np.asarray(source, dtype=np.float32).reshape(-1), sample_rate
        else:
            pcm, sr = read_wav(source)
        pcm = resample(pcm, sr, SAMPLE_RATE)
        tail = np.zeros(int(trailing_silence_s * SAMPLE_RATE), dtype=np.float32)
        pcm = np.concatenate([pcm, tail])
        pad = (-len(pcm)) % FRAME_SAMPLES
        self._pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.float32)])
        self.realtime = realtime
        self.name = name
        self.requires_wake_word = requires_wake_word
        self._closed = asyncio.Event()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self._closed.set()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(0, len(self._pcm), FRAME_SAMPLES):
            if self._closed.is_set():
                return
            if self.realtime:
                delay = t0 + (i / SAMPLE_RATE) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
            yield self._pcm[i : i + FRAME_SAMPLES]


class CaptureOutput:
    """AudioOutput that records everything played (resampled to 16 kHz).

    realtime=False: ``play`` returns right after recording. realtime=True: ``play``
    sleeps for the audio duration (cancellable mid-way; only the played part is kept).
    """

    def __init__(self, realtime: bool = False, aec: bool = False, name: str = "capture") -> None:
        self.name = name
        self.realtime = realtime
        self._aec = aec
        self.chunks: list[tuple[np.ndarray, int]] = []  # (original pcm, sample_rate)
        self._played16: list[np.ndarray] = []
        self.flushes = 0
        self._flush_ev = asyncio.Event()

    @property
    def aec(self) -> bool:
        return self._aec

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self.flush()

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.chunks.append((x.copy(), sample_rate))
        x16 = resample(x, sample_rate, SAMPLE_RATE)
        if not self.realtime:
            self._played16.append(x16)
            await asyncio.sleep(0)
            return
        self._flush_ev = asyncio.Event()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        dur = len(x) / sample_rate
        try:
            await asyncio.wait_for(self._flush_ev.wait(), dur)
        except TimeoutError:
            pass
        finally:
            played = min(1.0, (loop.time() - t0) / dur) if dur else 1.0
            self._played16.append(x16[: int(len(x16) * played)])

    def flush(self) -> None:
        self.flushes += 1
        self._flush_ev.set()

    def captured(self) -> np.ndarray:
        """All played audio so far, concatenated, 16 kHz float32."""
        if not self._played16:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._played16)
