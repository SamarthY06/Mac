"""sounddevice backends with fake streams (no real audio hardware)."""

import asyncio

import numpy as np
import pytest

from macd.audio import sd_backend
from macd.audio.sd_backend import SoundDeviceOutput, choose_input_device


class FakeStream:
    """A PortAudio stream whose callback is driven by the test (or never)."""

    def __init__(self, callback, blocksize: int = 480) -> None:
        self.callback, self.blocksize = callback, blocksize
        self.stopped = self.closed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def pump(self) -> None:
        out = np.zeros((self.blocksize, 1), np.float32)
        self.callback(out, self.blocksize, None, None)


def make_output(monkeypatch) -> tuple[SoundDeviceOutput, list[FakeStream]]:
    out = SoundDeviceOutput(sample_rate=48_000)
    streams: list[FakeStream] = []

    def fake_open() -> None:
        streams.append(FakeStream(out._callback))
        out._stream = streams[-1]

    monkeypatch.setattr(out, "_open", fake_open)
    return out, streams


async def test_play_completes_when_callback_drains(monkeypatch):
    out, streams = make_output(monkeypatch)
    await out.start()
    task = asyncio.create_task(out.play(np.zeros(2400, np.float32), 48_000))
    await asyncio.sleep(0.01)
    for _ in range(5):  # 5 x 480 = 2400 samples
        streams[0].pump()
        await asyncio.sleep(0)
    await asyncio.wait_for(task, 1)
    assert out.reopens == 0 and not out._items


async def test_play_times_out_and_reopens_on_stalled_callback(monkeypatch):
    monkeypatch.setattr(sd_backend, "DRAIN_SLACK_S", 0.2)
    out, streams = make_output(monkeypatch)
    await out.start()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.wait_for(out.play(np.zeros(4800, np.float32), 48_000), 2)  # 0.1 s of audio, never drained
    assert 0.25 < loop.time() - t0 < 1.5
    assert out.reopens == 1 and streams[0].stopped and streams[0].closed and len(streams) == 2
    assert not out._items  # the stalled item was dropped, nothing queued for the new stream


async def test_cancel_removes_item(monkeypatch):
    out, _ = make_output(monkeypatch)
    await out.start()
    task = asyncio.create_task(out.play(np.zeros(48_000, np.float32), 48_000))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not out._items


def test_choose_input_device_never_bluetooth():
    devs = [
        {"name": "AirPods Pro", "max_input_channels": 1},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        {"name": "External Speakers", "max_input_channels": 0},
    ]
    assert choose_input_device(devs, default_index=0) == 1
    assert choose_input_device(devs, default_index=0, prefer_builtin=False) == 1
    assert choose_input_device(devs, default_index=0, requested="airpods") == 0
    with pytest.raises(RuntimeError):
        choose_input_device(devs[:1], default_index=0)
