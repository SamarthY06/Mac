import asyncio
import threading

import numpy as np

from macd.audio.ring import RingBuffer, ring_frames


def test_append_and_read_since():
    r = RingBuffer(10)
    r.append(np.arange(4, dtype=np.float32))
    data, nxt = r.read_since(0)
    assert data.tolist() == [0, 1, 2, 3] and nxt == 4
    r.append(np.arange(4, 8, dtype=np.float32))
    data, nxt = r.read_since(nxt)
    assert data.tolist() == [4, 5, 6, 7] and nxt == 8
    assert r.read_since(8)[0].size == 0


def test_wraparound_and_last():
    r = RingBuffer(5)
    for i in range(3):
        r.append(np.arange(i * 3, i * 3 + 3, dtype=np.float32))
    assert r.written == 9
    assert r.slice(4, 9).tolist() == [4, 5, 6, 7, 8]
    assert r.slice(0, 9).tolist() == [4, 5, 6, 7, 8]  # clamped to the retained window
    assert r.slice(6, 8).tolist() == [6, 7]
    assert r.oldest == 4


def test_overrun_counted():
    r = RingBuffer(4)
    r.append(np.arange(10, dtype=np.float32))
    data, nxt = r.read_since(0)
    assert data.tolist() == [6, 7, 8, 9] and nxt == 10
    assert r.overruns == 1 and r.lost_samples == 6


def test_append_larger_than_capacity():
    r = RingBuffer(3)
    r.append(np.arange(2, dtype=np.float32))
    r.append(np.arange(10, 17, dtype=np.float32))
    assert r.written == 9
    assert r.slice(6, 9).tolist() == [14, 15, 16]


async def test_wakeup_from_foreign_thread_coalesced():
    r = RingBuffer(16000)
    r.bind()
    calls = 0
    loop = asyncio.get_running_loop()
    orig = loop.call_soon_threadsafe

    def counting(cb, *a):
        nonlocal calls
        calls += 1
        return orig(cb, *a)

    loop.call_soon_threadsafe = counting  # type: ignore[method-assign]
    try:
        def producer():
            for _ in range(100):
                r.append(np.ones(160, dtype=np.float32))

        t = threading.Thread(target=producer)
        t.start()
        t.join()
        assert await r.wait(0, timeout=1.0)
        assert calls == 1  # 100 appends -> one loop wakeup
        data, nxt = r.read_since(0)
        assert data.size == 16000 and nxt == 16000
    finally:
        loop.call_soon_threadsafe = orig  # type: ignore[method-assign]


async def test_ring_frames_chunks_and_ends_on_close():
    r = RingBuffer(4000)
    r.bind()
    got = []

    async def consume():
        async for f in ring_frames(r, 320):
            got.append(f)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    threading.Thread(target=lambda: [r.append(np.full(100, i, np.float32)) for i in range(10)]).start()
    for _ in range(100):
        if len(got) >= 3:
            break
        await asyncio.sleep(0.01)
    r.close()
    await asyncio.wait_for(task, 1)
    assert len(got) == 3 and all(f.shape == (320,) for f in got)
    assert got[0][0] == 0 and got[0][-1] == 3
