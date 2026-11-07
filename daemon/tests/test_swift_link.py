"""SwiftAudioLink against a tiny pure-python client playing the Swift app's role."""

import asyncio
import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path

import numpy as np
import pytest

from macd.audio import protocol as P
from macd.audio.swift_link import SwiftAudioLink
from macd.unixsock import peer_uid


class FakeSwiftClient:
    """Plays the Swift app: sends HELLO/MIC, records PLAY/END/FLUSH, answers DRAINED."""

    def __init__(self, path: Path, aec: bool = True, auto_drain: bool = True) -> None:
        self.path, self.aec, self.auto_drain = path, aec, auto_drain
        self.frames: list[P.Frame] = []
        self.hello: dict | None = None
        self.got_frame = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def connect(self) -> None:
        self.reader, self.writer = await asyncio.open_unix_connection(str(self.path))
        self.writer.write(P.encode_hello({"aec": self.aec, "input_device": "Fake Mic"}))
        await self.writer.drain()
        self._task = asyncio.create_task(self._read())

    async def _read(self) -> None:
        dec = P.FrameDecoder()
        while data := await self.reader.read(65536):
            for fr in dec.feed(data):
                if fr.type is P.FrameType.HELLO:
                    self.hello = P.decode_hello(fr.payload)
                    continue
                self.frames.append(fr)
                self.got_frame.set()
                if fr.type is P.FrameType.END and self.auto_drain:
                    self.writer.write(P.encode_drained(P.decode_stream_id(fr.payload)))
        self.eof = True

    def send_mic(self, pcm: np.ndarray) -> None:
        self.writer.write(P.encode_mic(pcm))

    def of(self, t: P.FrameType) -> list[P.Frame]:
        return [f for f in self.frames if f.type is t]

    async def close(self) -> None:
        self.writer.close()
        if self._task:
            await asyncio.wait_for(self._task, 2)


@pytest.fixture
async def link():
    d = tempfile.mkdtemp(prefix="macd-", dir="/tmp")  # short path: AF_UNIX limit is 104 bytes
    lk = SwiftAudioLink(Path(d) / "audio.sock", connect_grace_s=0.2, drain_slack_s=1.0)
    await lk.start()
    yield lk
    await lk.close()
    shutil.rmtree(d, ignore_errors=True)


async def _connected(lk: SwiftAudioLink, aec: bool = True, **kw) -> FakeSwiftClient:
    c = FakeSwiftClient(lk.sock_path, aec=aec, **kw)
    await c.connect()
    await asyncio.wait_for(lk.connected.wait(), 2)
    for _ in range(100):
        if lk.hello and c.hello:
            break
        await asyncio.sleep(0.01)
    return c


async def test_socket_permissions_and_hello(link):
    mode = stat.S_IMODE(os.stat(link.sock_path).st_mode)
    assert mode == 0o600
    assert link.output.aec is False
    c = await _connected(link, aec=True)
    assert link.output.aec is True
    assert c.hello and c.hello["version"] == P.PROTOCOL_VERSION
    await c.close()


async def test_mic_frames_reframed(link):
    it = link.input.frames().__aiter__()
    c = await _connected(link)
    pcm = (np.arange(1000) % 200 / 400).astype(np.float32)
    c.send_mic(pcm[:150])
    c.send_mic(pcm[150:])
    await c.writer.drain()
    f1 = await asyncio.wait_for(it.__anext__(), 2)
    f2 = await asyncio.wait_for(it.__anext__(), 2)
    assert f1.shape == (320,) and f2.shape == (320,)
    assert np.allclose(np.concatenate([f1, f2]), pcm[:640], atol=1e-4)
    await c.close()


async def test_play_chunks_end_and_drained(link):
    c = await _connected(link)
    pcm = np.sin(np.linspace(0, 100, 24000 * 35 // 100)).astype(np.float32) * 0.3  # 350 ms @ 24k
    await asyncio.wait_for(link.output.play(pcm, 24000), 2)
    plays = c.of(P.FrameType.PLAY)
    assert len(plays) == 4  # <= 100 ms chunks
    decoded = [P.decode_play(f.payload) for f in plays]
    sids = {sid for sid, _, _ in decoded}
    assert len(sids) == 1 and 0 not in sids
    assert all(sr == 24000 and len(x) <= 2400 for _, sr, x in decoded)
    assert np.allclose(np.concatenate([x for _, _, x in decoded]), pcm, atol=1e-4)
    ends = c.of(P.FrameType.END)
    assert [P.decode_stream_id(e.payload) for e in ends] == list(sids)
    # second play uses a fresh stream id
    await asyncio.wait_for(link.output.play(pcm[:100], 24000), 2)
    assert P.decode_stream_id(c.of(P.FrameType.END)[-1].payload) not in sids
    await c.close()


async def test_disconnect_fails_fast(link):
    c = await _connected(link, auto_drain=False)
    task = asyncio.create_task(link.output.play(np.zeros(24000, np.float32), 24000))
    await asyncio.wait_for(c.got_frame.wait(), 2)
    await c.close()
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(task, 2)
    assert not link.connected.is_set()


async def test_no_client_completes_after_grace(link):
    await asyncio.wait_for(link.output.play(np.zeros(16000, np.float32), 16000), 1)


async def test_cancel_sends_flush_for_stream(link):
    c = await _connected(link, auto_drain=False)
    task = asyncio.create_task(link.output.play(np.zeros(24000, np.float32), 24000))
    await asyncio.wait_for(c.got_frame.wait(), 2)
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if c.of(P.FrameType.FLUSH):
            break
        await asyncio.sleep(0.01)
    sid = P.decode_play(c.of(P.FrameType.PLAY)[0].payload)[0]
    assert [P.decode_stream_id(f.payload) for f in c.of(P.FrameType.FLUSH)] == [sid]
    await c.close()


async def test_flush_all_releases_play(link):
    c = await _connected(link, auto_drain=False)
    task = asyncio.create_task(link.output.play(np.zeros(24000, np.float32), 24000))
    await asyncio.wait_for(c.got_frame.wait(), 2)
    link.output.flush()
    await asyncio.wait_for(task, 1)
    for _ in range(100):
        if c.of(P.FrameType.FLUSH):
            break
        await asyncio.sleep(0.01)
    assert P.decode_stream_id(c.of(P.FrameType.FLUSH)[0].payload) == 0
    await c.close()


async def test_new_client_replaces_old(link):
    a = await _connected(link)
    b = FakeSwiftClient(link.sock_path, aec=False)
    await b.connect()
    await asyncio.wait_for(a._task, 2)  # old client sees EOF
    assert getattr(a, "eof", False)
    for _ in range(100):
        if link.hello.get("aec") is False:
            break
        await asyncio.sleep(0.01)
    assert link.output.aec is False and link.connected.is_set()
    await b.close()


def test_peer_uid_socketpair():
    a, b = socket.socketpair(socket.AF_UNIX)
    try:
        assert peer_uid(a) == os.getuid()
    finally:
        a.close()
        b.close()


async def test_stale_socket_replaced_and_non_socket_refused():
    d = Path(tempfile.mkdtemp(prefix="macd-", dir="/tmp"))
    try:
        p = d / "audio.sock"
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(p))
        s.close()  # stale socket file left behind
        lk = SwiftAudioLink(p)
        await lk.start()
        await lk.close()
        p.write_text("not a socket")
        with pytest.raises(FileExistsError):
            await SwiftAudioLink(p).start()
    finally:
        shutil.rmtree(d, ignore_errors=True)
