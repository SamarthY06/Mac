"""audio.sock server: macd <-> Mac.app (Swift) audio link (ARCHITECTURE §2).

macd is the server; exactly one client (the Swift app) at a time — a new connection
replaces the old one and is a new session (pending plays of the old one fail).
Client facts this relies on: HELLO is sent on connect and again whenever the input
device / AEC state changes (each one updates ``aec``); DRAINED arrives exactly once per
stream — after END, after a FLUSH, or when the engine is rebuilt (sleep/device change) —
so every play() resolves on whichever comes first. Stream ids are a monotonic counter
and are never reused.

Everything here runs on the asyncio loop (no threads), so the pending-stream table and
the client reference need no locking.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from macd import config
from macd.audio import protocol as P
from macd.audio.ring import RingBuffer, ring_frames
from macd.contracts import FRAME_SAMPLES, SAMPLE_RATE
from macd.unixsock import bind_private_unix_socket, peer_uid

log = logging.getLogger(__name__)

class _Client:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.hello: dict[str, Any] = {}
        self.closed = asyncio.Event()

    def send(self, data: bytes) -> None:
        if not self.writer.is_closing():
            self.writer.write(data)


class SwiftInput:
    """AudioInput fed by MIC frames from the Swift app."""

    def __init__(self, link: SwiftAudioLink) -> None:
        self.name = "local"
        self.requires_wake_word = True
        self._link = link
        self.ring = RingBuffer(SAMPLE_RATE * 10)

    async def start(self) -> None:
        await self._link.start()

    async def close(self) -> None:
        self.ring.close()

    def frames(self) -> AsyncIterator[np.ndarray]:
        return ring_frames(self.ring, FRAME_SAMPLES)


class SwiftOutput:
    """AudioOutput that streams PLAY frames to the Swift app and awaits DRAINED."""

    CHUNK_S = 0.1

    def __init__(self, link: SwiftAudioLink) -> None:
        self.name = "swift"
        self._link = link

    @property
    def aec(self) -> bool:
        return self._link.aec

    async def start(self) -> None:
        await self._link.start()

    async def close(self) -> None:
        self.flush()

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        await self._link.play(pcm, sample_rate)

    def flush(self) -> None:
        self._link.flush()


class SwiftAudioLink:
    """Unix-socket server for the Swift audio HAL. Use ``.input`` and ``.output``."""

    def __init__(self, sock_path: Path | str = config.AUDIO_SOCK, connect_grace_s: float = 1.0, drain_slack_s: float = 5.0) -> None:
        self.sock_path = Path(sock_path)
        self.connect_grace_s = connect_grace_s
        self.drain_slack_s = drain_slack_s
        self.connected = asyncio.Event()
        self.input = SwiftInput(self)
        self.output = SwiftOutput(self)
        self._server: asyncio.base_events.Server | None = None
        self._client: _Client | None = None
        self._pending: dict[int, asyncio.Future[None]] = {}
        self._ids = itertools.count(1)
        self._start_lock = asyncio.Lock()
        self.rejected_peers = 0
        self.protocol_errors = 0

    @property
    def aec(self) -> bool:
        """AEC state from the client's latest HELLO (False when disconnected)."""
        return bool(self._client and self._client.hello.get("aec"))

    @property
    def hello(self) -> dict[str, Any]:
        return dict(self._client.hello) if self._client else {}

    async def start(self) -> None:
        async with self._start_lock:
            if self._server is not None:
                return
            sock = bind_private_unix_socket(self.sock_path, backlog=4)
            self.input.ring.bind()
            self._server = await asyncio.start_unix_server(self._on_connect, sock=sock)
            log.info("audio link listening on %s", self.sock_path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        if self._client is not None:
            await self._drop_client(self._client, "link closed")
        self.input.ring.close()
        with contextlib.suppress(FileNotFoundError):
            self.sock_path.unlink()

    # ------------------------------------------------------------------ connections

    async def _on_connect(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sock: socket.socket = writer.get_extra_info("socket")
        try:
            uid = peer_uid(sock)
        except OSError:
            log.exception("getpeereid failed; rejecting audio client")
            writer.close()
            return
        if uid != os.getuid():
            self.rejected_peers += 1
            log.warning("rejecting audio client with uid %d", uid)
            writer.close()
            return
        client = _Client(reader, writer)
        old, self._client = self._client, client
        if old is not None:
            log.info("new audio client replaces the previous one")
            await self._drop_client(old, "replaced by a new client")
        client.send(P.encode_hello({"role": "macd", "sample_rate": SAMPLE_RATE}))
        self.connected.set()
        log.info("audio client connected")
        try:
            await self._read_loop(client)
        finally:
            if self._client is client:
                await self._drop_client(client, "client disconnected")

    async def _drop_client(self, client: _Client, why: str) -> None:
        if self._client is client:
            self._client = None
            self.connected.clear()
            err = ConnectionError(f"audio client gone: {why}")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(err)
            self._pending.clear()
        client.closed.set()
        client.writer.close()
        with contextlib.suppress(Exception):
            await client.writer.wait_closed()

    async def _read_loop(self, client: _Client) -> None:
        dec = P.FrameDecoder()
        ring = self.input.ring
        while self._client is client:
            data = await client.reader.read(65536)
            if not data:
                return
            try:
                frames = dec.feed(data)
            except P.ProtocolError:
                self.protocol_errors += 1
                log.exception("audio protocol error; dropping client")
                return
            for fr in frames:
                try:
                    self._handle(client, fr, ring)
                except P.ProtocolError:
                    self.protocol_errors += 1
                    log.warning("bad %s frame ignored", fr.type.name, exc_info=True)

    def _handle(self, client: _Client, fr: P.Frame, ring: RingBuffer) -> None:
        if fr.type is P.FrameType.MIC:
            ring.append(P.bytes_to_pcm(fr.payload))
        elif fr.type is P.FrameType.DRAINED:
            fut = self._pending.pop(P.decode_stream_id(fr.payload), None)
            if fut is not None and not fut.done():
                fut.set_result(None)
        elif fr.type is P.FrameType.HELLO:
            client.hello = P.decode_hello(fr.payload)
            log.info("audio client hello: %s", client.hello)
        else:
            log.warning("unexpected %s frame from client", fr.type.name)

    # ------------------------------------------------------------------ playback

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Stream ``pcm`` to the client; return once it reports DRAINED for it."""
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        if self._client is None:
            try:
                await asyncio.wait_for(self.connected.wait(), self.connect_grace_s)
            except TimeoutError:
                log.warning("no audio client; dropping %.2fs of playback", x.size / sample_rate)
                return
        client = self._client
        if client is None:
            return
        sid = next(self._ids)  # monotonic, never 0, never reused
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending[sid] = fut
        step = max(1, int(sample_rate * SwiftOutput.CHUNK_S))
        try:
            for i in range(0, x.size, step):
                client.send(P.encode_play(sid, sample_rate, x[i : i + step]))
            client.send(P.encode_end(sid))
            await client.writer.drain()
            timeout = x.size / sample_rate + self.drain_slack_s
            try:
                await asyncio.wait_for(asyncio.shield(fut), timeout)
            except TimeoutError:
                log.warning("no DRAINED for stream %d after %.1fs; continuing", sid, timeout)
        except asyncio.CancelledError:
            if self._client is client:
                client.send(P.encode_flush(sid))
            raise
        except (ConnectionResetError, BrokenPipeError) as e:
            raise ConnectionError(f"audio client write failed: {e}") from e
        finally:
            self._pending.pop(sid, None)

    def flush(self) -> None:
        """Stop all playback now (FLUSH 0) and release every waiting play()."""
        client = self._client
        if client is not None:
            client.send(P.encode_flush(0))
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(None)
        self._pending.clear()
