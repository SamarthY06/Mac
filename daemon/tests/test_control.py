from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from macd import control
from macd.control import MAX_LINE, ControlServer


def short_tmp() -> Path:
    # AF_UNIX paths are limited to 104 bytes on macOS; pytest's tmp_path is often longer.
    return Path(tempfile.mkdtemp(prefix="mc", dir="/tmp"))


class FakeHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((cmd, args))
        if cmd == "status":
            return {"state": "idle"}
        if cmd == "ask":
            if not args.get("text"):
                raise ValueError("ask needs a non-empty 'text'")
            return {"turn_id": 7}
        if cmd == "reset":
            raise RuntimeError("kaboom")
        if cmd not in ("status", "ask", "mute", "off", "say", "stop", "listen"):
            raise ValueError(f"unknown command {cmd!r}")
        return {}


@pytest.fixture
async def server() -> AsyncIterator[tuple[ControlServer, FakeHandler, Path]]:
    d = short_tmp()
    sock = d / "run" / "control.sock"
    h = FakeHandler()
    srv = ControlServer(h, sock, client_queue=8)
    await srv.start()
    try:
        yield srv, h, sock
    finally:
        await srv.close()
        shutil.rmtree(d, ignore_errors=True)


async def connect(sock: Path) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    return await asyncio.open_unix_connection(str(sock), limit=1 << 20)


async def rpc(r: asyncio.StreamReader, w: asyncio.StreamWriter, msg: dict[str, Any] | bytes) -> dict[str, Any]:
    w.write(msg if isinstance(msg, bytes) else (json.dumps(msg) + "\n").encode())
    await w.drain()
    line = await asyncio.wait_for(r.readline(), 2)
    return json.loads(line)


async def test_socket_permissions(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    _srv, _h, sock = server
    assert stat.S_IMODE(os.stat(sock).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(sock.parent).st_mode) == 0o700


async def test_commands(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    _srv, h, sock = server
    r, w = await connect(sock)
    assert await rpc(r, w, {"id": 1, "cmd": "status"}) == {"id": 1, "ok": True, "data": {"state": "idle"}}
    assert await rpc(r, w, {"id": "a", "cmd": "ask", "text": "hi"}) == {"id": "a", "ok": True, "data": {"turn_id": 7}}
    bad = await rpc(r, w, {"id": 3, "cmd": "ask"})
    assert bad["ok"] is False and "text" in bad["error"]
    unk = await rpc(r, w, {"id": 4, "cmd": "explode"})
    assert unk == {"id": 4, "ok": False, "error": "unknown command 'explode'"}
    assert (await rpc(r, w, {"id": 9, "cmd": 7}))["ok"] is False
    internal = await rpc(r, w, {"id": 5, "cmd": "reset"})
    assert internal["ok"] is False and "internal error" in internal["error"]
    assert (await rpc(r, w, {"id": 8, "cmd": "off"}))["ok"] and h.calls[-1] == ("off", {})
    ok = await rpc(r, w, {"id": 6, "cmd": "mute", "value": True})
    assert ok["ok"] and h.calls[-1] == ("mute", {"value": True})
    w.close()


async def test_malformed_json_keeps_connection(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    _srv, _h, sock = server
    r, w = await connect(sock)
    err = await rpc(r, w, b"{not json\n")
    assert err["ok"] is False and "malformed JSON" in err["error"]
    err2 = await rpc(r, w, b"[1,2,3]\n")
    assert err2["ok"] is False
    assert (await rpc(r, w, {"id": 2, "cmd": "status"}))["ok"] is True
    w.close()


async def test_oversized_line_rejected(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    _srv, _h, sock = server
    r, w = await connect(sock)
    payload = json.dumps({"id": 1, "cmd": "ask", "text": "x" * (MAX_LINE + 10)}).encode() + b"\n"
    w.write(payload)
    with contextlib.suppress(ConnectionError, OSError):
        await w.drain()
    line = await asyncio.wait_for(r.readline(), 2)
    msg = json.loads(line)
    assert msg["ok"] is False and "too long" in msg["error"]
    assert await asyncio.wait_for(r.read(), 2) == b""  # server closed the connection


async def test_subscribe_event_stream(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    srv, _h, sock = server
    r, w = await connect(sock)
    other_r, other_w = await connect(sock)
    assert (await rpc(r, w, {"id": 1, "cmd": "subscribe"}))["data"] == {"subscribed": True}
    await rpc(other_r, other_w, {"id": 1, "cmd": "status"})  # connected, not subscribed
    srv.publish({"event": "state", "state": "thinking", "turn_id": 1})
    srv.publish({"event": "transcript", "role": "user", "text": "hi"})
    got = [json.loads(await asyncio.wait_for(r.readline(), 2)) for _ in range(2)]
    assert [g["event"] for g in got] == ["state", "transcript"]
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(other_r.readline(), 0.1)
    w.close()
    other_w.close()


async def test_slow_client_dropped(server: tuple[ControlServer, FakeHandler, Path]) -> None:
    srv, _h, sock = server
    r, w = await connect(sock)
    await rpc(r, w, {"id": 1, "cmd": "subscribe"})
    big = "y" * 60_000
    for i in range(200):  # never read → per-client queue (8) overflows
        srv.publish({"event": "activity", "i": i, "blob": big})
    assert srv.dropped_clients >= 1
    await asyncio.sleep(0.05)
    assert srv.client_count == 0
    # a fresh client still works
    r2, w2 = await connect(sock)
    assert (await rpc(r2, w2, {"id": 1, "cmd": "status"}))["ok"]
    w2.close()
    w.close()


async def test_foreign_uid_refused(monkeypatch: pytest.MonkeyPatch, server: tuple[ControlServer, FakeHandler, Path]) -> None:
    srv, h, sock = server
    monkeypatch.setattr(control, "peer_uid", lambda s: os.getuid() + 1)
    r, w = await connect(sock)
    w.write(b'{"id":1,"cmd":"status"}\n')
    await w.drain()
    assert await asyncio.wait_for(r.read(), 2) == b""
    assert srv.rejected_peers == 1 and h.calls == []


def test_peer_uid_real_socketpair() -> None:
    import socket

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert control.peer_uid(a) == os.getuid()
    finally:
        a.close()
        b.close()


async def test_handle_command_dispatch() -> None:
    from macd.control import Hooks, handle_command

    class Orch:
        muted = False
        off = False
        calls: ClassVar[list[str]] = []

        def status(self) -> dict[str, Any]:
            return {"state": "idle"}

        def set_muted(self, v: bool) -> None:
            self.muted = v

        def set_off(self, v: bool) -> None:
            self.off = v

        def listen(self) -> str | None:
            return None

        def say(self, text: str) -> None:
            return

        def confirm(self, approved: bool) -> int:
            if not approved:
                raise ValueError("nothing pending")
            return 4

        async def power(self, event: str) -> None:
            self.calls.append(event)

    orch = Orch()
    powered: list[str] = []
    hooks = Hooks(status=lambda: {"audio": {"backend": "x"}}, remote_end=None)

    async def power(ev: str) -> None:
        powered.append(ev)

    hooks.power = power
    assert await handle_command(orch, hooks, "status", {}) == {"state": "idle", "audio": {"backend": "x"}}  # type: ignore[arg-type]
    assert await handle_command(orch, hooks, "mute", {"value": True}) == {"muted": True}  # type: ignore[arg-type]
    assert await handle_command(orch, hooks, "off", {}) == {"off": True}  # type: ignore[arg-type]
    assert await handle_command(orch, hooks, "power", {"event": "did_wake"}) == {}  # type: ignore[arg-type]
    assert orch.calls == ["did_wake"] and powered == ["did_wake"]
    assert await handle_command(orch, hooks, "remote_end", {}) == {}  # type: ignore[arg-type]
    assert await handle_command(orch, hooks, "confirm", {"approved": True}) == {"turn_id": 4, "approved": True}  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nothing pending"):
        await handle_command(orch, hooks, "confirm", {"approved": False})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        await handle_command(orch, hooks, "confirm", {"approved": "yes"})  # type: ignore[arg-type]
    for cmd, args in (("listen", {}), ("mute", {"value": "yes"}), ("power", {"event": "x"}), ("remote_link", {}),
                      ("audio_reinit", {}), ("say", {}), ("nope", {})):
        with pytest.raises(ValueError):
            await handle_command(orch, hooks, cmd, args)  # type: ignore[arg-type]


def test_validate_voice_sample(tmp_path: Path) -> None:
    from macd.control import MAX_VOICE_SAMPLE_BYTES, validate_voice_sample

    wav = tmp_path / "me.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    assert validate_voice_sample(str(wav)) == str(wav.resolve())
    for bad, msg in ((None, "needs a 'path'"), ("", "needs a 'path'"), (str(tmp_path / "nope.wav"), "no such file"),
                     (str(tmp_path), "no such file")):
        with pytest.raises(ValueError, match=msg):
            validate_voice_sample(bad)
    ogg = tmp_path / "me.ogg"
    ogg.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported audio format"):
        validate_voice_sample(str(ogg))
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="non-empty"):
        validate_voice_sample(str(empty))
    big = tmp_path / "big.m4a"
    with open(big, "wb") as fh:
        fh.truncate(MAX_VOICE_SAMPLE_BYTES + 1)
    with pytest.raises(ValueError, match="under 50 MB"):
        validate_voice_sample(str(big))


async def test_voice_commands_dispatch(tmp_path: Path) -> None:
    from macd.control import Hooks, handle_command

    wav = tmp_path / "me.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    calls: list[str | None] = []

    async def clone(path: str) -> dict[str, Any]:
        calls.append(path)
        return {"engine": "clone"}

    async def reset() -> dict[str, Any]:
        calls.append(None)
        return {"engine": "kokoro"}

    hooks = Hooks(voice_clone=clone, voice_reset=reset)
    assert await handle_command(None, hooks, "voice_clone", {"path": str(wav)}) == {"engine": "clone"}  # type: ignore[arg-type]
    assert await handle_command(None, hooks, "voice_reset", {}) == {"engine": "kokoro"}  # type: ignore[arg-type]
    assert calls == [str(wav.resolve()), None]
    with pytest.raises(ValueError, match="no such file"):
        await handle_command(None, hooks, "voice_clone", {"path": "/nope.wav"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not available"):
        await handle_command(None, Hooks(), "voice_reset", {})  # type: ignore[arg-type]


async def test_settings_commands_dispatch() -> None:
    from macd.control import Hooks, handle_command

    class FakeSettings:
        def get(self) -> dict[str, Any]:
            return {"name": "Mac"}

        async def set(self, changes: Any) -> dict[str, Any]:
            return {"applied": dict(changes), "errors": {}}

        async def voices(self) -> list[dict[str, Any]]:
            return [{"id": "af_heart"}]

        async def preview(self, voice: Any, text: Any = None) -> dict[str, Any]:
            return {"queued": True, "voice": voice, "text": text}

    hooks = Hooks(settings=FakeSettings())
    assert await handle_command(None, hooks, "settings_get", {}) == {"name": "Mac"}  # type: ignore[arg-type]
    assert (await handle_command(None, hooks, "settings_set", {"changes": {"name": "J"}}))["applied"] == {"name": "J"}  # type: ignore[arg-type]
    assert await handle_command(None, hooks, "voices_list", {}) == {"voices": [{"id": "af_heart"}]}  # type: ignore[arg-type]
    assert (await handle_command(None, hooks, "voice_preview", {"voice": "af_heart"}))["voice"] == "af_heart"  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not available"):
        await handle_command(None, Hooks(), "settings_get", {})  # type: ignore[arg-type]


async def test_stale_socket_replaced() -> None:
    import socket

    d = short_tmp()
    try:
        sock = d / "control.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(sock))
        s.close()  # leaves a stale socket file behind
        srv = ControlServer(FakeHandler(), sock)
        await srv.start()
        r, w = await connect(sock)
        assert (await rpc(r, w, {"id": 1, "cmd": "status"}))["ok"]
        w.close()
        await srv.close()
        assert not sock.exists()
    finally:
        shutil.rmtree(d, ignore_errors=True)
