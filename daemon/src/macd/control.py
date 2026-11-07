"""ControlServer: NDJSON over a unix socket (ARCHITECTURE §11) + the command vocabulary.

Security: socket directory 0700, socket 0600 (bound under umask 077), peer uid checked with
getpeereid on every connection (macd.unixsock).

Protocol: one JSON object per line, max 64 KB per line.
  client → daemon: {"id": <any>, "cmd": "<name>", ...args}
      status | ask{text} | confirm{approved} | listen | stop | mute{value} | say{text} | reset | off | on |
      remote_link | remote_end | remote_enable | audio_reinit | voice_clone{path} | voice_reset |
      power{event: will_sleep|did_wake|screen_locked|screen_unlocked} |
      settings_get | settings_set{changes} | voices_list | voice_preview{voice, text?} | connect_check{deep?} | power{event: will_sleep|did_wake} | subscribe
  daemon → client: {"id": <same>, "ok": true, "data": {...}} | {"id", "ok": false, "error": "..."}
  events (after `subscribe`): {"event": "state"|"transcript"|"activity"|"error"|"remote"|"notify", ...}
Each client has a bounded outgoing queue; a client that can't keep up is disconnected.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from macd.orchestrator import Orchestrator
from macd.unixsock import bind_private_unix_socket, peer_uid

log = logging.getLogger(__name__)

MAX_LINE = 64 * 1024
MAX_VOICE_SAMPLE_BYTES = 50 * 1024 * 1024
VOICE_SAMPLE_EXTS = (".wav", ".m4a", ".mp3", ".flac", ".aiff", ".aif", ".caf")


# ----------------------------------------------------------------------------- commands


@dataclass
class Hooks:
    """Daemon-level operations the orchestrator doesn't own (wired in __main__)."""

    remote_link: Callable[[], Awaitable[dict[str, Any]]] | None = None
    remote_end: Callable[[], Awaitable[None]] | None = None
    remote_enable: Callable[[], Awaitable[None]] | None = None
    audio_reinit: Callable[[], Awaitable[None]] | None = None
    power: Callable[[str], Awaitable[None]] | None = None
    status: Callable[[], dict[str, Any]] | None = None
    voice_clone: Callable[[str], Awaitable[dict[str, Any]]] | None = None
    voice_reset: Callable[[], Awaitable[dict[str, Any]]] | None = None
    settings: Any | None = None  # macd.settings.Settings
    connect_check: Callable[[bool], Awaitable[dict[str, Any]]] | None = None


def _text_arg(args: dict[str, Any], cmd: str) -> str:
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{cmd} needs a non-empty 'text'")
    return text


def validate_voice_sample(path: Any) -> str:
    """A readable audio file under 50 MB; returns its absolute path."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("voice_clone needs a 'path' to a short recording (wav/m4a)")
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    if p.suffix.lower() not in VOICE_SAMPLE_EXTS:
        raise ValueError(f"unsupported audio format {p.suffix!r}; use " + ", ".join(VOICE_SAMPLE_EXTS))
    if not os.access(p, os.R_OK):
        raise ValueError(f"cannot read {p}")
    size = p.stat().st_size
    if size == 0 or size > MAX_VOICE_SAMPLE_BYTES:
        raise ValueError(f"{p.name} is {size / 1e6:.1f} MB; the sample must be non-empty and under 50 MB")
    return str(p.resolve())


async def handle_command(orch: Orchestrator, hooks: Hooks, cmd: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one control command. Raises ValueError for bad input / unavailable features."""
    if cmd == "status":
        data = orch.status()
        if hooks.status is not None:
            data.update(hooks.status())
        return data
    if cmd == "ask":
        return {"turn_id": await orch.ask(_text_arg(args, cmd))}
    if cmd == "confirm":
        approved = args.get("approved")
        if not isinstance(approved, bool):
            raise ValueError("confirm needs a boolean 'approved'")
        return {"turn_id": orch.confirm(approved), "approved": approved}
    if cmd == "say":
        orch.say(_text_arg(args, cmd))
        return {}
    if cmd == "listen":
        src = orch.listen()
        if src is None:
            raise ValueError("no listener available (muted, off, or no audio input)")
        return {"source": src}
    if cmd == "stop":
        return {"stopped_turn": await orch.stop()}
    if cmd == "mute":
        value = args.get("value", True)
        if not isinstance(value, bool):
            raise ValueError("mute needs a boolean 'value'")
        orch.set_muted(value)
        return {"muted": orch.muted}
    if cmd in ("off", "on"):
        orch.set_off(cmd == "off")
        return {"off": orch.off}
    if cmd == "reset":
        await orch.reset()
        return {}
    if cmd == "power":
        event = args.get("event")
        if event not in ("will_sleep", "did_wake", "screen_locked", "screen_unlocked"):
            raise ValueError("power needs event will_sleep|did_wake|screen_locked|screen_unlocked")
        await orch.power(event)
        if hooks.power is not None:
            await hooks.power(event)
        return {}
    if cmd == "remote_link":
        if hooks.remote_link is None:
            raise ValueError("remote link is not available")
        return await hooks.remote_link()
    if cmd == "remote_enable":
        if hooks.remote_enable is None:
            raise ValueError("remote access cannot be enabled here")
        await hooks.remote_enable()
        return {"enabled": True}
    if cmd == "remote_end":
        if hooks.remote_end is not None:
            await hooks.remote_end()
        return {}
    if cmd == "connect_check":
        if hooks.connect_check is None:
            raise ValueError("connect check is not available")
        return await hooks.connect_check(bool(args.get("deep", False)))
    if cmd in ("settings_get", "settings_set", "voices_list", "voice_preview"):
        if hooks.settings is None:
            raise ValueError("settings are not available")
        if cmd == "settings_get":
            return hooks.settings.get()
        if cmd == "settings_set":
            return await hooks.settings.set(args.get("changes"))
        if cmd == "voices_list":
            return {"voices": await hooks.settings.voices()}
        return await hooks.settings.preview(args.get("voice"), args.get("text"))
    if cmd == "voice_clone":
        if hooks.voice_clone is None:
            raise ValueError("voice cloning is not available (no speech stack)")
        return await hooks.voice_clone(validate_voice_sample(args.get("path")))
    if cmd == "voice_reset":
        if hooks.voice_reset is None:
            raise ValueError("voice cloning is not available (no speech stack)")
        return await hooks.voice_reset()
    if cmd == "audio_reinit":
        if hooks.audio_reinit is None:
            raise ValueError("audio reinit is not available")
        await hooks.audio_reinit()
        return {}
    raise ValueError(f"unknown command {cmd!r}")


Handler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


# ----------------------------------------------------------------------------- client


class _Client:
    def __init__(self, cid: int, writer: asyncio.StreamWriter, qsize: int) -> None:
        self.cid = cid
        self.writer = writer
        self.out: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=qsize)
        self.subscribed = False
        self.closed = False

    def send(self, obj: dict[str, Any]) -> bool:
        """Non-blocking enqueue. Returns False (and closes the client) if its queue is full."""
        if self.closed:
            return False
        line = (json.dumps(obj, default=repr, ensure_ascii=False) + "\n").encode()
        try:
            self.out.put_nowait(line)
            return True
        except asyncio.QueueFull:
            self.kill()
            return False

    def kill(self) -> None:
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(asyncio.QueueEmpty):
            while True:
                self.out.get_nowait()
        self.out.put_nowait(None)
        if self.writer.transport is not None:
            self.writer.transport.abort()


# ----------------------------------------------------------------------------- server


class ControlServer:
    def __init__(
        self,
        handler: Handler,
        sock_path: Path | str,
        *,
        max_clients: int = 32,
        client_queue: int = 256,
        command_timeout_s: float = 90.0,
    ) -> None:
        self.handler = handler
        self.sock_path = Path(sock_path)
        self.max_clients = max_clients
        self.client_queue = client_queue
        self.command_timeout_s = command_timeout_s
        self._server: asyncio.base_events.Server | None = None
        self._clients: dict[int, _Client] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._next_id = 0
        self.rejected_peers = 0
        self.dropped_clients = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        sock = bind_private_unix_socket(self.sock_path)
        self._server = await asyncio.start_unix_server(self._on_client, sock=sock, limit=MAX_LINE)
        log.info("control socket listening at %s", self.sock_path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for c in list(self._clients.values()):
            c.kill()
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 1.0)
            self._server = None
        self.sock_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ events

    def publish(self, event: dict[str, Any]) -> None:
        """Broadcast an event to every subscribed client. Never blocks."""
        for c in list(self._clients.values()):
            if c.subscribed and not c.closed and not c.send(event):
                self.dropped_clients += 1
                log.warning("control client %d too slow; disconnected", c.cid)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def subscriber_count(self) -> int:
        return sum(1 for c in self._clients.values() if c.subscribed and not c.closed)

    # ------------------------------------------------------------------ connections

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            sock = writer.get_extra_info("socket")
            try:
                uid = peer_uid(sock) if sock is not None else None
            except OSError:
                uid = None
            if uid != os.getuid():
                self.rejected_peers += 1
                log.warning("refusing control connection from uid %s", uid)
                writer.close()
                return
            if len(self._clients) >= self.max_clients:
                writer.write(b'{"id":null,"ok":false,"error":"too many clients"}\n')
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(writer.drain(), 1.0)
                writer.close()
                return
            self._next_id += 1
            client = _Client(self._next_id, writer, self.client_queue)
            self._clients[client.cid] = client
            wtask = asyncio.create_task(self._writer_loop(client), name=f"control-writer:{client.cid}")
            self._tasks.add(wtask)
            wtask.add_done_callback(self._tasks.discard)
            try:
                await self._reader_loop(client, reader)
            finally:
                self._clients.pop(client.cid, None)
                if not client.closed:
                    client.closed = True
                    with contextlib.suppress(asyncio.QueueFull):
                        client.out.put_nowait(None)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(wtask, 2.0)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("control client failed")
        finally:
            with contextlib.suppress(Exception):
                writer.close()
            if task is not None:
                self._tasks.discard(task)

    async def _writer_loop(self, c: _Client) -> None:
        try:
            while True:
                line = await c.out.get()
                if line is None:
                    break
                c.writer.write(line)
                await c.writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                c.writer.close()

    async def _reader_loop(self, c: _Client, reader: asyncio.StreamReader) -> None:
        while not c.closed:
            try:
                raw = await reader.readline()
            except (ValueError, asyncio.LimitOverrunError):
                c.send({"id": None, "ok": False, "error": f"line too long (max {MAX_LINE} bytes)"})
                await asyncio.sleep(0)  # let the writer flush the error
                break
            except (ConnectionError, OSError):
                break
            if not raw:
                break
            line = raw.strip()
            if line:
                await self._handle_line(c, line)

    async def _handle_line(self, c: _Client, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            c.send({"id": None, "ok": False, "error": f"malformed JSON: {e}"})
            return
        if not isinstance(msg, dict):
            c.send({"id": None, "ok": False, "error": "expected a JSON object"})
            return
        mid = msg.get("id")
        cmd = msg.get("cmd")
        if not isinstance(cmd, str):
            c.send({"id": mid, "ok": False, "error": "missing 'cmd'"})
            return
        if cmd == "subscribe":
            c.subscribed = True
            c.send({"id": mid, "ok": True, "data": {"subscribed": True}})
            return
        args = {k: v for k, v in msg.items() if k not in ("id", "cmd")}
        try:
            data = await asyncio.wait_for(self.handler(cmd, args), self.command_timeout_s)
            c.send({"id": mid, "ok": True, "data": data or {}})
        except TimeoutError:
            c.send({"id": mid, "ok": False, "error": f"{cmd} timed out"})
        except (ValueError, KeyError, TypeError) as e:
            c.send({"id": mid, "ok": False, "error": str(e)})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("control command %s failed", cmd)
            c.send({"id": mid, "ok": False, "error": f"internal error: {e}"})
