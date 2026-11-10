"""chatd's connection to agentd (docs/agentd-api.md, kind=chat).

- The token comes from the first line of the lifeline pipe (`{"token": ...}`), never env or disk;
  `dev_token()` reads `dev-tokens.json` for development without Mac.app.
- `run()` connects, says `hello` with this process's `client_epoch`, then calls `on_connect`
  (where the runner subscribes with `after: {boot, seq}`). A lost connection reconnects with
  backoff 0.2, 0.5, 1, 2 ... 10 s and the same epoch (api §8).
- `unauthorized` and `version_mismatch` are fatal: `run()` raises AgentFatal and Mac.app
  restarts the services.
- Events are decoded with the schema and handed to `on_event` in stream order; unknown events
  are skipped (clients ignore what they do not know).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agentd import schema as s

log = logging.getLogger(__name__)

CLIENT_NAME = "chatd/0.9.0"
BACKOFF_S = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0)
CALL_TIMEOUT_S = 10.0
_FATAL = {s.ErrorCode.UNAUTHORIZED, s.ErrorCode.VERSION_MISMATCH}


class AgentError(Exception):
    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class AgentUnavailable(AgentError):
    """agentd is not connected (or the connection dropped mid-call)."""

    def __init__(self, message: str = "Mac's brain is restarting.") -> None:
        super().__init__("unavailable", message)


class AgentFatal(AgentError):
    """unauthorized or version_mismatch: retrying cannot help."""


def read_lifeline_token(line: str | bytes) -> str:
    """The lifeline's first line `{"token": "<64 hex>"}` -> the token."""
    obj = json.loads(line)
    tok = obj.get("token") if isinstance(obj, dict) else None
    if not isinstance(tok, str) or len(tok) < 32:
        raise ValueError('expected {"token": "<hex>"} on the lifeline')
    return tok


def dev_token(run_dir: Path) -> str:
    """Development only: the chat token from `agentd --dev`'s dev-tokens.json."""
    return str(json.loads((run_dir / "dev-tokens.json").read_text())["chat"])


EventCallback = Callable[[s.Event], None]
ConnectCallback = Callable[[s.HelloData], Awaitable[None]]
NoticeCallback = Callable[[Any], None]


class AgentClient:
    def __init__(self, sock: Path, token: str, conversations: list[str], *, on_event: EventCallback,
                 on_connect: ConnectCallback, on_disconnect: Callable[[], None] | None = None,
                 on_notice: NoticeCallback | None = None, client: str = CLIENT_NAME,
                 epoch: str | None = None, backoff: tuple[float, ...] = BACKOFF_S) -> None:
        self.sock = Path(sock)
        self.token = token
        self.conversations = list(conversations)
        self.on_event = on_event
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_notice = on_notice
        self.client = client
        self.epoch = epoch or os.urandom(16).hex()  # one per process; reused on reconnect (V2)
        self.backoff = backoff
        self.connected = asyncio.Event()
        self.hello: s.HelloData | None = None
        self.connects = 0
        self._w: asyncio.StreamWriter | None = None
        self._id = 0
        self._pending: dict[int, tuple[str, asyncio.Future[Any]]] = {}
        self._closed = False

    # -- lifecycle ---------------------------------------------------------------------------------
    async def run(self) -> None:
        attempt = 0
        while not self._closed:
            try:
                await self._session()
                attempt = 0
            except AgentFatal:
                raise
            except (OSError, AgentError, asyncio.IncompleteReadError) as e:
                log.info("agent: not connected (%s)", type(e).__name__)
            except Exception:
                log.exception("agent: connection failed")
            if self._closed:
                return
            delay = self.backoff[min(attempt, len(self.backoff) - 1)]
            attempt += 1
            await asyncio.sleep(delay)

    async def close(self) -> None:
        self._closed = True
        if self._w is not None:
            self._w.close()

    async def _session(self) -> None:
        r, w = await asyncio.open_unix_connection(str(self.sock), limit=s.MAX_LINE_BYTES + 2)
        self._w = w
        reader = asyncio.create_task(self._read(r), name="chatd:agent-read")
        up = False
        try:
            hello = await self.call("hello", api=s.API_VERSION, token=self.token, client=self.client,
                                    client_epoch=self.epoch, conversations=self.conversations, _connecting=True)
            self.hello = hello
            self.connects += 1
            self.connected.set()
            up = True
            log.info("agent: connected (boot %s)", hello.boot)
            await self.on_connect(hello)
            await reader
        finally:
            self.connected.clear()
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            w.close()
            self._w = None
            for _, fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AgentUnavailable())
            self._pending.clear()
            if self.on_disconnect is not None and up:
                self.on_disconnect()

    async def _read(self, r: asyncio.StreamReader) -> None:
        try:
            await self._read_lines(r)
        finally:
            # the socket is gone: every call still waiting (on_connect may be mid-way) fails now,
            # not after CALL_TIMEOUT_S each, and new calls fail at once
            self.connected.clear()
            self._w = None
            for _, fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(AgentUnavailable())
            self._pending.clear()

    async def _read_lines(self, r: asyncio.StreamReader) -> None:
        while line := await r.readline():
            try:
                obj = s.parse_line(line)
            except s.SchemaError:
                log.warning("agent: bad frame from agentd; dropping the connection")
                return
            if s.is_event(obj) and obj.get("ev") in s.NOTICES:  # connection-level, no conv or seq
                try:
                    notice = s.decode_notice(obj)
                except s.SchemaError as e:
                    log.debug("agent: skipped a notice (%s)", e)
                    continue
                if self.on_notice is not None:
                    self.on_notice(notice)
                continue
            if s.is_event(obj):
                try:
                    ev = s.decode_event(obj)
                except s.SchemaError as e:
                    log.debug("agent: skipped an event (%s)", e)
                    continue
                self.on_event(ev)
                continue
            rid = obj.get("id")
            entry = self._pending.pop(rid, None) if isinstance(rid, int) else None
            if entry is None:
                if obj.get("ok") is False:
                    log.warning("agent: %s", obj.get("code"))
                continue
            op, fut = entry
            if fut.done():
                continue
            try:
                resp = s.decode_response(obj, op)
            except s.SchemaError as e:
                fut.set_exception(AgentError("bad_frame", str(e)))
                continue
            if isinstance(resp, s.ErrorResponse):
                cls = AgentFatal if resp.code in _FATAL else AgentError
                fut.set_exception(cls(resp.code.value, resp.error, resp.data))
            else:
                fut.set_result(resp.data)

    # -- requests ----------------------------------------------------------------------------------
    async def call(self, op: str, *, timeout: float = CALL_TIMEOUT_S, _connecting: bool = False,
                   **args: Any) -> Any:
        """Send one op and return its typed response data. Raises AgentError (or AgentUnavailable
        when not connected)."""
        w = self._w
        if w is None or (not _connecting and not self.connected.is_set()):
            raise AgentUnavailable()
        self._id += 1
        rid = self._id
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = (op, fut)
        req = {"id": rid, "op": op, **{k: s.to_json(v) for k, v in args.items() if v is not None}}
        try:
            w.write(s.encode_line(req))
            return await asyncio.wait_for(fut, timeout)
        except (ConnectionError, OSError, TimeoutError) as e:
            raise AgentUnavailable() from e
        finally:
            self._pending.pop(rid, None)
