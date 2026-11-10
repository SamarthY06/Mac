"""The agentd API server (api §1-§5): NDJSON over agent.sock, one reader task per connection.

- Socket 0600 in a 0700 dir; a peer with another uid is closed with no reply.
- The first line must be `hello`. Identity comes from the token's kind, never from the payload.
- Ops on one connection are processed strictly in order by the reader (V1). No op makes it wait for
  long work: `ask`, `handoff` and `connect_check` reply at once (B2).
- Each connection has one ordered output queue (responses and events). More than 10,000 queued
  events closes it with `slow_client`.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentd import schema as s
from agentd.service import SERVER_NAME, ApiError, Service
from agentd.stream import GapError, Stream
from macd.unixsock import bind_private_unix_socket, peer_uid

log = logging.getLogger(__name__)

_HIGH_WATER = 512 * 1024  # bytes buffered in the transport before the writer drains


@dataclass
class _Subscribed:
    data: s.SubscribeData
    attach: Callable[[], None]


class Conn:
    def __init__(self, server: Server, writer: asyncio.StreamWriter) -> None:
        self.server = server
        self.writer = writer
        self.kind: s.ClientKind | None = None
        self.epoch = ""
        self.client = ""
        self.granted: list[str] = []
        self.voice_bound = False  # this connection drives voice:local (epoch rules, V2)
        self.subs: dict[str, Stream] = {}
        self._q: collections.deque[tuple[bytes, bool]] = collections.deque()
        self._events = 0
        self._wake = asyncio.Event()
        self.closing = False

    # -- output ------------------------------------------------------------------------
    def send(self, obj: dict[str, Any]) -> None:
        if not self.closing:
            self._q.append((s.encode_line(obj), False))
            self._wake.set()

    def deliver(self, conv: str, line: bytes, epoch: str | None) -> None:
        """Stream sink. Voice only: events of another client epoch are never delivered (B1)."""
        if self.closing or (conv == s.VOICE_CONV and epoch is not None and epoch != self.epoch):
            return
        if self._events >= self.server.queue_events:
            log.warning("api: %s client too slow; closing", self.kind.value if self.kind else "?")
            self.fatal(s.ErrorCode.SLOW_CLIENT, "You fell too far behind. Subscribe again from your last position.")
            return
        self._events += 1
        self._q.append((line, True))
        self._wake.set()

    def fatal(self, code: s.ErrorCode, message: str, req_id: int | None = None) -> None:
        """Send one error, then close (bad_frame, hello_required, slow_client, ...)."""
        if self.closing:
            return
        if code is s.ErrorCode.SLOW_CLIENT:
            self._q.clear()
            self._events = 0
        self._q.append((s.encode_line(s.encode_error(req_id, code, message)), False))
        self.closing = True
        self.detach_all()
        self._wake.set()

    def detach_all(self) -> None:
        for stream in self.subs.values():
            stream.detach(self)
        self.subs.clear()

    async def write_loop(self) -> None:
        w = self.writer
        try:
            while True:
                while self._q:
                    line, is_event = self._q.popleft()
                    w.write(line)
                    if is_event:
                        self._events -= 1
                    if w.transport.get_write_buffer_size() > _HIGH_WATER:
                        await w.drain()
                if self.closing:
                    await w.drain()
                    break
                self._wake.clear()
                await self._wake.wait()
        except (ConnectionError, OSError):
            pass
        finally:
            self.closing = True
            self.detach_all()
            with contextlib.suppress(Exception):
                w.close()


class Server:
    def __init__(self, service: Service, tokens: dict[str, s.ClientKind], sock_path: Path, *,
                 uid: int | None = None, queue_events: int = s.CLIENT_QUEUE_EVENTS) -> None:
        self.svc = service
        self.tokens = dict(tokens)
        self.sock_path = Path(sock_path)
        self.uid = os.getuid() if uid is None else uid
        self.queue_events = queue_events
        self.conns: set[Conn] = set()
        self._server: asyncio.base_events.Server | None = None
        self._handlers: set[asyncio.Task[Any]] = set()
        service.notice_sink = self._notice_chat_clients

    async def start(self) -> None:
        sock = bind_private_unix_socket(self.sock_path)
        self._server = await asyncio.start_unix_server(self._handle, sock=sock, limit=s.MAX_LINE_BYTES + 2)
        log.info("api: listening on %s", self.sock_path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for c in list(self.conns):
            c.closing = True
            c._wake.set()
        for t in list(self._handlers):
            t.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        with contextlib.suppress(OSError):
            self.sock_path.unlink()

    def _notice_chat_clients(self, notice: dict[str, Any]) -> None:
        """`profile_changed` (connection-level: no conv, no seq, not replayed) to every chat client."""
        for c in self.conns:
            if c.kind is s.ClientKind.CHAT:
                c.send(notice)

    # -- connection ----------------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            uid = peer_uid(writer.get_extra_info("socket"))
        except OSError:
            uid = -1
        if uid != self.uid:
            log.warning("api: refused a peer with uid %d", uid)
            writer.close()
            return
        conn = Conn(self, writer)
        self.conns.add(conn)
        wtask = asyncio.create_task(conn.write_loop(), name="agentd:writer")
        try:
            await self._read_loop(conn, reader)
        except asyncio.CancelledError:
            pass
        finally:
            conn.closing = True
            conn.detach_all()
            conn._wake.set()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(wtask, 2.0)
            self.conns.discard(conn)
            if conn.voice_bound:
                self.svc.voice_disconnected(conn.epoch)
            if task is not None:
                self._handlers.discard(task)

    async def _read_loop(self, conn: Conn, reader: asyncio.StreamReader) -> None:
        while not conn.closing:
            try:
                line = await reader.readline()
            except (ValueError, asyncio.LimitOverrunError):
                conn.fatal(s.ErrorCode.BAD_FRAME, "A line was longer than 1 MiB.")
                return
            except (ConnectionError, OSError):
                return
            if not line:
                return
            try:
                obj = s.parse_line(line)
            except s.SchemaError as e:
                conn.fatal(s.ErrorCode.BAD_FRAME, f"That was not a valid frame ({e}).")
                return
            raw_id = obj.get("id")
            req_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else None
            if conn.kind is None and obj.get("op") != "hello":
                conn.fatal(s.ErrorCode.HELLO_REQUIRED, "Send hello first.", req_id)
                return
            try:
                req_id, op = s.decode_request(obj)
                result = await self._dispatch(conn, op)
            except s.SchemaError as e:
                conn.send(s.encode_error(req_id, e.code, str(e)))
                continue
            except ApiError as e:
                if e.code in s.CLOSING_CODES:
                    conn.fatal(e.code, e.message, req_id)
                    return
                conn.send(s.encode_error(req_id, e.code, e.message, e.data))
                continue
            except Exception:
                log.exception("api: op failed")
                conn.send(s.encode_error(req_id, s.ErrorCode.INTERNAL, "Something went wrong inside Mac."))
                continue
            if isinstance(result, _Subscribed):
                conn.send(s.encode_ok(req_id, result.data))
                result.attach()  # replay, then live: synchronous, so nothing slips between
            else:
                conn.send(s.encode_ok(req_id, result))

    # -- ops -----------------------------------------------------------------------------
    async def _hello(self, conn: Conn, op: s.Hello) -> s.HelloData:
        if conn.kind is not None:
            raise ApiError(s.ErrorCode.BAD_ARGS, "hello was already done on this connection.")
        if op.api != s.API_VERSION:
            raise ApiError(s.ErrorCode.VERSION_MISMATCH,
                           f"agentd speaks API {s.API_VERSION} but this client speaks {op.api}. "
                           "Restart Mac to update all parts together.")
        kind = self.tokens.get(op.token)
        if kind is None or kind is s.ClientKind.TEST:
            raise ApiError(s.ErrorCode.UNAUTHORIZED, "That token is not known to this Mac.")
        if not op.client_epoch:
            raise ApiError(s.ErrorCode.BAD_ARGS, "client_epoch: must not be empty")
        convs = list(dict.fromkeys(op.conversations))
        if kind is s.ClientKind.VOICE:
            ok = all(c == s.VOICE_CONV for c in convs)
        elif kind is s.ClientKind.CHAT:
            await self.svc.refresh_never()
            if not convs:  # live grants (P2): the enabled, not never-listed profiles
                convs = self.svc.live_chat_convs()
            for c in convs:
                try:
                    self.svc.chat_check(c)
                except ApiError as e:
                    if e.code is s.ErrorCode.PROFILES_UNAVAILABLE:
                        raise
                    raise ApiError(s.ErrorCode.FORBIDDEN_CONVERSATION,
                                   "A chat client may not use one of those conversations.") from None
            ok = True
        else:
            ok = not convs
        if not ok:
            raise ApiError(s.ErrorCode.FORBIDDEN_CONVERSATION,
                           f"A {kind.value} client may not use one of those conversations.")
        conn.kind, conn.epoch, conn.client, conn.granted = kind, op.client_epoch, op.client, convs
        if kind is s.ClientKind.VOICE and s.VOICE_CONV in convs:  # not a side connection (stop_other)
            conn.voice_bound = True
            await self.svc.voice_connected(op.client_epoch)
        log.info("api: hello from %s (%s)", op.client, kind.value)
        return s.HelloData(api=s.API_VERSION, server=SERVER_NAME, kind=kind, boot=self.svc.boot,
                           conversations=convs, interrupted=self.svc.take_notices(convs))

    def _check_conv(self, conn: Conn, conv: str) -> None:
        if conn.kind is s.ClientKind.APP:
            if conv not in self.svc.convs:
                raise ApiError(s.ErrorCode.BAD_ARGS, "There is no such conversation.")
        elif conn.kind is s.ClientKind.CHAT:
            self.svc.chat_check(conv)  # live, every op (P2)
        elif conv not in conn.granted:
            raise ApiError(s.ErrorCode.FORBIDDEN_CONVERSATION, "This client did not open that conversation.")

    async def _dispatch(self, conn: Conn, op: s.Op) -> Any:
        if isinstance(op, s.Hello):
            return await self._hello(conn, op)
        assert conn.kind is not None
        try:
            s.check_op_for_kind(conn.kind, op)
        except s.SchemaError as e:
            raise ApiError(e.code, str(e)) from None
        conv = getattr(op, "conv", None)
        if isinstance(op, s.Pause) and op.reason == "never_listed" and op.value:
            if op.conv == s.VOICE_CONV or ":" not in op.conv:  # only narrows: any kind, any chat
                raise ApiError(s.ErrorCode.BAD_ARGS, "never_listed is for a chat conversation.")
        elif conv is not None:
            self._check_conv(conn, conv)
        svc, ep = self.svc, conn.epoch
        match op:
            case s.Ask():
                return await svc.ask(ep, op)
            case s.Steer():
                return await svc.steer(ep, op)
            case s.Interrupt():
                return await svc.interrupt(ep, op)
            case s.Cancel():
                return await svc.cancel(op.conv)
            case s.StopOther():
                return await svc.stop_other()
            case s.Confirm():
                return svc.confirm(op)
            case s.RevokeAll():
                svc.revoke_all(op.conv)
            case s.QuestionPosted():
                svc.question_posted(op)
            case s.Reply():
                return await svc.reply(op)
            case s.NewSession():
                await svc.new_session(op)
            case s.Reset():
                await svc.reset(op.conv)
            case s.Warm():
                await svc.warm(op.conv)
            case s.Sleep():
                await svc.sleep(op.conv)
            case s.Handoff():
                return svc.handoff(op.conv)
            case s.ConnectCheck():
                return svc.connect_check(op)
            case s.SetContext():
                await svc.set_context(op.conv, op.locked)
            case s.SetLanguage():
                await svc.set_language(op.conv, op.lang)
            case s.SetName():
                await svc.set_name(op.conv, op.name)
            case s.VoiceSettings():
                await svc.set_voice_settings(op)
            case s.Subscribe():
                return self._subscribe(conn, op)
            case s.Ack():
                await svc.ack(op.conv, op.turn_uid)
            case s.Status():
                if op.conv is not None:
                    return svc.conversation(op.conv).status()
                if conn.kind is s.ClientKind.CHAT:
                    return svc.status(svc.live_chat_convs())
                return svc.status(None if conn.kind is s.ClientKind.APP else conn.granted)
            case s.Pause():
                return await svc.pause(op.conv, op.value, op.reason, conn.kind)
            case s.WsList():
                return await svc.ws_list()
            case s.WsApprove():
                return await svc.ws_approve(op.path)
            case s.WsRemove():
                await svc.ws_remove(op.path)
            case s.ProfileList():
                return await svc.profile_list(conn.kind)
            case s.ProfileCreate():
                return await svc.profile_create(op)
            case s.ProfileSet():
                return await svc.profile_set(op)
            case s.ProfileDelete():
                await svc.profile_delete(op)
            case s.ProfileCommand():
                return await svc.profile_command(op)
            case _:
                raise ApiError(s.ErrorCode.UNKNOWN_OP, f"unknown op {op.OP!r}")
        return None

    def _subscribe(self, conn: Conn, op: s.Subscribe) -> _Subscribed:
        stream = self.svc.conversation(op.conv).stream
        try:
            stream.check_after(op.after)
        except GapError as g:
            raise ApiError(s.ErrorCode.GAP, "That stream position is gone. Rebuild from status, then "
                           "subscribe again.", g.data) from None

        def attach() -> None:
            old = conn.subs.pop(op.conv, None)
            if old is not None:
                old.detach(conn)
            if not conn.closing:
                stream.attach(conn, op.after)
                conn.subs[op.conv] = stream

        return _Subscribed(s.SubscribeData(boot=stream.boot, seq=stream.seq), attach)
