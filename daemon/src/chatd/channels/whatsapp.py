"""The WhatsApp channel (docs/whatsapp/ARCHITECTURE.md W5.1, W5.2, W5.6): runs `wa-bridge` as a
child process and speaks its stdio protocol (docs/whatsapp/bridge-protocol.md, version 2).

- **Process.** The bridge gets its flags from config and the channel context: the allowlist
  (never-listed chats removed), the immutable never-list, trusted devices, the per-minute cap, an
  optional budget, `--dry-run` and `--test`. Its stdin is the lifeline: closing it stops the bridge.
- **Requests** carry increasing ids, are answered in any order, and time out after 30 s. A timeout
  marks the channel degraded (status ERROR) until the next answer; two in a row restart the bridge.
- **Supervisor** (bridge exit codes): 1 or a crash restarts with backoff 1, 2, 4 ... 60 s and stops
  after 5 crashes in 10 minutes; 2 (bad flags, version mismatch) and 3 (budget spent) never
  restart; 75 (store locked by a test run) is never retried and waits for `resume()`.
- **Inbound.** The bridge filters (§5); this adapter re-checks every rule it can see (allowlist and
  never-list, trusted device, own device, sent ids, the `Mac · ` prefix, age, handled ids) and
  maps the event to an `InboundMessage`. Voice, audio, video and stickers arrive flagged
  `unsupported` and are never downloaded.
- **Outbound** (R1, R2): destination check, then the governor, then the bridge.
- **Context** (W10): in chats of the context set, other members' messages (and on-demand history)
  arrive as `context` events and go only to `on_context`, never to the trigger sink. Quotes carry
  sender, links and a quoted file (realpath-checked); a quote from another chat carries nothing.
- **Allowlist** (W9.8): chatd starts the bridge with the context's chats (empty in production, so
  it fails closed) and fills it with `set_allowed` from agentd's profiles. The last allowed set is
  re-sent after every bridge start and every `connected`, then `on_connected` runs (chatd's member
  check). Under `--test` the allowlist never changes and `set_allow` is never sent.

The adapter never interprets text: it moves it between WhatsApp and the runner unchanged. Logs carry
ids, kinds, lengths and codes only, never message text, captions, file names or chat subjects.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import json
import logging
import logging.handlers
import mimetypes
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from chatd.channels.base import (
    Attachment,
    Cap,
    ChannelStatus,
    ChatChanged,
    ChatChangedCallback,
    ContextCallback,
    ContextMessage,
    DestinationRefused,
    InboundMessage,
    Limits,
    LinkState,
    MessageRef,
    MessageSink,
    Quoted,
    QuotedMedia,
    StatusCallback,
)
from chatd.channels.registry import ChannelContext
from chatd.format import PREFIX
from chatd.governor import BudgetExceeded

log = logging.getLogger(__name__)

PROTOCOL = "2"
HOME = Path("~/.mac-agent").expanduser()
DEFAULT_BRIDGE = HOME / "bin" / "wa-bridge"
DEFAULT_STORE = HOME / "chat" / "whatsapp" / "device.db"
DEFAULT_MEDIA = HOME / "chat" / "whatsapp" / "media"
DEFAULT_LOG = Path("~/Library/Logs/Mac/chat/wa-bridge.log").expanduser()
LINE_LIMIT = 2 * 1024 * 1024  # the bridge's lines are at most 1 MiB
MAX_REQUEST = 1024 * 1024
REQUEST_TIMEOUT_S = 30.0
CONNECT_WAIT_S = 10.0
CLOSE_WAIT_S = 5.0
BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
BREAKER = (5, 600.0)  # crashes, within seconds
EDIT_WINDOW_S = 14 * 60.0  # WhatsApp allows 15 min; a minute of margin for clock skew
KNOWN_MAX = 2_000
EXIT_CRASH, EXIT_BAD_ARGS, EXIT_BUDGET, EXIT_LOCKED = 1, 2, 3, 75
UNSUPPORTED = ("voice", "audio", "video", "sticker")
IMAGE_MIMES = ("image/jpeg", "image/png")

# Status details: plain words for the app's WhatsApp row and pane.
CONNECTING = "Connecting to WhatsApp."
SCAN = "On your phone, open WhatsApp ▸ Settings ▸ Linked devices ▸ Link a device, then scan this."
NOT_LINKED = "Not linked."
OFFLINE = "Offline. Mac reconnects by itself."
UNLINKED_PHONE = "Unlinked from your phone. Link again in Mac's Settings."
UNLINKED = "Unlinked from this Mac. Link again in Mac's Settings."
BANNED = "WhatsApp blocked this linked device, so Mac can't use WhatsApp on this number."
PAIR_TIMEOUT = "The QR code expired. Try linking again."
STORE_ERROR = "Mac can't read its WhatsApp data. Unlink and link again."
IN_USE = "In use by a test run."
BAD_BRIDGE = "The WhatsApp helper doesn't match this version of Mac. Reinstall Mac."
BUDGET_SPENT = "The test run's message budget is spent."
CRASHING = "The WhatsApp helper keeps stopping. Quit and reopen Mac."
RESTARTING = "The WhatsApp helper stopped. Restarting it."
NOT_ANSWERING = "The WhatsApp helper stopped answering."
NOT_INSTALLED = "The WhatsApp helper isn't installed. Run install.sh again."
STOPPED = "WhatsApp stopped."
NOT_RUNNING = "WhatsApp isn't running."


class BridgeError(Exception):
    """A failed bridge request. `code` is the protocol's error code (or "timeout"/"offline"
    when the bridge did not answer); the message is plain words and never holds message text."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class _Recent(collections.OrderedDict[str, Any]):
    """The last KNOWN_MAX entries, oldest dropped first."""

    def put(self, key: str, value: Any) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > KNOWN_MAX:
            self.popitem(last=False)


class WhatsAppChannel:
    name = "whatsapp"

    def __init__(self, cfg: dict[str, Any], ctx: ChannelContext) -> None:
        self.ctx = ctx
        self.typing = bool(cfg.get("typing", False))  # chat_presence only if spike P0.2 keeps typing
        self.caps = Cap.EDIT | Cap.REACT | Cap.FILES | (Cap.TYPING if self.typing else Cap.NONE)
        self.limits = Limits(edit_window_s=float(cfg.get("edit_window_s", EDIT_WINDOW_S)),
                             min_edit_interval_s=float(cfg.get("min_edit_interval_s", 4.0)),
                             text_format="whatsapp")
        self.command = [str(x) for x in cfg.get("command") or [Path(cfg.get("bridge", DEFAULT_BRIDGE)).expanduser()]]
        self.store = Path(cfg.get("store", DEFAULT_STORE)).expanduser()
        self.media_dir = Path(cfg.get("media_dir", DEFAULT_MEDIA)).expanduser()
        trusted = cfg.get("trusted_devices", "all")  # W10.8: every device of the owner's account
        self.trusted_all = trusted == "all"
        self.trusted_devices = () if self.trusted_all else tuple(int(d) for d in trusted)
        self.max_per_min = int(cfg.get("max_per_min", 20))
        self.test = bool(ctx.test or cfg.get("test", False))
        budget = cfg.get("budget")
        if budget is None and self.test:
            budget = ctx.governor.budget.total
        if self.test and budget is None:
            raise ValueError("WhatsApp test mode needs a message budget")
        self.budget: int | None = None if budget is None else int(budget)
        self.presence = str(cfg.get("presence", "unavailable"))
        if self.presence not in ("available", "unavailable"):
            raise ValueError("presence must be available or unavailable")
        self.request_timeout_s = float(cfg.get("request_timeout_s", REQUEST_TIMEOUT_S))
        self.connect_wait_s = float(cfg.get("connect_wait_s", CONNECT_WAIT_S))
        self.backoff = tuple(float(x) for x in cfg.get("backoff", BACKOFF_S))
        self.breaker = tuple(cfg.get("breaker", BREAKER))
        log_path = cfg.get("log_path", DEFAULT_LOG)
        self.log_path = Path(log_path).expanduser() if log_path else None

        self.exit_code: int | None = None  # the bridge's last exit code
        self.degraded = False
        self._status = ChannelStatus(LinkState.OFFLINE, detail=CONNECTING)
        self._before_degraded: ChannelStatus | None = None
        self._sink: MessageSink | None = None
        self._on_status: StatusCallback | None = None
        self._on_chat_changed: ChatChangedCallback | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._wlock = asyncio.Lock()
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._supervisor: asyncio.Task[None] | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._timeouts = 0
        self._closing = False
        self._exhausted = False  # the bridge's budget is spent (exit 3 follows)
        self._own_device: int | None = None
        self._sent = _Recent()  # our message id -> sent time (ctx clock)
        self._from_me = _Recent()  # message id -> sent from the owner's number (for reply_to/react)
        self._bridge_log: logging.Handler | None = None
        self._allowed: frozenset[str] | None = None  # the last set_allowed set (re-sent after restarts)
        self._context: frozenset[str] = frozenset()  # allowed chats whose other messages are context
        self._on_context: ContextCallback | None = None
        self._resync: asyncio.Task[None] | None = None
        # chatd's hook, run after every bridge start or `connected` once the allowlist is re-sent
        # (W9.8 member compare: chatd calls list_chats and pauses chats that became shared).
        self.on_connected: Callable[[], Awaitable[None]] | None = None

    # -- lifecycle ------------------------------------------------------------------------------------
    async def start(self, sink: MessageSink, on_status: StatusCallback,
                    on_chat_changed: ChatChangedCallback, *, on_context: ContextCallback | None = None) -> None:
        self._sink, self._on_status, self._on_chat_changed = sink, on_status, on_chat_changed
        self._on_context = on_context
        if self.log_path is not None and self._bridge_log is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._bridge_log = logging.handlers.RotatingFileHandler(self.log_path, maxBytes=5 * 1024 * 1024,
                                                                     backupCount=3)
            self._bridge_log.setFormatter(logging.Formatter("%(message)s"))  # the bridge stamps its lines
        self._dispatcher = asyncio.create_task(self._dispatch(), name="whatsapp:events")
        await self._launch()

    async def resume(self) -> None:
        """Start the bridge again after it stopped for good (exit 75 "in use by a test run", 2, 3,
        or the crash breaker). A running bridge is left alone."""
        if self._closing or (self._supervisor is not None and not self._supervisor.done()):
            return
        self._exhausted = False
        await self._launch()

    async def _launch(self) -> None:
        self._ready.clear()
        self._supervisor = asyncio.create_task(self._supervise(), name="whatsapp:bridge")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._ready.wait(), self.request_timeout_s + self.connect_wait_s)

    async def close(self) -> None:
        self._closing = True
        proc = self._proc
        if proc is not None and proc.stdin is not None:
            with contextlib.suppress(Exception):
                proc.stdin.close()  # the lifeline: the bridge disconnects and exits 0
        if self._supervisor is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._supervisor), CLOSE_WAIT_S)
            except TimeoutError:
                log.warning("whatsapp: the bridge did not stop in %.0f s; killing it", CLOSE_WAIT_S)
                if self._proc is not None:
                    with contextlib.suppress(ProcessLookupError):
                        self._proc.kill()
                with contextlib.suppress(Exception):
                    await self._supervisor
            except Exception:
                log.warning("whatsapp: the bridge supervisor failed", exc_info=True)
        if self._resync is not None:
            self._resync.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._resync
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatcher
        if self._bridge_log is not None:
            self._bridge_log.close()
            self._bridge_log = None

    def status(self) -> ChannelStatus:
        return self._status

    def _set(self, state: LinkState, **kw: Any) -> None:
        self.degraded = False
        self._publish(ChannelStatus(state, **kw))

    def _publish(self, status: ChannelStatus) -> None:
        self._status = status
        if self._on_status is not None:
            self._on_status(status)

    def _stop(self, state: LinkState, detail: str) -> None:
        self._set(state, detail=detail)
        self._ready.set()

    # -- the bridge process and its supervisor ----------------------------------------------------------
    def argv(self) -> list[str]:
        allow = sorted(self.ctx.chats - self.ctx.never_chats)
        args = [*self.command, "--store", str(self.store), "--media-dir", str(self.media_dir),
                f"--allow={','.join(allow)}", f"--deny={','.join(sorted(self.ctx.never_chats))}",
                f"--context={','.join(sorted(self._context & set(allow)))}",
                "--trusted-devices=" + ("all" if self.trusted_all else ",".join(str(d) for d in self.trusted_devices)),
                f"--max-per-min={self.max_per_min}"]
        if self.budget is not None:
            args.append(f"--budget={self.budget}")
        if self.ctx.dry_run:
            args.append("--dry-run")
        if self.test:
            args.append("--test")
        return args

    async def _supervise(self) -> None:
        loop = asyncio.get_running_loop()
        crashes: collections.deque[float] = collections.deque()
        limit, window = int(self.breaker[0]), float(self.breaker[1])
        while not self._closing:
            try:
                code = await self._run_once()
            except OSError as e:
                log.error("whatsapp: the bridge could not start (%s)", type(e).__name__)
                self._stop(LinkState.ERROR, NOT_INSTALLED)
                return
            self.exit_code = code
            if self._closing:
                return
            log.warning("whatsapp: the bridge exited with code %s", code)
            if code == 0:
                self._stop(LinkState.OFFLINE, STOPPED)
                return
            if code == EXIT_BAD_ARGS:
                self._stop(LinkState.ERROR, BAD_BRIDGE)
                return
            if code == EXIT_BUDGET:
                self._exhausted = True
                self._stop(LinkState.ERROR, BUDGET_SPENT)
                return
            if code == EXIT_LOCKED:
                self._stop(LinkState.ERROR, IN_USE)  # no retry; waits for resume()
                return
            now = loop.time()
            crashes.append(now)
            while crashes and crashes[0] < now - window:
                crashes.popleft()
            if len(crashes) >= limit:
                self._stop(LinkState.ERROR, CRASHING)
                return
            self._stop(LinkState.OFFLINE, RESTARTING)
            await asyncio.sleep(self.backoff[min(len(crashes), len(self.backoff)) - 1])

    async def _run_once(self) -> int:
        proc = await asyncio.create_subprocess_exec(
            *self.argv(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=LINE_LIMIT)
        self._proc, self._timeouts = proc, 0
        log.info("whatsapp: bridge started (pid %d)", proc.pid)
        stderr = asyncio.create_task(self._pump_stderr(proc), name="whatsapp:stderr")
        hello = asyncio.create_task(self._handshake(), name="whatsapp:hello")
        try:
            await self._read(proc)
        finally:
            self._proc = None
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(BridgeError("offline", NOT_RUNNING))
            self._pending.clear()
            code = await proc.wait()
            hello.cancel()
            for t in (hello, stderr):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(t, 1.0)
        return code

    async def _handshake(self) -> None:
        try:
            data = await self.request("hello", protocol=PROTOCOL)
        except BridgeError as e:
            log.warning("whatsapp: hello failed (%s)", e.code)
            return
        if isinstance(data.get("device"), int):
            self._own_device = data["device"]
        log.info("whatsapp: bridge %s, protocol %s, linked %s, dry run %s, test %s",
                 data.get("bridge_version"), data.get("protocol"), data.get("linked"), data.get("dry_run"),
                 data.get("test"))
        await self._reapply()
        if self.presence == "available":
            with contextlib.suppress(BridgeError):
                await self.request("presence", state="available")
        if not data.get("linked"):
            self._stop(LinkState.UNLINKED, NOT_LINKED)
            return
        await asyncio.sleep(self.connect_wait_s)
        if not self._ready.is_set():
            self._stop(LinkState.OFFLINE, CONNECTING)

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while True:
            try:
                line = await proc.stdout.readline()
            except ValueError:
                log.warning("whatsapp: dropped an overlong line from the bridge")
                continue
            if not line:
                return
            try:
                obj = json.loads(line)
            except ValueError:
                log.warning("whatsapp: unreadable line from the bridge (%d bytes)", len(line))
                continue
            if not isinstance(obj, dict):
                continue
            if "ev" in obj:
                self._events.put_nowait(obj)
            elif "id" in obj:
                self._answer(obj)

    async def _pump_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """The bridge's own log (ids, kinds and lengths only) goes to wa-bridge.log, rotated."""
        assert proc.stderr is not None
        with contextlib.suppress(ValueError):
            while line := await proc.stderr.readline():
                if self._bridge_log is not None:
                    rec = logging.makeLogRecord({"msg": line.decode("utf-8", "replace").rstrip(),
                                                 "levelno": logging.INFO, "levelname": "INFO"})
                    self._bridge_log.handle(rec)

    # -- requests ------------------------------------------------------------------------------------------
    async def request(self, op: str, **args: Any) -> dict[str, Any]:
        """Send one command and wait for its answer (answers may arrive in any order)."""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.is_closing():
            raise BridgeError("offline", NOT_RUNNING)
        rid = next(self._ids)
        line = json.dumps({"id": rid, "op": op, **args}, ensure_ascii=False).encode() + b"\n"
        if len(line) > MAX_REQUEST:
            raise BridgeError("bad_args", "That is too long to send.")
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            async with self._wlock:
                proc.stdin.write(line)
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            self._pending.pop(rid, None)
            raise BridgeError("offline", NOT_RUNNING) from e
        try:
            return await asyncio.wait_for(fut, self.request_timeout_s)
        except TimeoutError:
            self._pending.pop(rid, None)
            self._timed_out(op, rid)
            raise BridgeError("timeout", NOT_ANSWERING) from None

    def _answer(self, obj: dict[str, Any]) -> None:
        rid = obj.get("id")
        fut = self._pending.pop(rid, None) if isinstance(rid, int) else None
        if fut is None:
            log.warning("whatsapp: bridge answer for unknown request %r (%s)", rid, obj.get("code", "ok"))
            return
        self._timeouts = 0
        if self.degraded and self._before_degraded is not None:
            self.degraded = False
            self._publish(self._before_degraded)
        if fut.done():
            return
        if obj.get("ok"):
            data = obj.get("data")
            fut.set_result(data if isinstance(data, dict) else {})
        else:
            fut.set_exception(BridgeError(str(obj.get("code", "internal")), str(obj.get("error", ""))))

    def _timed_out(self, op: str, rid: int) -> None:
        self._timeouts += 1
        log.warning("whatsapp: request %d (%s) timed out; the channel is degraded", rid, op)
        if not self.degraded:
            self._before_degraded = self._status
            self._publish(ChannelStatus(LinkState.ERROR, detail=NOT_ANSWERING))
            self.degraded = True
        if self._timeouts >= 2 and self._proc is not None:
            log.warning("whatsapp: the bridge stopped answering; restarting it")
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()

    async def _call(self, op: str, **args: Any) -> dict[str, Any]:
        """A request whose refusals map to the contract's exceptions."""
        try:
            return await self.request(op, **args)
        except BridgeError as e:
            if e.code in ("denied", "not_allowed"):
                raise DestinationRefused("That chat is not one Mac may write to.") from e
            if e.code == "budget":
                self._exhausted = True
                raise BudgetExceeded("the WhatsApp budget is spent") from e
            if e.code == "rate":
                raise BudgetExceeded("the WhatsApp per-minute cap was reached") from e
            raise

    # -- events --------------------------------------------------------------------------------------------
    async def _dispatch(self) -> None:
        while True:
            ev = await self._events.get()
            try:
                await self._event(ev)
            except Exception as e:  # noqa: BLE001 (a sink failure must not stop the channel; no traceback: it may hold text)
                log.error("whatsapp: handling a %s event failed (%s)", ev.get("ev"), type(e).__name__)
            finally:
                self._events.task_done()

    async def _event(self, ev: dict[str, Any]) -> None:
        name = ev.get("ev")
        if name == "message":
            await self._message(ev)
        elif name == "context":
            await self._context_event(ev)
        elif name == "connected":
            if isinstance(ev.get("device"), int):
                self._own_device = ev["device"]
            self._set(LinkState.CONNECTED, account=str(ev.get("jid", "")))
            self._ready.set()
            self._resync = asyncio.create_task(self._after_connected(), name="whatsapp:resync")
        elif name == "qr":
            self._set(LinkState.PAIRING, detail=SCAN, qr=str(ev.get("code", "")))
        elif name == "offline":
            log.info("whatsapp: offline (%s)", ev.get("reason"))
            self._set(LinkState.OFFLINE, detail=OFFLINE)
        elif name == "logged_out":
            reason = ev.get("reason")
            log.warning("whatsapp: logged out (%s)", reason)
            detail = {"banned": BANNED, "phone": UNLINKED_PHONE}.get(str(reason), UNLINKED)
            self._stop(LinkState.UNLINKED, detail)
        elif name == "chat_changed":
            chat = str(ev.get("chat", ""))
            if self.ctx.allowed(chat) and self._on_chat_changed is not None:
                log.info("whatsapp: chat %s changed (members %s, owner only %s)", chat, ev.get("members"),
                         ev.get("owner_only"))
                await self._on_chat_changed(ChatChanged(chat, str(ev.get("subject", "")),
                                                        owner_only=bool(ev.get("owner_only", False)),
                                                        extra={"members": ev.get("members")}))
        elif name == "budget_exhausted":
            self._exhausted = True
            log.warning("whatsapp: the bridge budget is spent (%s used)", ev.get("used"))
            self._set(LinkState.ERROR, detail=BUDGET_SPENT)
        elif name == "error":
            code = ev.get("code")
            log.warning("whatsapp: bridge error %s", code)
            if code == "pair_timeout":
                self._set(LinkState.UNLINKED, detail=PAIR_TIMEOUT)
            elif code == "store_error":
                self._set(LinkState.ERROR, detail=STORE_ERROR)
        elif name == "ready":
            log.debug("whatsapp: bridge ready (linked %s)", ev.get("linked"))

    def _drop_reason(self, chat: str, mid: str, text: str, device: Any, sent_at: float, from_me: Any) -> str:
        """W5.2 checks 1-5, re-checked here, plus the handled ids (R4). Owner-ness (check 2) is the
        event's raw `from_me`; a message without it is dropped."""
        if not chat or not mid:
            return "malformed"
        if not self.ctx.allowed(chat):
            return "chat"
        if from_me is not True:
            return "not owner"
        if not isinstance(device, int) or not (self.trusted_all or device in self.trusted_devices):
            return "device"
        if device == self._own_device or mid in self._sent or text.strip().startswith(PREFIX.strip()):
            return "echo"
        if self.ctx.too_old(sent_at):
            return "old"
        if f"{chat}:{mid}" in self.ctx.seen:
            return "handled"
        return ""

    async def _message(self, ev: dict[str, Any]) -> None:
        chat, mid = str(ev.get("chat", "")), str(ev.get("msg", ""))
        text = ev.get("text") if isinstance(ev.get("text"), str) else ""
        sent_at = float(ev.get("sent_at") or 0.0)
        why = self._drop_reason(chat, mid, text, ev.get("sender_device"), sent_at, ev.get("from_me"))
        if why or self._sink is None:
            log.debug("whatsapp: inbound %s dropped (%s)", mid, why or "not started")
            return
        unsupported = ev.get("unsupported") if ev.get("unsupported") in UNSUPPORTED else ""
        attachments = () if unsupported else self._attachments(mid, ev.get("attachments"))
        reply_to = reaction = None
        if isinstance(q := ev.get("quoted"), dict) and q.get("msg"):
            reply_to = self._quoted(chat, mid, q)
        if isinstance(r := ev.get("reaction"), dict) and r.get("target_msg"):
            target = str(r["target_msg"])
            reaction, text = str(r.get("emoji", "")), ""
            reply_to = Quoted(MessageRef(chat, target), "", from_owner=bool(self._from_me.get(target, True)))
        self._from_me.put(mid, True)
        msg = InboundMessage(channel=self.name, chat_id=chat, ref=MessageRef(chat, mid), text=text,
                             attachments=attachments, reply_to=reply_to, reaction=reaction,
                             unsupported=unsupported, sent_at=sent_at, replayed=bool(ev.get("replayed")))
        log.debug("whatsapp: inbound %s delivered (len %d, %d files, %s%s)", mid, len(text), len(attachments),
                  unsupported or "text", ", replayed" if msg.replayed else "")
        await self._sink(msg)

    def _quoted(self, chat: str, mid: str, q: dict[str, Any]) -> Quoted:
        """The extended quote (W10.5). A quote from another chat carries nothing but its id; a
        quoted file is kept only if it resolves inside the media folder."""
        ref = MessageRef(chat, str(q["msg"]))
        if q.get("kind") == "other_chat":
            return Quoted(ref, "", from_owner=False, kind="other_chat")
        owner = bool(q.get("from_owner"))
        self._from_me.put(ref.message_id, owner)
        links = tuple(u for u in (q.get("links") or []) if isinstance(u, str)
                      and u.lower().startswith(("http://", "https://")))[:10]
        media = None
        if isinstance(m := q.get("media"), dict) and m.get("kind"):
            path = self._in_media(mid, m.get("path")) if m.get("kind") in ("image", "document") else None
            media = QuotedMedia(kind=str(m["kind"]), name=str(m.get("name", "")), mime=str(m.get("mime", "")),
                                size=int(m.get("size") or 0), path=path)
        return Quoted(ref, str(q.get("text", "")), from_owner=owner, sender=str(q.get("sender", "")),
                      sender_name=str(q.get("sender_name", "")), from_mac=bool(q.get("from_mac")),
                      at=float(q.get("at") or 0.0), links=links, media=media)

    def _in_media(self, mid: str, raw: Any) -> Path | None:
        """realpath of a bridge-written file, or None when it is missing or outside the media dir."""
        if not raw:
            return None
        path = Path(str(raw)).resolve()
        if not path.is_relative_to(self.media_dir.resolve()) or not path.is_file():
            log.warning("whatsapp: inbound %s file outside the media folder ignored", mid)
            return None
        return path

    async def _context_event(self, ev: dict[str, Any]) -> None:
        """Another member's message, or on-demand history, in a context chat (W10.1). Data only:
        it goes to on_context, never to the sink that leads to commands, replies, steers and asks."""
        chat, mid = str(ev.get("chat", "")), str(ev.get("msg", ""))
        text = ev.get("text") if isinstance(ev.get("text"), str) else ""
        if not chat or not mid or not self.ctx.allowed(chat) or chat not in self._context:
            why = "chat"
        elif mid in self._sent or text.strip().startswith(PREFIX.strip()):
            why = "echo"
        else:
            why = "" if self._on_context is not None else "no listener"
        if why:
            log.debug("whatsapp: context %s dropped (%s)", mid, why)
            return
        media = ev.get("media") if isinstance(ev.get("media"), dict) else None
        if media is not None:
            media = {k: str(media[k]) for k in ("kind", "name") if media.get(k)}  # never a path
        msg = ContextMessage(channel=self.name, chat_id=chat, ref=MessageRef(chat, mid),
                             sender=str(ev.get("sender", "")), sender_name=str(ev.get("sender_name", "")),
                             from_owner=ev.get("from_owner") is True, at=float(ev.get("at") or 0.0),
                             text=text[:4000], media=media, quoted_msg=str(ev.get("quoted_msg", "")),
                             replayed=bool(ev.get("replayed")), history=bool(ev.get("history")))
        log.debug("whatsapp: context %s (len %d%s)", mid, len(text), ", history" if msg.history else "")
        if (cb := self._on_context) is not None:
            await cb(msg)

    def _attachments(self, mid: str, raw: Any) -> tuple[Attachment, ...]:
        """Images and documents the bridge wrote into the media dir; anything else is ignored."""
        out: list[Attachment] = []
        media = self.media_dir.resolve()
        for a in raw if isinstance(raw, list) else []:
            if not isinstance(a, dict) or a.get("kind") not in ("image", "document"):
                continue
            path = Path(str(a.get("path", ""))).resolve()
            if not path.is_relative_to(media):
                log.warning("whatsapp: inbound %s attachment outside the media folder ignored", mid)
                continue
            out.append(Attachment(kind=a["kind"], path=path, mime=str(a.get("mime", "")),
                                  name=str(a.get("name", "")), size=int(a.get("size") or 0)))
        return tuple(out)

    # -- outbound (R1, R2: destination, then the governor, then the bridge) -------------------------------
    def _check(self, chat_id: str) -> None:
        self.ctx.check_destination(chat_id)
        if self._exhausted:
            raise BudgetExceeded("the WhatsApp budget is spent")

    def _sent_now(self, mid: str) -> None:
        self._sent.put(mid, self.ctx.clock())
        self._from_me.put(mid, True)

    async def mark_handled(self, ref: MessageRef) -> None:
        await self.ctx.seen.add(f"{ref.chat_id}:{ref.message_id}")

    async def send_text(self, chat_id: str, text: str, reply_to: MessageRef | None = None,
                        mentions: tuple[str, ...] = ()) -> MessageRef:
        self._check(chat_id)
        await self.ctx.governor.charge("send", chat_id)
        args: dict[str, Any] = {"chat": chat_id, "text": text}
        if mentions:
            args["mentions"] = list(mentions)  # the bridge keeps only members named in the text
        if reply_to is not None:
            args["reply_to"] = {"msg": reply_to.message_id,
                                "from_me": bool(self._from_me.get(reply_to.message_id, True))}
        mid = str((await self._call("send_text", **args))["msg"])
        self._sent_now(mid)
        return MessageRef(chat_id, mid)

    async def edit_text(self, ref: MessageRef, text: str, mentions: tuple[str, ...] = ()) -> bool:
        self._check(ref.chat_id)
        sent = self._sent.get(ref.message_id)
        if sent is not None and self.ctx.clock() - sent > self.limits.edit_window_s:
            return False  # past the window: the runner sends a new message
        await self.ctx.governor.charge("edit", ref.chat_id)
        try:
            args: dict[str, Any] = {"mentions": list(mentions)} if mentions else {}
            await self._call("edit", chat=ref.chat_id, msg=ref.message_id, text=text, **args)
        except BridgeError as e:
            if e.code in ("edit_window", "not_found"):
                return False
            raise
        return True

    async def react(self, ref: MessageRef, emoji: str) -> None:
        self._check(ref.chat_id)
        await self.ctx.governor.charge("react", ref.chat_id)
        await self._call("react", chat=ref.chat_id, msg=ref.message_id,
                         from_me=bool(self._from_me.get(ref.message_id, True)), emoji=emoji)

    async def set_typing(self, chat_id: str, on: bool) -> None:
        self.ctx.check_destination(chat_id)
        if not self.typing:
            return
        await self.ctx.governor.charge("typing", chat_id)
        await self._call("chat_presence", chat=chat_id, state="composing" if on else "paused")

    async def send_file(self, chat_id: str, path: Path, caption: str = "") -> MessageRef:
        self._check(chat_id)
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise BridgeError("bad_args", "That file doesn't exist.")
        if p.stat().st_size > self.limits.max_file_mb * 1024 * 1024:
            raise BridgeError("too_large", "That file is over 100 MB.")
        await self.ctx.governor.charge("send", chat_id)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        args: dict[str, Any] = {"chat": chat_id, "path": str(p), "kind": "image" if mime in IMAGE_MIMES else "document",
                                "file_name": p.name, "mime": mime}
        if caption:
            args["caption"] = caption
        mid = str((await self._call("send_media", **args))["msg"])
        self._sent_now(mid)
        return MessageRef(chat_id, mid)

    # -- linking and chats ------------------------------------------------------------------------------
    async def link(self) -> None:
        """Start QR pairing; the codes arrive as PAIRING statuses. A linked device: nothing to do."""
        try:
            await self._call("pair")
        except BridgeError as e:
            if e.code != "already_linked":
                raise

    async def unlink(self) -> None:
        """Unlink this Mac from the phone and delete the keys; UNLINKED follows."""
        await self._call("logout")

    async def list_chats(self) -> list[dict[str, Any]]:
        chats = (await self._call("list_chats")).get("chats", [])
        return [c for c in chats if isinstance(c, dict)] if isinstance(chats, list) else []

    async def set_allowed(self, chat_ids: Iterable[str], context_ids: Iterable[str] = frozenset()) -> None:
        """The contract's allowlist update (W9.8 step 4, W10.6): the enabled chats from agentd, and
        those of them with read_chat on. Never-listed ids are filtered out. Under test mode nothing
        changes and nothing is sent (step 8). A bridge that is down gets the sets at its next start."""
        if self.test:
            log.warning("whatsapp: allowlist change ignored in test mode")
            return
        allow = frozenset(chat_ids) - self.ctx.never_chats
        self._allowed = allow
        self._context = frozenset(context_ids) & allow  # narrowed at once; also the --context at restart
        try:
            await self.set_allow(allow, self._context)
        except BridgeError as e:
            if e.code not in ("offline", "timeout"):
                raise
            self.ctx.chats = allow  # applied through --allow at the next start
            log.info("whatsapp: bridge not reachable (%s); the allowlist applies at its next start", e.code)

    async def _reapply(self) -> None:
        """Re-send the last allowed set (the bridge keeps it only in memory)."""
        if self._allowed is None or self.test:
            return
        try:
            await self._call("set_allow", allow=sorted(self._allowed), context=sorted(self._context))
        except (BridgeError, DestinationRefused) as e:
            log.warning("whatsapp: re-sending the allowlist failed (%s)", type(e).__name__)

    async def _after_connected(self) -> None:
        await self._reapply()
        if self.on_connected is not None:
            try:
                await self.on_connected()
            except Exception as e:  # noqa: BLE001 (chatd's hook must not stop the channel)
                log.error("whatsapp: on_connected failed (%s)", type(e).__name__)

    async def set_allow(self, chats: Iterable[str], context: Iterable[str] | None = None) -> None:
        """Replace the allowlist (and, if given, the context set ⊆ it). Never-listed chats are
        refused, and so is any change under test mode; the bridge enforces both again."""
        allow = frozenset(chats)
        if allow & self.ctx.never_chats:
            raise DestinationRefused("Mac never uses that chat.")
        if self.test:
            raise BridgeError("test_mode", "The allowlist can't change during a test run.")
        ctx_set = self._context & allow if context is None else frozenset(context) & allow
        self.ctx.chats = self.ctx.chats & allow  # narrow first: an error never leaves a wider set
        self._context = self._context & ctx_set
        await self._call("set_allow", allow=sorted(allow), context=sorted(ctx_set))
        self.ctx.chats, self._context = allow, ctx_set  # widen only once the bridge has it

    @property
    def context_ids(self) -> frozenset[str]:
        """The chats whose other members' messages arrive as context."""
        return self._context

    async def members(self, chat_id: str) -> list[dict[str, Any]]:
        """The chat's participants `{jid, number, name}` (for mentions, W10.4)."""
        self.ctx.check_destination(chat_id)
        raw = (await self._call("members", chat=chat_id)).get("members", [])
        return [m for m in raw if isinstance(m, dict)] if isinstance(raw, list) else []

    async def request_history(self, chat_id: str, before: MessageRef, count: int = 30) -> bool:
        """Ask the phone for up to `count` messages before `before` (W10.3). Results arrive later as
        context with `history`. False when refused (not a context chat, dry run, hourly limit,
        budget, offline); never raises for those."""
        self.ctx.check_destination(chat_id)
        if self.ctx.dry_run or chat_id not in self._context or self._exhausted:
            return False
        try:
            await self._call("history", chat=chat_id, before_msg=before.message_id, count=max(1, min(count, 50)))
        except (BridgeError, BudgetExceeded) as e:
            log.info("whatsapp: history not requested (%s)", getattr(e, "code", type(e).__name__))
            return False
        return True

    async def bridge_status(self) -> dict[str, Any]:
        """The bridge's counters: linked, connected, sent_last_min, budget_left, total_outbound."""
        return await self._call("status")


def create(cfg: dict[str, Any], ctx: ChannelContext) -> WhatsAppChannel:
    return WhatsAppChannel(cfg, ctx)
