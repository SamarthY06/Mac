"""RemoteBrain and RemoteGate: the voice brain over agent.sock (ARCHITECTURE W6, docs/agentd-api.md).

`brain.mode = remote` swaps the in-process ClaudeBrain for this client. The orchestrator does not
change: RemoteBrain implements `contracts.Brain` and RemoteGate the orchestrator's `Gate`.

- **One connection, ops in order.** Every op is written synchronously on one connection, so the
  orchestrator's cancel sequence `resolve(False)`, `revoke_all()`, `interrupt` reaches agentd in
  that order (V1). Events of `voice:local` arrive on the same connection.
- **Client epoch** (V2): a random 128-bit hex per process, sent in every `hello`; a reconnect from
  the same process reuses it. agentd maps (epoch, turn) to its own turn ids, so the orchestrator's
  counter, which restarts at 1 with every macd start, never collides with an older turn.
- **Confirm binding** (V1): RemoteGate remembers the `question_id` of the CONFIRM the orchestrator
  is answering. A newer CONFIRM of the same turn never replaces it, so a late "yes" can only reach
  the question it answers (agentd rejects it with `stale_question` once that one is closed).
- **Long ops** (B2): `handoff` and `connect_check` reply `{accepted}` at once; their results arrive
  as `status{handoff_done}` and `status{connect_check}` events, which these methods await.
- **Liveness:** `alive` is agentd's pushed `status{alive}` (read once from `status` per connect).
- **agentd unreachable:** `ask` ends the turn with ERROR(internal) and a plain line; the connection
  is retried with backoff (api §8), and `warm()` retries at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentd import schema as s
from macd import config
from macd.config import Config
from macd.contracts import (
    TASK_OPS,
    BrainErrorKind,
    BrainEvent,
    BrainEventKind,
    TaskApiError,
)

log = logging.getLogger(__name__)

CLIENT_NAME = "macd/0.9.0"
CONV = s.VOICE_CONV
UNREACHABLE = "I lost my connection to my brain. Please try again."
BUSY = "I'm still finishing the last task. Please try again."
BACKOFF_S = (0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0)  # api §8
HELLO_TIMEOUT_S = 5.0
OP_TIMEOUT_S = 5.0
ASK_CONNECT_WAIT_S = 1.0  # an ask waits this long for a reconnect in progress, then fails plainly
HANDOFF_WAIT_S = 65.0  # agentd bounds the handoff at 60 s
CONNECT_CHECK_WAIT_S = 30.0  # agentd bounds the check at 25 s
_AGENTD_GENERIC_ERROR = "Something went wrong."  # agentd's stand-in for an empty brain message
TASK_OP_TIMEOUT_S = 15.0
VOICE_TOKEN_FILE = "voice-token"  # written by macsvc next to app-token (tasks DESIGN D1)
TASKS_NO_TOKEN = "Tasks are not available right now: Mac's background service is not running."
TASKS_WHICH_TASK = "Which task? Call mcp__mac__task_list and give its task_id."

_ERROR_KINDS = {k.value: k for k in BrainErrorKind}


def agent_sock_path() -> Path:
    return config.RUN_DIR / "agent.sock"


def voice_token(lifeline_line: str | None = None) -> str | None:
    """The voice token: `{"token": ...}`, the first line Mac.app writes on macd's lifeline (stdin);
    under macsvc, `run/voice-token` (0600, removed at stop; tasks D1); for development without
    the app, `dev-tokens.json` that `agentd --dev` writes (api §3)."""
    if lifeline_line:
        with contextlib.suppress(ValueError, AttributeError):
            tok = json.loads(lifeline_line).get("token")
            if isinstance(tok, str) and tok:
                return tok
    with contextlib.suppress(OSError, ValueError):
        tok = (config.RUN_DIR / VOICE_TOKEN_FILE).read_text().strip()
        if tok:
            return tok
    with contextlib.suppress(OSError, ValueError, AttributeError):
        tok = json.loads((config.RUN_DIR / "dev-tokens.json").read_text()).get("voice")
        if isinstance(tok, str) and tok:
            return tok
    return None


class RemoteError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ----------------------------------------------------------------------------- connection


class AgentLink:
    """One NDJSON connection to agentd: `hello`, requests written in order, events to a callback."""

    def __init__(self, sock_path: Path, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.sock_path = Path(sock_path)
        self.on_event = on_event
        self.closed = asyncio.Event()
        self._r: asyncio.StreamReader | None = None
        self._w: asyncio.StreamWriter | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._task: asyncio.Task[None] | None = None

    async def open(self, token: str, epoch: str, conversations: list[str],
                   client: str = CLIENT_NAME) -> dict[str, Any]:
        self._r, self._w = await asyncio.open_unix_connection(str(self.sock_path), limit=s.MAX_LINE_BYTES + 2)
        self._task = asyncio.create_task(self._read(), name="agent-link:reader")
        try:
            return await self.call(s.Hello(api=s.API_VERSION, token=token, client=client, client_epoch=epoch,
                                           conversations=conversations), HELLO_TIMEOUT_S)
        except BaseException:
            await self.close()
            raise

    def send(self, op: s.Op) -> asyncio.Future[dict[str, Any]]:
        """Write `op` now (one write per op, in call order); the future gets the response."""
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        if self._w is None or self.closed.is_set():
            fut.set_exception(ConnectionError("not connected to agentd"))
            return fut
        self._next_id += 1
        self._pending[self._next_id] = fut
        self._w.write(s.encode_line(s.encode_request(self._next_id, op)))
        return fut

    async def call(self, op: s.Op, timeout: float = OP_TIMEOUT_S) -> dict[str, Any]:
        """Send `op` and return its `data`; RemoteError for an error response."""
        resp = await asyncio.wait_for(asyncio.shield(self.send(op)), timeout)
        return _data(resp)

    async def _read(self) -> None:
        assert self._r is not None
        try:
            while line := await self._r.readline():
                try:
                    obj = s.parse_line(line)
                except s.SchemaError:
                    log.warning("agentd sent a bad frame; ignored")
                    continue
                if "ev" in obj:
                    if self.on_event is not None:
                        try:
                            self.on_event(obj)
                        except Exception:
                            log.exception("agentd event handling failed")
                    continue
                fut = self._pending.pop(obj.get("id"), None) if obj.get("id") is not None else None
                if fut is None:
                    log.warning("agentd: %s (%s)", obj.get("error", "unexpected frame"), obj.get("code", "?"))
                elif not fut.done():
                    fut.set_result(obj)
        except (ConnectionError, OSError, ValueError, asyncio.LimitOverrunError):
            pass
        finally:
            self._fail_pending()
            self.closed.set()

    def _fail_pending(self) -> None:
        pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("agentd connection lost"))

    async def close(self) -> None:
        w, self._w = self._w, None
        if w is not None:
            w.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(w.wait_closed(), 1.0)
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._fail_pending()
        self.closed.set()


def _data(resp: dict[str, Any]) -> dict[str, Any]:
    if resp.get("ok") is True:
        data = resp.get("data")
        return data if isinstance(data, dict) else {}
    raise RemoteError(str(resp.get("code", "internal")), str(resp.get("error", "")))


async def stop_other_once(token: str | None, epoch: str, sock_path: Path | None = None) -> list[str]:
    """Local mode's `stop_other` (W6): a short voice-token connection just for this op. It names
    no conversation, so it never touches voice:local. Raises if agentd can't be reached."""
    if not token:
        raise ConnectionError("no voice token for agentd")
    link = AgentLink(sock_path or agent_sock_path())
    try:
        await link.open(token, epoch, [])
        data = await link.call(s.StopOther())
        return [str(c) for c in data.get("stopped", [])]
    finally:
        await link.close()


async def task_call_once(token: str | None, epoch: str, op: str, args: dict[str, Any],
                         sock_path: Path | None = None) -> dict[str, Any]:
    """Local mode's TaskApi (tasks DESIGN 9.1): one short voice-token connection per task op,
    naming no conversation (agentd derives the voice origin from the token). Returns the op's
    data; every failure is a TaskApiError with plain words for the model."""
    if op not in TASK_OPS:
        raise TaskApiError("internal", f"Unknown task op {op}.")
    if not token:
        raise TaskApiError("unauthorized", TASKS_NO_TOKEN)
    try:
        request = s.from_dict(s.OPS[op], dict(args))
    except s.SchemaError as e:
        message = TASKS_WHICH_TASK if str(e).startswith("task_id") else str(e)
        raise TaskApiError(e.code.value, message) from None
    link = AgentLink(sock_path or agent_sock_path())
    try:
        await link.open(token, epoch, [])
        return await link.call(request, TASK_OP_TIMEOUT_S)
    except RemoteError as e:
        raise TaskApiError(e.code, e.message or TASKS_NO_TOKEN) from None
    except (ConnectionError, OSError, TimeoutError) as e:
        log.warning("task op %s: agentd unreachable (%s)", op, type(e).__name__)
        raise TaskApiError("unreachable", TASKS_NO_TOKEN) from None
    finally:
        await link.close()


# ----------------------------------------------------------------------------- gate


@dataclass
class _Bound:
    question_id: str
    turn: int
    summary: str
    closed: bool = False  # agentd closed it (expired, skipped); a resolve is then stale


class RemoteGate:
    """The orchestrator's Gate over the socket: `resolve` answers the CONFIRM it bound, by id."""

    def __init__(self, brain: RemoteBrain) -> None:
        self._brain = brain
        self._bound: _Bound | None = None

    @property
    def pending(self) -> tuple[int, str] | None:
        b = self._bound
        return (b.turn, b.summary) if b is not None and not b.closed else None

    @property
    def question_id(self) -> str | None:
        return self._bound.question_id if self._bound is not None else None

    def resolve(self, approved: bool) -> None:
        b, self._bound = self._bound, None
        if b is None:
            log.debug("gate: resolve(%s) with nothing pending", approved)
            return
        self._brain._send_quiet(s.Confirm(conv=CONV, question_id=b.question_id, approved=bool(approved)))

    def revoke_all(self) -> None:
        self._bound = None
        self._brain._send_quiet(s.RevokeAll(conv=CONV))

    # -- from the event stream ------------------------------------------------------------
    def _offer(self, question_id: str, turn: int, summary: str) -> None:
        b = self._bound
        if b is not None and b.turn == turn:
            # The orchestrator still answers the earlier one (or ignores this one, N16): a late
            # "yes" must never approve this newer question (V1). agentd expires it unanswered.
            log.info("gate: a second confirmation in turn %d stays unbound", turn)
            return
        self._bound = _Bound(question_id, turn, summary)

    def _closed(self, question_id: str) -> None:
        if self._bound is not None and self._bound.question_id == question_id:
            self._bound.closed = True

    def _turn_ended(self, turn: int) -> None:
        if self._bound is not None and self._bound.turn == turn:
            self._bound = None

    def _drop(self) -> None:
        self._bound = None


# ----------------------------------------------------------------------------- brain


class RemoteBrain:
    """`contracts.Brain` for `voice:local` in agentd. Construct, `start()`, use as a Brain."""

    def __init__(self, cfg: Config, *, token: Callable[[], str | None], sock_path: Path | None = None,
                 on_notify: Callable[[str, str, str | None], Any] | None = None,
                 chrome_status_cb: Callable[[bool], None] | None = None, epoch: str | None = None,
                 on_rename: Callable[[str], Awaitable[Any]] | None = None,
                 on_reset_voice: Callable[[], Awaitable[Any]] | None = None,
                 on_go_to_sleep: Callable[[], Awaitable[Any]] | None = None,
                 client: str = CLIENT_NAME, backoff_s: tuple[float, ...] = BACKOFF_S) -> None:
        self.cfg = cfg
        self.epoch = epoch or secrets.token_hex(16)
        self.gate = RemoteGate(self)
        self.boot = ""
        self.connected = asyncio.Event()
        self._token = token
        self._sock = sock_path
        self._client = client
        self._backoff = backoff_s
        self._on_notify = on_notify
        self._chrome_cb = chrome_status_cb
        self._chrome: bool | None = None
        self._on_rename = on_rename  # mac.rename_assistant in agentd's brain → macd settings (AG15)
        self._on_reset_voice = on_reset_voice  # mac.reset_voice → macd's TTS (AG16)
        self._on_go_to_sleep = on_go_to_sleep  # mac.go_to_sleep → OFF after the reply (AG20)
        self._alive = False
        self._link: AgentLink | None = None
        self._task: asyncio.Task[None] | None = None
        self._kick = asyncio.Event()
        self._closing = False
        self._out: asyncio.Queue[BrainEvent | None] = asyncio.Queue()
        self._open_turns: set[int] = set()  # asked, no DONE/ERROR yet
        self._orphans: set[int] = set()  # ended here on a lost connection; interrupted on reconnect
        self._handoffs: list[asyncio.Future[bool]] = []
        self._checks: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._early_checks: dict[str, dict[str, Any]] = {}
        self._sent_settings: tuple[Any, ...] | None = None
        self._bg: set[asyncio.Task[Any]] = set()

    # -- Brain protocol ----------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._alive

    def _set_alive(self, value: bool) -> None:
        self._alive = bool(value)

    async def start(self) -> None:
        if self._task is None:
            self._closing = False
            self._task = asyncio.create_task(self._run(), name="remote-brain")

    async def close(self) -> None:
        self._closing = True
        self._kick.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._link is not None:
            await self._link.close()
            self._link = None
        for t in list(self._bg):
            t.cancel()
        self.connected.clear()
        await self._out.put(None)

    async def events(self) -> AsyncIterator[BrainEvent]:  # type: ignore[override]
        while True:
            ev = await self._out.get()
            if ev is None:
                return
            yield ev

    async def ask(self, turn_id: int, text: str) -> None:
        self._open_turns.add(turn_id)
        sent = False
        try:
            link = await self._ready(ASK_CONNECT_WAIT_S)
            if link is None:
                self._end_turn(turn_id, UNREACHABLE)
                return
            self._push_settings(link)
            fut = link.send(s.Ask(conv=CONV, turn=turn_id, text=text))
            sent = True
            _data(await asyncio.wait_for(asyncio.shield(fut), OP_TIMEOUT_S))
        except asyncio.CancelledError:
            if not sent:
                self._open_turns.discard(turn_id)
            raise
        except RemoteError as e:
            log.warning("ask refused by agentd: %s", e)
            self._end_turn(turn_id, BUSY if e.code == s.ErrorCode.BUSY.value else e.message or UNREACHABLE)
        except (ConnectionError, OSError, TimeoutError):
            self._end_turn(turn_id, UNREACHABLE)

    async def steer(self, turn_id: int, text: str) -> None:
        data = await self._call(s.Steer(conv=CONV, turn=turn_id, text=text))
        if data is not None and data.get("accepted") is False:
            log.info("steer for turn %d was not taken (the turn had finished)", turn_id)

    async def interrupt(self, turn_id: int) -> None:
        if turn_id:
            await self._call(s.Interrupt(conv=CONV, turn=turn_id))

    async def reset(self, hard: bool = False) -> None:
        await self._call(s.Reset(conv=CONV, hard=hard))

    async def warm(self) -> None:
        link = self._link
        if link is None:
            self._kick.set()  # retry the connection now (W6)
            return
        self._push_settings(link)
        await self._call(s.Warm(conv=CONV), timeout=30.0)  # status{alive} precedes the answer

    async def sleep(self) -> None:
        await self._call(s.Sleep(conv=CONV))

    async def summarize(self, prompt: str, timeout: float) -> str | None:
        """Voice never summarises a task session; agentd does that on its own brains."""
        return None

    async def handoff(self) -> bool:
        link = self._link
        if link is None:
            return False
        waiter: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._handoffs.append(waiter)
        try:
            _data(await asyncio.wait_for(asyncio.shield(link.send(s.Handoff(conv=CONV))), OP_TIMEOUT_S))
            return bool(await asyncio.wait_for(asyncio.shield(waiter), HANDOFF_WAIT_S))
        except (RemoteError, ConnectionError, OSError, TimeoutError) as e:
            log.warning("handoff failed: %s", e)
            return False
        finally:
            with contextlib.suppress(ValueError):
                self._handoffs.remove(waiter)

    async def connect_check(self) -> dict[str, Any]:
        link = self._link
        if link is None:
            return _check_error(UNREACHABLE)
        try:
            data = await link.call(s.ConnectCheck(conv=CONV))
            check_id = str(data.get("check_id", ""))
            early = self._early_checks.pop(check_id, None)
            if early is not None:
                return early
            waiter: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._checks[check_id] = waiter
            try:
                return await asyncio.wait_for(asyncio.shield(waiter), CONNECT_CHECK_WAIT_S)
            finally:
                self._checks.pop(check_id, None)
        except TimeoutError:
            return _check_error("The check took too long.")
        except (RemoteError, ConnectionError, OSError) as e:
            log.warning("connect_check failed: %s", e)
            return _check_error(UNREACHABLE)

    async def set_name(self, name: str) -> None:
        await self._call(s.SetName(conv=CONV, name=name))

    async def set_context(self, locked: bool) -> None:
        await self._call(s.SetContext(conv=CONV, locked=bool(locked)))

    async def set_language(self, lang: str) -> None:
        await self._call(s.SetLanguage(conv=CONV, lang=lang))

    async def stop_other(self) -> list[str]:
        """"Hey Mac, stop the other task": cancel the running chat turns (W6). Raises if agentd
        can't be reached, so the orchestrator can say so."""
        link = self._link
        if link is None:
            raise ConnectionError("not connected to agentd")
        data = await link.call(s.StopOther())
        return [str(c) for c in data.get("stopped", [])]

    # -- ops ---------------------------------------------------------------------------
    async def _call(self, op: s.Op, timeout: float = OP_TIMEOUT_S) -> dict[str, Any] | None:
        """Send on the live connection and wait for the answer; never raises (logs instead)."""
        link = self._link
        if link is None:
            log.info("agentd unreachable; %s skipped", op.OP)
            return None
        try:
            return await link.call(op, timeout)
        except (RemoteError, ConnectionError, OSError, TimeoutError) as e:
            log.warning("agentd %s failed: %s", op.OP, e)
            return None

    def _send_quiet(self, op: s.Op) -> None:
        """Write now, in order (V1); the answer is only logged. Without a connection agentd's own
        timeout closes the question, which fails closed."""
        link = self._link
        if link is None:
            log.warning("agentd unreachable; %s not sent", op.OP)
            return
        fut = link.send(op)

        def done(f: asyncio.Future[dict[str, Any]]) -> None:
            exc = f.exception() if not f.cancelled() else None
            if exc is not None:
                log.warning("agentd %s failed: %s", op.OP, exc)
            elif not f.cancelled() and f.result().get("ok") is not True:
                log.warning("agentd %s: %s", op.OP, f.result().get("code"))

        fut.add_done_callback(done)

    def _settings(self) -> tuple[Any, ...]:
        b = self.cfg.brain
        return (b.model, bool(b.chrome), tuple(b.setting_sources), self.cfg.name, self.cfg.tts.lang)

    def _push_settings(self, link: AgentLink, *, force: bool = False) -> None:
        """voice_settings (W0): macd's [brain] values, after every hello and whenever they change."""
        cur = self._settings()
        if not force and cur == self._sent_settings:
            return
        model, chrome, sources, name, lang = cur
        self._sent_settings = cur
        self._send_quiet_on(link, s.VoiceSettings(model=model, chrome=chrome, setting_sources=list(sources),
                                                  name=name, language=lang))

    def _send_quiet_on(self, link: AgentLink, op: s.Op) -> None:
        fut = link.send(op)
        fut.add_done_callback(lambda f: None if f.cancelled() else f.exception())

    async def _ready(self, wait_s: float) -> AgentLink | None:
        if self._link is None:
            self._kick.set()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.connected.wait(), wait_s)
        return self._link

    def _spawn(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self._bg.add(t)

        def done(task: asyncio.Task[Any]) -> None:
            self._bg.discard(task)
            if not task.cancelled() and task.exception() is not None:
                log.warning("remote brain callback failed: %r", task.exception())

        t.add_done_callback(done)

    # -- connection ----------------------------------------------------------------------
    async def _run(self) -> None:
        attempt = 0
        warned = False
        while not self._closing:
            token = self._token()
            if token:
                link = AgentLink(self._sock or agent_sock_path(), self._on_event)
                try:
                    await self._connect(link, token)
                except RemoteError as e:
                    await link.close()
                    if e.code == s.ErrorCode.VERSION_MISMATCH.value:
                        log.error("agentd: %s", e.message)
                    else:
                        log.warning("agentd refused the voice client: %s", e)
                except (OSError, ConnectionError, TimeoutError) as e:
                    await link.close()
                    if not warned:
                        log.warning("agentd unreachable (%s); retrying", e)
                        warned = True
                else:
                    attempt, warned = 0, False
                    await self._serve(link)
                    if self._closing:
                        return
                    log.warning("lost the connection to agentd; reconnecting")
            elif not warned:
                log.warning("no voice token for agentd yet; retrying")
                warned = True
            delay = self._backoff[min(attempt, len(self._backoff) - 1)]
            attempt += 1
            self._kick.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._kick.wait(), delay)

    async def _connect(self, link: AgentLink, token: str) -> None:
        hello = await link.open(token, self.epoch, [CONV], self._client)
        self.boot = str(hello.get("boot", ""))
        for item in hello.get("interrupted") or []:
            log.info("agentd restarted during a voice turn: %r", item.get("text", ""))
        self._push_settings(link, force=True)
        await link.call(s.Subscribe(conv=CONV))
        status = await link.call(s.Status(conv=CONV))
        self._alive = bool(status.get("alive", False))
        for turn in sorted(self._orphans):  # nobody listens to those any more (V2: no headless voice)
            self._send_quiet_on(link, s.Interrupt(conv=CONV, turn=turn))
        self._orphans.clear()
        self._link = link
        self.connected.set()
        log.info("connected to agentd (boot %s)", self.boot)

    async def _serve(self, link: AgentLink) -> None:
        try:
            await link.closed.wait()
        finally:
            if self._link is link:
                self._link = None
            self.connected.clear()
            await link.close()
            self._lost()

    def _lost(self) -> None:
        """The connection is gone: every open turn ends now, in plain words (W6)."""
        self.gate._drop()
        self._alive = False
        for turn in sorted(self._open_turns):
            self._orphans.add(turn)
            self._end_turn(turn, UNREACHABLE)
        for w in self._handoffs:
            if not w.done():
                w.set_result(False)
        for w in self._checks.values():
            if not w.done():
                w.set_result(_check_error(UNREACHABLE))

    def _end_turn(self, turn_id: int, message: str) -> None:
        if turn_id not in self._open_turns:
            return
        self._open_turns.discard(turn_id)
        self.gate._turn_ended(turn_id)
        self._out.put_nowait(BrainEvent(BrainEventKind.ERROR, turn_id, text=message, is_error=True,
                                        error_kind=BrainErrorKind.INTERNAL))

    # -- events --------------------------------------------------------------------------
    def _on_event(self, obj: dict[str, Any]) -> None:
        try:
            ev = s.decode_event(obj)
        except s.SchemaError as e:
            log.warning("agentd event skipped: %s", e)
            return
        if ev.conv != CONV:
            return
        out = self._translate(ev)
        if out is not None and out.kind is BrainEventKind.CONFIRM and isinstance(obj.get("question"), str):
            out.data["question"] = obj["question"]  # the model's action question (AG10, AG18)
        if out is not None:
            self._out.put_nowait(out)

    def _translate(self, ev: s.Event) -> BrainEvent | None:
        K = BrainEventKind
        if isinstance(ev, s.TextEvent):
            return BrainEvent(K.TEXT, ev.turn, text=ev.text, data={"tool_follows": True} if ev.tool_follows else {})
        if isinstance(ev, s.SayEvent):
            return BrainEvent(K.SAY, ev.turn, text=ev.text)
        if isinstance(ev, s.ToolStartEvent):
            return BrainEvent(K.TOOL_START, ev.turn, tool_name=ev.tool, tool_use_id=ev.tool_use_id,
                              data={"narration": ev.narration} if ev.narration else {})
        if isinstance(ev, s.ToolEndEvent):
            return BrainEvent(K.TOOL_END, ev.turn, tool_name=ev.tool, tool_use_id=ev.tool_use_id,
                              is_error=ev.is_error)
        if isinstance(ev, s.ConfirmEvent):
            self.gate._offer(ev.question_id, ev.turn, ev.summary)
            return BrainEvent(K.CONFIRM, ev.turn, text=ev.summary,
                              data={"target": ev.target, "question_id": ev.question_id})
        if isinstance(ev, s.QuestionClosedEvent):
            self.gate._closed(ev.question_id)
            return None
        if isinstance(ev, s.AttachEvent):
            return BrainEvent(K.ATTACH, ev.turn, text=ev.path,
                              data={"attach_id": ev.attach_id, "path": ev.path, "caption": ev.caption})
        if isinstance(ev, s.StatusEvent):
            return self._status(ev)
        if isinstance(ev, s.DoneEvent):
            self._open_turns.discard(ev.turn)
            self.gate._turn_ended(ev.turn)
            data: dict[str, Any] = {"interrupted": bool(ev.data.interrupted), "duration_s": ev.data.duration_s}
            if ev.data.cost is not None:
                data["cost_usd"] = ev.data.cost
            return BrainEvent(K.DONE, ev.turn, data=data)
        if isinstance(ev, s.ErrorEvent):
            if ev.turn is None:
                log.warning("agentd error without a turn: %s", ev.message)
                return None
            self._open_turns.discard(ev.turn)
            self.gate._turn_ended(ev.turn)
            kind = _ERROR_KINDS.get(ev.kind.value, BrainErrorKind.INTERNAL)
            # agentd fills an empty brain message with a generic line; the orchestrator's one
            # table (errors.MESSAGES) says it better per kind.
            text = "" if ev.message == _AGENTD_GENERIC_ERROR else ev.message
            return BrainEvent(K.ERROR, ev.turn, text=text, is_error=True, error_kind=kind)
        if isinstance(ev, s.InterruptedEvent):
            log.info("agentd: an earlier voice turn was interrupted by a restart (%r)", ev.text)
        return None

    def _status(self, ev: s.StatusEvent) -> BrainEvent | None:
        data = dict(ev.data)
        if "handoff_done" in data:
            ok = bool((data["handoff_done"] or {}).get("ok")) if isinstance(data["handoff_done"], dict) else False
            waiter = next((w for w in self._handoffs if not w.done()), None)
            if waiter is not None:
                waiter.set_result(ok)
            return None
        if "connect_check" in data:
            res = dict(data["connect_check"]) if isinstance(data["connect_check"], dict) else {}
            check_id = str(res.pop("check_id", ""))
            w = self._checks.get(check_id)
            if w is not None and not w.done():
                w.set_result(res)
            else:
                self._early_checks[check_id] = res
                while len(self._early_checks) > 8:
                    self._early_checks.pop(next(iter(self._early_checks)))
            return None
        if "alive" in data:  # agentd pushes liveness changes; not a HUD status
            self._alive = bool(data.pop("alive"))
        rename = data.pop("rename", None)
        if isinstance(rename, dict):
            rename = rename.get("name")
        if isinstance(rename, str) and rename.strip() and self._on_rename is not None:
            self._spawn(self._on_rename(rename.strip()))
        if data.pop("reset_voice", None) is not None and self._on_reset_voice is not None:
            self._spawn(self._on_reset_voice())
        if data.pop("go_to_sleep", None) is not None and self._on_go_to_sleep is not None:
            self._spawn(self._on_go_to_sleep())
        notify = data.pop("notify", None)
        if isinstance(notify, dict) and self._on_notify is not None:
            try:
                self._on_notify(str(notify.get("title", "Mac")), str(notify.get("body", "")), notify.get("path"))
            except Exception:
                log.exception("on_notify failed")
        if "chrome_connected" in data and isinstance(data["chrome_connected"], bool):
            connected = data["chrome_connected"]
            if connected != self._chrome and self._chrome_cb is not None:
                try:
                    self._chrome_cb(connected)
                except Exception:
                    log.exception("chrome_status_cb failed")
            self._chrome = connected
        if not data:
            return None
        return BrainEvent(BrainEventKind.STATUS, ev.turn or 0, data=data)


def _check_error(detail: str) -> dict[str, Any]:
    return {"auth": "error", "auth_detail": detail, "chrome": "unknown", "chrome_detail": "",
            "cli_version": "", "session_ok": False}
