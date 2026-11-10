"""ConversationService: conversations, turns, records, questions and profiles (W2.4, W3).

Every request is interpreted from (request, client identity, stores). In memory there is only
work in flight: running turns, brains, the desktop lock, outward confirmations and the replay
buffers. Losing it fails closed: no confirmation survives a restart, and a turn that was running
is announced once as `interrupted`.

Turn identity (V2, B1): a client turn `(client_epoch, turn)` maps to a fresh internal turn int for
the brain, and `turn_uid = boot:conv:epoch:turn`. Voice events carry the client's turn number and
are filtered by epoch at delivery; chat streams are per conversation.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd import schema as s
from agentd.brains import (
    CHAT,
    VOICE,
    BrainFactory,
    BrainSpec,
    Profile,
    RecordSessionStore,
    SessionInfo,
    close_quietly,
)
from agentd.breaker import PlanBreaker
from agentd.desktop import DesktopLock
from agentd.questions import Binding, Question, bind
from agentd.records import ConversationRecord, OpenQuestion, RunningTurn
from agentd.router import Route, Router, RouterModel
from agentd.stores import JsonStore
from agentd.stream import GapError, LagMonitor, Stream
from agentd.workspaces import (
    CHAT_APPROVAL_MAX_BYTES,
    WorkspaceStore,
    approval_text,
    display,
    is_skipped,
    resolve_user_path,
)
from macd.brain.gate import ConfirmationGate
from macd.contracts import Brain, BrainEvent, BrainEventKind

log = logging.getLogger(__name__)

SERVER_NAME = "agentd/0.9.0"
HANDOFF_TIMEOUT_S = 60.0
CONNECT_CHECK_TIMEOUT_S = 25.0
VOICE_GRACE_S = 3.0  # a voice connection gone this long interrupts its running turn (V2)
CATCH_UP_AFTER_S = 3600.0
QUESTION_TTL_S = {s.QuestionKind.FOLDER_APPROVAL: 600.0, s.QuestionKind.ROUTE: 3600.0,
                  s.QuestionKind.CATCH_UP: 3600.0, s.QuestionKind.RESTART: 3600.0}
NEVER_CHAT_NAMES = ("Timetable",)
RESTART_TEXT = "My brain on the Mac restarted, so I stopped. Want me to pick it up?"
AGENTD_EPOCH = "agentd"  # epoch of turns agentd starts itself (a "yes" to a restart question)


class ApiError(Exception):
    def __init__(self, code: s.ErrorCode, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class AgentPaths:
    home: Path  # the user's home
    mac_home: Path  # ~/.mac-agent
    log_dir: Path  # ~/Library/Logs/Mac/agent

    @classmethod
    def default(cls) -> AgentPaths:
        from macd import config

        return cls(Path.home(), Path(config.HOME), Path(config.LOG_DIR) / "agent")

    @property
    def state_dir(self) -> Path:
        return self.mac_home / "agent" / "state"

    @property
    def sock(self) -> Path:
        return self.mac_home / "run" / "agent.sock"

    @property
    def chat_config(self) -> Path:
        """Fixed (api §3): never taken from a hello or a request."""
        return self.mac_home / "chat" / "config.toml"

    @property
    def voice_cwd(self) -> Path:
        return self.mac_home / "workspace"

    @property
    def chat_cwd(self) -> Path:
        return self.mac_home / "agent" / "chat-workspace"

    @property
    def memory(self) -> Path:
        return self.mac_home / "workspace" / "memory.md"


@dataclass
class VoiceSettings:
    model: str = ""
    chrome: bool = True
    setting_sources: list[str] = field(default_factory=lambda: ["user"])
    name: str = "Mac"
    language: str = "en-us"


@dataclass(eq=False)
class Turn:
    internal: int
    uid: str
    epoch: str
    client_turn: int
    text: str
    started: float  # monotonic
    msg_ref: str = ""
    task: asyncio.Task[None] | None = None
    in_brain: bool = False
    stopping: bool = False
    finished: bool = False
    answer: list[str] = field(default_factory=list)
    attaches: list[dict[str, Any]] = field(default_factory=list)
    steers: list[str] = field(default_factory=list)  # arrived before the brain started the turn
    tool_ids: dict[str, list[str]] = field(default_factory=dict)
    n_tools: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Conversation:
    def __init__(self, svc: Service, record: ConversationRecord) -> None:
        self.svc = svc
        self.conv = record.conv
        self.record = record
        self.profile: Profile = VOICE if record.conv == s.VOICE_CONV else CHAT
        self.voice = self.profile is VOICE
        self.stream = Stream(self.conv, svc.boot)
        self.gate = ConfirmationGate()
        self.brain: Brain | None = None
        self.brain_key: tuple[Any, ...] | None = None
        self._pump: asyncio.Task[None] | None = None
        self.turn: Turn | None = None  # the newest turn
        self.turns: dict[int, Turn] = {}  # internal id -> turn, until finished
        self.uids: set[str] = set()
        self.question: Question | None = None
        self._q_waiters: dict[str, asyncio.Future[Binding | str]] = {}
        self._q_timer: asyncio.TimerHandle | None = None
        self.recent: collections.deque[str] = collections.deque(maxlen=10)
        self.locked = False
        self.tasks: set[asyncio.Task[Any]] = set()
        self.desktop = svc.desktop.handle(self.conv, self.profile.desktop_label, self._running_text)

    # -- helpers -------------------------------------------------------------------------
    def _running_text(self) -> str:
        return self.turn.text[:80] if self.turn else ""

    def spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task[Any]:
        t = asyncio.ensure_future(coro)
        t.set_name(f"{self.conv}:{name}")
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    def emit(self, cls: type[s.Event], turn: Turn | None = None, **fields: Any) -> None:
        epoch = None
        if turn is not None:
            fields.update(turn=turn.client_turn, turn_uid=turn.uid)
            epoch = turn.epoch if self.voice else None
        self.stream.emit(cls, epoch=epoch, **fields)

    def running(self) -> Turn | None:
        t = self.turn
        return t if t is not None and not t.finished else None

    # -- brain ---------------------------------------------------------------------------
    def _session_get(self) -> SessionInfo | None:
        r = self.record
        if not r.session_id:
            return None
        return SessionInfo(r.session_id, r.session_updated_at, r.session_started_at, r.session_fork)

    def _session_put(self, info: SessionInfo) -> None:
        r = self.record
        r.session_id = info.session_id
        r.session_updated_at = info.updated_at
        r.session_started_at = info.started_at
        r.session_fork = False
        if info.session_id:
            r.remember_own(info.session_id)
        self.svc.save_soon()

    async def ensure_brain(self, cwd: Path, sources: list[str], *, rebuild: bool = False) -> Brain:
        vs = self.svc.voice_settings
        spec = BrainSpec(
            conv=self.conv, profile=self.profile, cwd=cwd, setting_sources=list(sources), gate=self.gate,
            session_store=RecordSessionStore(self._session_get, self._session_put),
            memory_file=self.svc.memory_file, desktop=self.desktop, recent_user_text=lambda: list(self.recent),
            model=vs.model if self.voice else "", chrome=vs.chrome if self.voice else True,
            name=vs.name, language=vs.language, never_chat_names=NEVER_CHAT_NAMES)
        busy = any(t.in_brain and not t.finished for t in self.turns.values())
        if self.brain is not None and (busy or (not rebuild and spec.key() == self.brain_key)):
            return self.brain  # never swap a brain under a running turn
        await self.close_brain()
        await asyncio.to_thread(cwd.mkdir, parents=True, exist_ok=True)
        brain = self.svc.brain_factory(spec)
        await brain.start()
        self.brain, self.brain_key = brain, spec.key()
        self._pump = asyncio.create_task(self._pump_events(brain), name=f"{self.conv}:pump")
        if self.locked:
            await brain.set_context(True)
        if self.voice:
            await brain.set_name(vs.name)
            await brain.set_language(vs.language)
        log.info("conv %s: brain started (sources=%s)", self.conv, ",".join(sources))
        return brain

    async def close_brain(self) -> None:
        brain, pump = self.brain, self._pump
        self.brain, self.brain_key, self._pump = None, None, None
        if pump is not None:  # stop relaying first: nothing a closing brain says is applied
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
        if brain is not None:
            await close_quietly(brain)

    async def _pump_events(self, brain: Brain) -> None:
        try:
            async for ev in brain.events():
                try:
                    await self._on_event(ev)
                except Exception:
                    log.exception("conv %s: event handling failed", self.conv)
        finally:
            for t in list(self.turns.values()):  # the brain ended under a running turn
                if t.in_brain and not t.finished and self.brain is brain:
                    self.emit(s.ErrorEvent, t, kind=s.EventErrorKind.INTERNAL,
                              message="Mac's brain stopped unexpectedly.")
                    await self.finish(t, error=True)

    async def _on_event(self, ev: BrainEvent) -> None:
        t = self.turns.get(ev.turn_id)
        if t is None or t.finished:
            return
        k = ev.kind
        if k is BrainEventKind.TEXT:
            follows = ev.data.get("tool_follows")
            if ev.text and not follows:
                t.answer.append(ev.text)
            if ev.text or follows:
                self.emit(s.TextEvent, t, text=ev.text, tool_follows=True if follows else None)
        elif k is BrainEventKind.SAY:
            self.emit(s.SayEvent, t, text=ev.text)
        elif k is BrainEventKind.TOOL_START:
            t.n_tools += 1
            t.answer.clear()  # the text so far was a step line
            tid = ev.tool_use_id or f"{t.internal}-{t.n_tools}"
            t.tool_ids.setdefault(ev.tool_name, []).append(tid)
            narration = ev.data.get("narration") if self.voice else None
            self.emit(s.ToolStartEvent, t, tool=ev.tool_name, tool_use_id=tid, narration=narration)
        elif k is BrainEventKind.TOOL_END:
            ids = t.tool_ids.get(ev.tool_name) or []
            tid = ev.tool_use_id or (ids.pop(0) if ids else f"{t.internal}-end")
            if ev.tool_use_id and ev.tool_use_id in ids:
                ids.remove(ev.tool_use_id)
            self.emit(s.ToolEndEvent, t, tool=ev.tool_name, tool_use_id=tid, is_error=ev.is_error)
        elif k is BrainEventKind.CONFIRM:
            self.open_confirm(t, ev.text, str(ev.data.get("target", "")))
        elif k is BrainEventKind.ATTACH:
            await self._attach(t, ev)
        elif k is BrainEventKind.STATUS:
            data = dict(ev.data)
            if isinstance(data.get("session_title"), str):
                self.record.session_title = data["session_title"]
            if data:
                self.emit(s.StatusEvent, t, data=data)
        elif k is BrainEventKind.DONE:
            self.emit(s.DoneEvent, t, data=s.DoneData(
                duration_s=round(time.monotonic() - t.started, 3),
                interrupted=bool(ev.data.get("interrupted")) or t.stopping or None,
                cost=ev.data.get("cost_usd", ev.data.get("cost"))))
            await self.finish(t, interrupted=bool(ev.data.get("interrupted")) or t.stopping)
        elif k is BrainEventKind.ERROR:
            kind = ev.error_kind.value if ev.error_kind else "internal"
            if kind == "rate_limit":
                self.svc.breaker.trip(ev.data.get("reset_at") or ev.data.get("resets_at"))
            self.emit(s.ErrorEvent, t, kind=s.EventErrorKind(kind), message=ev.text or "Something went wrong.")
            await self.finish(t, error=True)

    async def _attach(self, t: Turn, ev: BrainEvent) -> None:
        path = str(ev.data.get("path") or ev.text)
        ok = await asyncio.to_thread(os.path.isfile, path)  # attach path check off the loop (V7)
        if not ok:
            log.warning("conv %s: attach of a missing file dropped", self.conv)
            return
        item = {"attach_id": str(ev.data.get("attach_id") or os.urandom(8).hex()), "path": path,
                "caption": str(ev.data.get("caption", ""))}
        t.attaches.append(item)
        self.emit(s.AttachEvent, t, **item)

    # -- turn lifecycle --------------------------------------------------------------------
    async def finish(self, t: Turn, *, interrupted: bool = False, error: bool = False) -> None:
        if t.finished:
            return
        t.finished = True
        self.turns.pop(t.internal, None)
        q = self.question
        if q is not None and q.turn_internal == t.internal:
            self.close_question(q, s.CloseOutcome.EXPIRED if q.kind is s.QuestionKind.CONFIRM
                                else s.CloseOutcome.SKIPPED)
        self.svc.desktop.cancel_turn(self.conv)
        r = self.record
        if r.running is not None and r.running.uid == t.uid:
            r.running = None
        if not self.voice:  # voice turns never enter unacked
            r.add_unacked({"turn_uid": t.uid, "answer": "".join(t.answer).strip(), "attaches": t.attaches})
        if not interrupted and not error:
            r.last_answer_at = self.svc.clock()
        if self.turn is t:
            self.turn = None
        t.done.set()
        await self.svc.persist()

    async def run_turn(self, t: Turn, meta: s.AskMeta | None, before: Turn | None) -> None:
        try:
            if before is not None:
                await before.done.wait()  # a stopped turn drains first (one turn at a time)
            if self.svc.breaker.is_open():
                self.emit(s.ErrorEvent, t, kind=s.EventErrorKind.RATE_LIMIT, message=self.svc.breaker.message())
                await self.finish(t, error=True)
                return
            if self.voice:
                await self.svc.ensure_voice_brain(self)
            elif not await self._prepare_chat(t, meta):
                return
            if t.stopping:
                raise asyncio.CancelledError
            assert self.brain is not None
            t.in_brain = True
            await self.brain.ask(t.internal, t.text)
            for text in t.steers:
                await self.brain.steer(t.internal, text)
            await t.done.wait()
        except asyncio.CancelledError:
            if self.svc.closing:
                raise  # shutdown: the record keeps `running`, announced as interrupted next boot
            if not t.finished and not t.in_brain:
                self.emit(s.DoneEvent, t, data=s.DoneData(duration_s=round(time.monotonic() - t.started, 3),
                                                          interrupted=True))
                await self.finish(t, interrupted=True)
        except Exception:
            log.exception("conv %s: turn failed", self.conv)
            if not t.finished:
                self.emit(s.ErrorEvent, t, kind=s.EventErrorKind.INTERNAL,
                          message="Something went wrong on my side, so I stopped.")
                await self.finish(t, error=True)

    async def _prepare_chat(self, t: Turn, meta: s.AskMeta | None) -> bool:
        svc, r = self.svc, self.record
        now = svc.clock()
        if meta is not None and meta.replayed and meta.sent_at and now - meta.sent_at > CATCH_UP_AFTER_S:
            when = time.strftime("%I:%M %p", time.localtime(meta.sent_at)).lstrip("0")
            ans = await self.ask_question(t, s.QuestionKind.CATCH_UP,
                                          f'Back online. You asked at {when} while I was asleep: '
                                          f'"{_short(t.text)}". Still want it?')
            if not (isinstance(ans, Binding) and ans.choice == 0):
                raise asyncio.CancelledError
        current = Path(r.folder) if r.folder else None
        route: Route = await svc.router.route(t.text, folder=current, last_answer_at=r.last_answer_at,
                                              fresh=r.fresh_next, named=r.fresh_folder)
        folder, session = route.folder, route.session
        if route.question:
            opts = [{"label": o.label, "keys": o.keys} for o in route.options]
            ans = await self.ask_question(t, s.QuestionKind.ROUTE, route.question, {"options": opts})
            if not isinstance(ans, Binding) or ans.choice is None:
                raise asyncio.CancelledError
            folder, session = route.options[ans.choice].folder, route.options[ans.choice].session
        r.fresh_next, r.fresh_folder = False, None
        fresh_session, prev_sid = False, r.session_id
        if session == "same" and folder == current:
            pass
        elif session.startswith("continue:"):
            sid = session.split(":", 1)[1]
            r.session_id, r.session_fork = sid, sid not in r.own_sessions
            r.session_updated_at = r.session_started_at = 0.0
        else:
            r.session_id, fresh_session = None, True
        r.folder = str(folder) if folder else None
        if r.running is not None and r.running.uid == t.uid:
            r.running.folder = display(folder, svc.paths.home)
        sources = ["user"]
        if folder is not None:  # hash re-checked at every turn start (W3.5)
            trust = await svc.workspaces.trust(folder)
            digest = trust.scan.digest
            if (trust.needs_approval and not svc.workspaces.was_declined(folder, digest)
                    and trust.scan.claude_bytes <= CHAT_APPROVAL_MAX_BYTES):
                ans = await self.ask_question(t, s.QuestionKind.FOLDER_APPROVAL,
                                              approval_text(folder.name, trust.scan),
                                              {"path": str(folder), "hash": digest})
                if isinstance(ans, Binding) and ans.outcome is s.ReplyOutcome.APPROVED:
                    await svc.workspaces.approve(folder, digest)  # pinned to the hash shown
                elif isinstance(ans, Binding):
                    await svc.workspaces.decline(folder, digest)
                trust = await svc.workspaces.trust(folder)
            sources = trust.setting_sources
        await svc.persist()
        self.emit(s.StatusEvent, t, data={"folder": display(folder, svc.paths.home) or ""})
        existing = self.brain
        rebuild = session.startswith("continue:") and r.session_id != prev_sid
        brain = await self.ensure_brain(folder or svc.paths.chat_cwd, sources, rebuild=rebuild)
        if fresh_session and brain is existing:
            await brain.reset()  # a reused brain may hold an older session
        return True

    async def interrupt(self, t: Turn | None) -> bool:
        if t is None or t.finished:
            return False
        if t.stopping:
            return True
        t.stopping = True
        self.svc.desktop.cancel_turn(self.conv)
        if t.in_brain and self.brain is not None:
            await self.brain.interrupt(t.internal)
        elif t.task is not None:
            t.task.cancel()
        return True

    # -- questions -----------------------------------------------------------------------
    def _set_timer(self, q: Question) -> None:
        if self._q_timer is not None:
            self._q_timer.cancel()
        delay = max(0.0, q.expires_at - self.svc.clock())
        self._q_timer = asyncio.get_running_loop().call_later(delay, self._expire, q.question_id)

    def _expire(self, qid: str) -> None:
        self._q_timer = None
        q = self.question
        if q is not None and q.question_id == qid:
            self.close_question(q, s.CloseOutcome.EXPIRED)

    def _replace(self) -> None:
        if self.question is not None:
            self.close_question(self.question, s.CloseOutcome.SKIPPED)

    def open_confirm(self, t: Turn, summary: str, target: str) -> None:
        self._replace()
        q = Question(_qid(), s.QuestionKind.CONFIRM, summary,
                     self.svc.clock() + self.profile.confirm_window_s + 1.0,
                     payload={"target": target}, turn_internal=t.internal)
        self.question = q
        self._set_timer(q)
        self.svc.desktop.release_conv(self.conv)  # never hold the desktop while confirming (F3)
        self.emit(s.ConfirmEvent, t, question_id=q.question_id, summary=summary, target=target)

    def open_question(self, kind: s.QuestionKind, text: str, payload: dict[str, Any] | None = None,
                      turn: Turn | None = None) -> Question:
        self._replace()
        q = Question(_qid(), kind, text, self.svc.clock() + QUESTION_TTL_S[kind], payload=payload or {},
                     turn_internal=turn.internal if turn else 0)
        self.question = q
        self.record.question = OpenQuestion(q.question_id, kind.value, text, q.expires_at, "", q.payload)
        self._set_timer(q)
        self.svc.save_soon()
        self.emit(s.QuestionEvent, question_id=q.question_id, kind=kind, text=text, expires_at=q.expires_at)
        return q

    async def ask_question(self, t: Turn, kind: s.QuestionKind, text: str,
                           payload: dict[str, Any] | None = None) -> Binding | str:
        q = self.open_question(kind, text, payload, t)
        fut: asyncio.Future[Binding | str] = asyncio.get_running_loop().create_future()
        self._q_waiters[q.question_id] = fut
        try:
            return await fut
        finally:
            self._q_waiters.pop(q.question_id, None)
            if self.question is q:  # the turn was stopped while asking
                self.close_question(q, s.CloseOutcome.SKIPPED)

    def close_question(self, q: Question, outcome: s.CloseOutcome, binding: Binding | None = None) -> None:
        if self.question is not q:
            return
        self.question = None
        if self._q_timer is not None:
            self._q_timer.cancel()
            self._q_timer = None
        if q.kind is not s.QuestionKind.CONFIRM:
            self.record.question = None
            self.svc.save_soon()
        fut = self._q_waiters.get(q.question_id)
        if fut is not None and not fut.done():
            fut.set_result(binding if binding is not None else outcome.value)
        self.emit(s.QuestionClosedEvent, question_id=q.question_id, outcome=outcome)

    # -- status --------------------------------------------------------------------------
    def status(self) -> s.ConversationStatus:
        r, svc = self.record, self.svc
        running = None
        if r.running is not None:
            running = s.RunningStatus(uid=r.running.uid, text=r.running.text, folder=r.running.folder,
                                      minutes=int(max(0.0, svc.clock() - r.running.started_at) // 60))
        q = self.question
        question = s.QuestionStatus(question_id=q.question_id, kind=q.kind, text=q.text,
                                    expires_at=q.expires_at) if q else None
        unacked = [] if self.voice else [
            s.UnackedTurn(turn_uid=u["turn_uid"], answer=u.get("answer", ""),
                          attaches=[s.AttachRef(**a) for a in u.get("attaches", [])]) for u in r.unacked]
        return s.ConversationStatus(
            conv=self.conv, alive=bool(self.brain is not None and getattr(self.brain, "alive", False)),
            folder=display(r.folder, svc.paths.home), session_title=r.session_title, paused=r.paused,
            running=running, question=question, desktop=svc.desktop.status(), unacked=unacked,
            loop_lag_ms_p99=svc.lag.p99())


def _qid() -> str:
    return os.urandom(8).hex()


def _short(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


class _NullEvents:
    def emit(self, *a: Any, **k: Any) -> bool:
        return True


class Service:
    def __init__(self, paths: AgentPaths, brain_factory: BrainFactory, *,
                 router_model: RouterModel | None = None, router: Router | None = None,
                 workspaces: WorkspaceStore | None = None, memory_file: Any = None,
                 clock: Callable[[], float] = time.time, boot: str | None = None,
                 voice_grace_s: float = VOICE_GRACE_S, events: Any = None,
                 lag_interval_s: float = 0.1) -> None:
        self.paths = paths
        self.brain_factory = brain_factory
        self.clock = clock
        self.boot = boot or os.urandom(8).hex()
        self.voice_grace_s = voice_grace_s
        self.events = events or _NullEvents()
        self.breaker = PlanBreaker(clock=clock)
        self.desktop = DesktopLock()
        self.lag = LagMonitor(lag_interval_s)
        self.workspaces = workspaces or WorkspaceStore(paths.state_dir / "workspaces.json", paths.home)
        self.router = router or Router(self.workspaces, router_model, clock=clock)
        self.memory_file = memory_file
        self.voice_settings = VoiceSettings()
        self.convs: dict[str, Conversation] = {}
        self._store = JsonStore(paths.state_dir / "conversations.json")
        self._notices: dict[str, list[s.InterruptedTurn]] = {}
        self._next_internal = 0
        self._save_pending = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._voice_epoch: str | None = None
        self._voice_conns: collections.Counter[str] = collections.Counter()
        self._voice_timer: asyncio.TimerHandle | None = None
        self._lag_task: asyncio.Task[None] | None = None
        self.closing = False

    # -- lifecycle -----------------------------------------------------------------------
    async def start(self) -> None:
        await self.workspaces.load()
        raw = await self._store.load({})
        for conv, d in (raw.get("conversations", {}) if isinstance(raw, dict) else {}).items():
            if isinstance(d, dict):
                with contextlib.suppress(TypeError, ValueError):
                    rec = ConversationRecord.from_dict({**d, "conv": conv})
                    self.convs[conv] = Conversation(self, rec)
        for c in self.convs.values():
            self._recover(c)
        await self.persist()  # notices are marked before any client can connect (W2.4)
        self._lag_task = asyncio.create_task(self.lag.run(), name="agentd:lag")

    def _recover(self, c: Conversation) -> None:
        r = c.record
        if r.question is not None:  # a restart never approves (M5)
            c.emit(s.QuestionClosedEvent, question_id=r.question.question_id, outcome=s.CloseOutcome.EXPIRED)
            r.question = None
        if r.running is None:
            return
        run, r.running = r.running, None
        if r.restart_notice_for == run.uid:
            return
        r.restart_notice_for = run.uid
        self._notices.setdefault(c.conv, []).append(s.InterruptedTurn(conv=c.conv, uid=run.uid, text=run.text))
        c.emit(s.InterruptedEvent, uid=run.uid, text=run.text)
        if not c.voice:
            c.open_question(s.QuestionKind.RESTART, RESTART_TEXT, {"text": run.text, "msg_ref": run.msg_ref})

    async def close(self) -> None:
        self.closing = True
        if self._lag_task is not None:
            self._lag_task.cancel()
        if self._voice_timer is not None:
            self._voice_timer.cancel()
        for c in self.convs.values():
            for t in list(c.tasks):
                t.cancel()
            for t in list(c.turns.values()):
                if t.task is not None:
                    t.task.cancel()
            await c.close_brain()
        for t in list(self._tasks):
            t.cancel()
        await self.persist()

    def spawn(self, coro: Awaitable[Any]) -> None:
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def persist(self) -> None:
        await self._store.save(lambda: {"conversations": {k: c.record.to_dict() for k, c in self.convs.items()}})

    def save_soon(self) -> None:
        if self._save_pending:
            return
        self._save_pending = True

        async def run() -> None:
            self._save_pending = False
            await self.persist()

        self.spawn(run())

    def conversation(self, conv: str) -> Conversation:
        c = self.convs.get(conv)
        if c is None:
            c = self.convs[conv] = Conversation(self, ConversationRecord(conv=conv))
        return c

    # -- identity --------------------------------------------------------------------------
    async def chat_allowed(self, convs: list[str]) -> bool:
        """Every conv is `<channel>:<chat id>` with the id in that channel's `chats` and not in its
        `never_chats`, read from the fixed chatd config path (api §3)."""
        cfg = await asyncio.to_thread(_read_toml, self.paths.chat_config)
        channels = cfg.get("channels") if isinstance(cfg.get("channels"), dict) else {}
        for conv in convs:
            channel, sep, chat_id = conv.partition(":")
            sec = channels.get(channel) if sep and chat_id else None
            if not isinstance(sec, dict):
                return False
            chats, never = sec.get("chats") or [], sec.get("never_chats") or []
            if chat_id not in chats or chat_id in never:
                return False
        return True

    def take_notices(self, convs: list[str]) -> list[s.InterruptedTurn]:
        out: list[s.InterruptedTurn] = []
        for conv in convs:
            out.extend(self._notices.pop(conv, []))
        return out

    async def voice_connected(self, epoch: str) -> None:
        self._voice_conns[epoch] += 1
        if self._voice_timer is not None:
            self._voice_timer.cancel()
            self._voice_timer = None
        c = self.conversation(s.VOICE_CONV)
        if not c.record.voice_migrated:  # M3: one-time copy of the local-mode voice session
            await self._migrate_voice(c)
        if self._voice_epoch != epoch:
            self._voice_epoch = epoch
            t = c.running()
            if t is not None and t.epoch != epoch:
                log.info("voice: new client epoch; interrupting the old epoch's turn")
                await c.interrupt(t)

    def voice_disconnected(self, epoch: str) -> None:
        self._voice_conns[epoch] -= 1
        if self._voice_conns[epoch] > 0:
            return
        del self._voice_conns[epoch]
        loop = asyncio.get_running_loop()
        self._voice_timer = loop.call_later(self.voice_grace_s, lambda: self.spawn(self._voice_gone(epoch)))

    async def _voice_gone(self, epoch: str) -> None:
        self._voice_timer = None
        if self._voice_conns.get(epoch):
            return
        c = self.convs.get(s.VOICE_CONV)
        t = c.running() if c else None
        if c is not None and t is not None and t.epoch == epoch:
            log.info("voice: client gone for %.0f s; interrupting its turn (voice never runs headless)",
                     self.voice_grace_s)
            await c.interrupt(t)

    async def _migrate_voice(self, c: Conversation) -> None:
        from macd import config

        state = await asyncio.to_thread(config.read_state)
        b = state.get("brain") if isinstance(state, dict) else None
        r = c.record
        if isinstance(b, dict) and b.get("session_id") and not r.session_id:
            r.session_id = str(b["session_id"])
            r.session_updated_at = float(b.get("updated_at", 0) or 0)
            r.session_started_at = float(b.get("started_at", r.session_updated_at) or 0)
            r.remember_own(r.session_id)
        r.voice_migrated = True
        await self.persist()

    async def ensure_voice_brain(self, c: Conversation) -> Brain:
        return await c.ensure_brain(self.paths.voice_cwd, self.voice_settings.setting_sources)

    # -- turns ---------------------------------------------------------------------------
    def _new_turn(self, c: Conversation, epoch: str, client_turn: int, text: str, msg_ref: str) -> Turn:
        uid = s.turn_uid(self.boot, c.conv, epoch, client_turn)
        if uid in c.uids:
            raise ApiError(s.ErrorCode.BAD_ARGS, "That turn number was already used.")
        self._next_internal += 1
        t = Turn(self._next_internal, uid, epoch, client_turn, text, time.monotonic(), msg_ref)
        c.uids.add(uid)
        return t

    async def ask(self, epoch: str, op: s.Ask) -> s.AskData:
        c = self.conversation(op.conv)
        if c.record.paused:
            raise ApiError(s.ErrorCode.PAUSED, "This chat is paused. Unpause it in the Mac menu.")
        cur = c.running()
        if cur is not None and not cur.stopping:
            raise ApiError(s.ErrorCode.BUSY, "A task is already running here.")
        q = c.question
        if q is not None and q.turn_internal == 0:  # a free-standing question: the user moved on
            c.close_question(q, s.CloseOutcome.SKIPPED)
        meta = op.meta
        t = self._new_turn(c, epoch, op.turn, op.text, meta.msg_ref if meta else "")
        await self._start_turn(c, t, meta)
        return s.AskData(turn_uid=t.uid)

    async def _start_turn(self, c: Conversation, t: Turn, meta: s.AskMeta | None) -> None:
        before = c.turn if c.turn is not None and not c.turn.finished else None
        c.turn = t
        c.turns[t.internal] = t
        c.recent.append(t.text)
        c.record.running = RunningTurn(t.uid, t.text, self.clock(), t.msg_ref)
        await self.persist()
        self.events.emit("agentd.turn", conv=c.conv, turn_uid=t.uid)
        t.task = c.spawn(c.run_turn(t, meta, before), f"turn{t.internal}")

    def _find(self, c: Conversation, epoch: str, voice_turn: int | None, uid: str | None) -> Turn | None:
        t = c.running()
        if t is None:
            return None
        if c.voice:
            return t if t.epoch == epoch and t.client_turn == voice_turn else None
        return t if uid is None or t.uid == uid else None

    async def steer(self, epoch: str, op: s.Steer) -> s.AcceptedData:
        c = self.conversation(op.conv)
        t = self._find(c, epoch, op.turn, op.turn_uid)
        if t is None or t.stopping:
            return s.AcceptedData(accepted=False)
        c.recent.append(op.text)
        if not t.in_brain:
            t.steers.append(op.text)  # still routing or asking: delivered when the brain starts
            return s.AcceptedData(accepted=True)
        assert c.brain is not None
        res = await c.brain.steer(t.internal, op.text)
        return s.AcceptedData(accepted=res is not False)

    async def interrupt(self, epoch: str, op: s.Interrupt) -> s.StoppedData:
        c = self.conversation(op.conv)
        return s.StoppedData(stopped=await c.interrupt(self._find(c, epoch, op.turn, op.turn_uid)))

    async def cancel(self, conv: str) -> s.StoppedData:
        c = self.convs.get(conv)
        return s.StoppedData(stopped=await c.interrupt(c.running()) if c else False)

    async def stop_other(self) -> s.StopOtherData:
        stopped = []
        for conv, c in list(self.convs.items()):
            if conv != s.VOICE_CONV and await c.interrupt(c.running()):
                stopped.append(conv)
        return s.StopOtherData(stopped=stopped)

    # -- questions -------------------------------------------------------------------------
    def confirm(self, op: s.Confirm) -> s.ConfirmData:
        c = self.conversation(op.conv)
        q = c.question
        if q is None or q.kind is not s.QuestionKind.CONFIRM or q.question_id != op.question_id \
                or self.clock() >= q.expires_at:
            log.warning("confirm: stale question id; not applied")
            raise ApiError(s.ErrorCode.STALE_QUESTION, "That question is no longer open.")
        c.close_question(q, s.CloseOutcome.APPROVED if op.approved else s.CloseOutcome.DENIED)
        c.gate.resolve(op.approved)
        return s.ConfirmData(applied=True)

    def revoke_all(self, conv: str) -> None:
        c = self.conversation(conv)
        q = c.question
        if q is not None and q.kind is s.QuestionKind.CONFIRM:
            c.close_question(q, s.CloseOutcome.DENIED)
        c.gate.revoke_all()

    def question_posted(self, op: s.QuestionPosted) -> None:
        c = self.conversation(op.conv)
        q = c.question
        if q is None or q.question_id != op.question_id:
            return
        q.prompt_ref = op.msg_ref
        if c.record.question is not None and c.record.question.question_id == q.question_id:
            c.record.question.prompt_ref = op.msg_ref
            self.save_soon()

    async def reply(self, op: s.Reply) -> s.ReplyData:
        c = self.conversation(op.conv)
        q = c.question
        b = bind(q, op, self.clock())
        if not b.bound or q is None or b.outcome is None:
            return s.ReplyData(bound=False)
        if op.text:
            c.recent.append(op.text)
        outcome = s.CloseOutcome(b.outcome.value)
        if q.kind is s.QuestionKind.CONFIRM:
            c.close_question(q, outcome)
            c.gate.resolve(b.outcome is s.ReplyOutcome.APPROVED)
            t = c.turns.get(q.turn_internal)
            if b.outcome is s.ReplyOutcome.CHANGED and t is not None and t.in_brain and c.brain is not None:
                await c.brain.steer(t.internal, b.steer_text)  # "change it": deny, then steer (U11)
        elif q.kind is s.QuestionKind.RESTART:
            c.close_question(q, outcome, b)
            if b.outcome is s.ReplyOutcome.APPROVED and c.running() is None:
                t = self._new_turn(c, AGENTD_EPOCH, self._next_internal + 1, str(q.payload.get("text", "")),
                                   str(q.payload.get("msg_ref", "")))
                await self._start_turn(c, t, None)
        else:
            c.close_question(q, outcome, b)
        return s.ReplyData(bound=True, outcome=b.outcome)

    # -- long ops (B2): reply at once, result as a status event --------------------------
    def handoff(self, conv: str) -> s.AcceptedData:
        c = self.conversation(conv)

        async def run() -> None:
            ok = False
            try:
                if c.brain is not None:
                    ok = bool(await asyncio.wait_for(c.brain.handoff(), HANDOFF_TIMEOUT_S))
            except Exception:
                log.warning("handoff failed", exc_info=True)
            c.emit(s.StatusEvent, data={"handoff_done": {"ok": ok}})

        c.spawn(run(), "handoff")
        return s.AcceptedData(accepted=True)

    def connect_check(self, op: s.ConnectCheck) -> s.ConnectCheckData:
        c = self.conversation(op.conv)
        check_id = os.urandom(6).hex()

        async def run() -> None:
            try:
                brain = c.brain or (await self.ensure_voice_brain(c) if c.voice else None)
                if brain is None:
                    res: dict[str, Any] = {"auth": "error", "auth_detail": "No brain for this conversation."}
                else:
                    res = dict(await asyncio.wait_for(brain.connect_check(), CONNECT_CHECK_TIMEOUT_S))
            except TimeoutError:
                res = {"auth": "error", "auth_detail": "The check took too long."}
            except Exception:
                log.warning("connect_check failed", exc_info=True)
                res = {"auth": "error", "auth_detail": "The check failed."}
            c.emit(s.StatusEvent, data={"connect_check": {**res, "check_id": check_id}})

        c.spawn(run(), "connect_check")
        return s.ConnectCheckData(accepted=True, check_id=check_id)

    # -- brain passthrough -------------------------------------------------------------------
    async def reset(self, conv: str) -> None:
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.reset()

    async def warm(self, conv: str) -> None:
        c = self.conversation(conv)
        try:
            brain = await self.ensure_voice_brain(c) if c.voice else c.brain
            if brain is not None:
                await brain.warm()
        except Exception:
            log.warning("warm failed", exc_info=True)  # never fails (Brain.warm)

    async def sleep(self, conv: str) -> None:
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.sleep()

    async def set_context(self, conv: str, locked: bool) -> None:
        c = self.conversation(conv)
        c.locked = locked
        if c.brain is not None:
            await c.brain.set_context(locked)

    async def set_language(self, conv: str, lang: str) -> None:
        self.voice_settings.language = lang
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.set_language(lang)

    async def set_name(self, conv: str, name: str) -> None:
        self.voice_settings.name = name
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.set_name(name)

    async def set_voice_settings(self, op: s.VoiceSettings) -> None:
        """macd's [brain] values (W0). Model, Chrome and setting sources apply from the next
        brain start; a running turn is never cut."""
        self.voice_settings = VoiceSettings(op.model, op.chrome, list(op.setting_sources), op.name, op.language)
        c = self.conversation(s.VOICE_CONV)
        if c.brain is not None:
            await c.brain.set_name(op.name)
            await c.brain.set_language(op.language)

    # -- records ---------------------------------------------------------------------------
    async def new_session(self, op: s.NewSession) -> None:
        r = self.conversation(op.conv).record
        r.fresh_next, r.fresh_folder, r.last_answer_at = True, op.folder or None, 0.0
        await self.persist()

    async def ack(self, conv: str, uid: str) -> None:
        r = self.conversation(conv).record
        before = len(r.unacked)
        r.unacked = [u for u in r.unacked if u.get("turn_uid") != uid]
        if len(r.unacked) != before:
            await self.persist()

    async def pause(self, conv: str, value: bool, reason: str | None, kind: s.ClientKind) -> s.PauseData:
        c = self.conversation(conv)
        if not value and kind is not s.ClientKind.APP:
            raise ApiError(s.ErrorCode.FORBIDDEN, "Only the Mac menu can unpause a chat.")
        c.record.paused = (reason or "user") if value else ""
        await self.persist()
        return s.PauseData(paused=c.record.paused)

    def status(self, convs: list[str] | None) -> s.StatusAll:
        names = list(self.convs) if convs is None else convs
        return s.StatusAll(conversations=[self.conversation(n).status() for n in names])

    # -- workspaces ------------------------------------------------------------------------
    def _ws_path(self, raw: str) -> Path:
        p = resolve_user_path(raw, self.paths.home)
        if p is None or is_skipped(p, self.paths.home):
            raise ApiError(s.ErrorCode.BAD_ARGS, "That folder is not one Mac can use.")
        return p

    async def ws_list(self) -> dict[str, Any]:
        return {"workspaces": [s.to_json(w) for w in await self.workspaces.listing()]}

    async def ws_approve(self, raw: str) -> s.WsApproveData:
        p = self._ws_path(raw)
        if not await asyncio.to_thread(p.is_dir):
            raise ApiError(s.ErrorCode.BAD_ARGS, "That folder does not exist.")
        return s.WsApproveData(settings_hash=await self.workspaces.approve(p))

    async def ws_remove(self, raw: str) -> None:
        await self.workspaces.remove(self._ws_path(raw))


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        log.warning("chat config at %s is unreadable; no chat conversations allowed", path)
        return {}


__all__ = ["AgentPaths", "ApiError", "Conversation", "GapError", "Service"]
