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
import dataclasses
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentd import context as ctxmod
from agentd import offload
from agentd import profiles as pf
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
    under,
)
from macd.brain.claude_brain import WAKE_TURN_BASE
from macd.brain.gate import ConfirmationGate
from macd.contracts import Brain, BrainEvent, BrainEventKind
from macd.settings import rename_refusal

log = logging.getLogger(__name__)

SERVER_NAME = "agentd/0.9.0"
HANDOFF_TIMEOUT_S = 60.0
CONNECT_CHECK_TIMEOUT_S = 25.0
VOICE_GRACE_S = 3.0  # a voice connection gone this long interrupts its running turn (V2)
CATCH_UP_NOTE_S = 300.0  # a replayed ask older than this tells the model when it was sent (AG5)
QUESTION_TTL_S = {s.QuestionKind.FOLDER_APPROVAL: 600.0, s.QuestionKind.ROUTE: 3600.0,
                  s.QuestionKind.CATCH_UP: 3600.0, s.QuestionKind.RESTART: 3600.0,
                  s.QuestionKind.PROFILE: 600.0}
RESTART_TEXT = "I restarted on your Mac in the middle of this, so I stopped. Want me to pick it up?"
RESUME_WINDOW_S = 1800.0  # a turn cut by a restart this recently is picked up again by itself; older ones ask
RESUMED_NOTE = ("[Note from Mac: Mac was restarted while working on this request and is picking it up where it "
                "left off, in the same conversation. Continue; do not start over or repeat what is already done. "
                "The request was:]\n\n")
AGENTD_EPOCH = "agentd"  # epoch of turns agentd starts itself (a "yes" to a restart question)
# a late STATUS/DONE of a finished turn is dropped; these mean the brain started a new query on it
_NEW_QUERY_KINDS = frozenset({BrainEventKind.TEXT, BrainEventKind.SAY, BrainEventKind.TOOL_START,
                              BrainEventKind.CONFIRM, BrainEventKind.ATTACH})


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
    def pairing(self) -> Path:
        """Fixed, read-only (P1): the WhatsApp pairing's chat list."""
        return self.mac_home / "chat" / "pairing.json"

    @property
    def chat_media(self) -> Path:
        """Fixed: where the bridge saves media; a quoted file must resolve inside it (W10.5)."""
        return self.mac_home / "chat" / "whatsapp" / "media"

    def chat_dir(self, conv: str) -> Path:
        """The empty per-chat dir a chat without a folder works in (W9.5)."""
        return self.mac_home / "agent" / "chats" / hashlib.sha1(conv.encode()).hexdigest()[:12]

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
    note: str = ""  # context for the brain only, before the text (catch-up time, restart)
    switch: asyncio.Future[tuple[Path | None, str] | str] | None = None  # AG1: a move the model asked for
    held: BrainEvent | None = None  # the brain's DONE/ERROR, held while a switch is decided
    moved: bool = False  # a request moves at most once (no ping-pong between folders)
    torn: bool = False  # its profile was disabled or deleted: nothing more of it is shown (P3)
    switch_tools: int = 0  # tools the moving run had already used: the user decides first
    context: s.AskContext | None = None  # chat (1.2): validated; framed as a per-session delta (V1)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    ended: asyncio.Event = field(default_factory=asyncio.Event)  # finished, or the brain turn held


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
        self._alive_sent = False  # the last `status{alive}` pushed on this stream
        self.turn: Turn | None = None  # the newest turn
        self.turns: dict[int, Turn] = {}  # internal id -> turn, until finished
        self.uids: set[str] = set()
        self.last_finished: Turn | None = None
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
        if turn is not None and turn.torn:
            return  # P3/I2: a torn-down turn shows nothing more, not even its `done`
        if turn is not None:
            fields.update(turn=turn.client_turn, turn_uid=turn.uid)
            epoch = turn.epoch if self.voice else None
        self.stream.emit(cls, epoch=epoch, **fields)

    def running(self) -> Turn | None:
        t = self.turn
        return t if t is not None and not t.finished else None

    # -- brain ---------------------------------------------------------------------------
    def brain_alive(self) -> bool:
        return bool(self.brain is not None and getattr(self.brain, "alive", False))

    def check_alive(self) -> None:
        """Push `status{alive}` when the brain process connected or went away (warm, idle
        shutdown, sleep, crash, close), so clients need not poll `status` for it."""
        alive = self.brain_alive()
        if alive != self._alive_sent:
            self._alive_sent = alive
            self.emit(s.StatusEvent, data={"alive": alive})

    def _on_alive(self, _alive: bool) -> None:
        self.check_alive()  # re-read the brain: the callback may race a close

    def _confirm_files(self) -> bool:
        """Live: a file send asks first if the profile says so; with no readable row, it asks."""
        p = self.svc.profile_of(self.conv)
        return True if p is None else bool(p.confirm_files)

    def _capabilities(self) -> frozenset[str]:
        """Live (W9.4): the gate reads this at every tool call, so narrowing applies at once."""
        return self.svc.live_caps(self.conv)

    async def _on_set_persona(self, text: str) -> str | None:
        """The model's set_persona, after the owner's bound yes (B's tool asks): write the row."""
        try:
            await self.svc.set_persona(self.conv, text)
        except pf.ProfileError as e:
            return e.message
        return None

    def _on_rename(self, name: str) -> str | None:
        """Voice: the model's rename_assistant tool. The voice client applies and saves it, after
        the same pure check here, so a refusal reaches the model instead of "Renamed" (N11)."""
        if refused := rename_refusal(name):
            return refused
        self.emit(s.StatusEvent, data={"rename": {"name": name}})
        return None

    def _on_reset_voice(self) -> None:
        self.emit(s.StatusEvent, data={"reset_voice": {}})

    def _on_go_to_sleep(self) -> None:
        self.emit(s.StatusEvent, data={"go_to_sleep": {}})

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
            pending = r.framed.pop("", None)
            if pending:  # a brand-new session's framed refs, filed under its id now it has one
                ctxmod.remember(r.framed, info.session_id, pending)
        self.svc.save_soon()

    async def ensure_brain(self, cwd: Path, sources: list[str], *, rebuild: bool = False) -> Brain:
        vs, svc = self.svc.voice_settings, self.svc
        prof = None if self.voice else svc.profile_of(self.conv)
        caps = pf.granted(prof)
        spec = BrainSpec(
            conv=self.conv, profile=self.profile, cwd=cwd, setting_sources=list(sources), gate=self.gate,
            session_store=RecordSessionStore(self._session_get, self._session_put),
            memory_file=self.svc.memory_file, desktop=self.desktop, recent_user_text=lambda: list(self.recent),
            model=vs.model if self.voice else "", chrome=vs.chrome if self.voice else "chrome" in caps,
            name=vs.name, language=vs.language, on_alive=self._on_alive,
            never_chat_names=() if self.voice else svc.never.names,
            capabilities=None if self.voice else self._capabilities,
            confirm_send_file=self._confirm_files,
            chat_persona=prof.persona if prof else "", display_name=prof.display_name if prof else "",
            chat_shared=bool(prof and prof.shared),
            on_set_persona=None if self.voice else self._on_set_persona,
            on_rename=self._on_rename if self.voice else None,
            on_reset_voice=self._on_reset_voice if self.voice else None,
            on_go_to_sleep=self._on_go_to_sleep if self.voice else None)
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
        self.check_alive()
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
        self.check_alive()

    async def _pump_events(self, brain: Brain) -> None:
        try:
            async for ev in brain.events():
                try:
                    await self._on_event(ev)
                except Exception:
                    log.exception("conv %s: event handling failed", self.conv)
                self.check_alive()
        finally:
            crashed = self.brain is brain  # its event stream ended without close_brain
            for t in list(self.turns.values()):  # the brain ended under a running turn
                if crashed and t.in_brain and not t.finished:
                    self.emit(s.ErrorEvent, t, kind=s.EventErrorKind.INTERNAL,
                              message="Mac's brain stopped unexpectedly.")
                    await self.finish(t, error=True)
            if crashed and self.brain is brain:  # the next turn starts a new brain
                self.brain, self.brain_key, self._pump = None, None, None
                self.check_alive()
                await close_quietly(brain)

    async def _on_event(self, ev: BrainEvent) -> None:
        t = self.turns.get(ev.turn_id)
        if t is None or t.finished:
            t = self._resumed(ev) if ev.kind in _NEW_QUERY_KINDS else None
            if t is None:
                return
        k = ev.kind
        if t.switch is not None and t.internal == ev.turn_id and not t.stopping:
            if k in (BrainEventKind.DONE, BrainEventKind.ERROR):  # run_turn moves the request
                t.held, t.in_brain = ev, False
                t.ended.set()
            return  # nothing more from the turn that is moving is shown
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
            self.open_confirm(t, ev.text, str(ev.data.get("target", "")), str(ev.data.get("question") or ""),
                              str(ev.data.get("detail") or ""))  # detail: set_persona's full new text
        elif k is BrainEventKind.ATTACH:
            await self._attach(t, ev)
        elif k is BrainEventKind.STATUS:
            data = dict(ev.data)
            req = data.pop("switch_workspace", None)
            if isinstance(req, dict) and not self.voice and t.switch is None and not t.stopping:
                self._switch_requested(t, req)
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
            self.emit(s.ErrorEvent, t, kind=s.EventErrorKind(kind), message=ev.text)  # "" passes through: voice speaks its own per-kind line
            await self.finish(t, error=True)

    def _resumed(self, ev: BrainEvent) -> Turn | None:
        """Voice: a steer that crossed the turn's DONE makes the brain run a new query on the
        same turn (local ClaudeBrain semantics). Its events continue that turn, as in local mode.
        Chat: Claude reporting back on its own after a background task finished is a new turn."""
        if not self.voice:
            if ev.turn_id < WAKE_TURN_BASE or self.running() is not None:
                return None
            return self.svc.wake_turn(self, ev.turn_id)
        last = self.last_finished
        if not self.voice or last is None or last.internal != ev.turn_id or last.stopping or self.running():
            return None
        t = self.svc.continue_turn(self, last, "", internal=ev.turn_id)
        t.in_brain = True
        return t

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
        if not self.voice and not t.torn:  # voice and torn turns never enter unacked
            r.add_unacked({"turn_uid": t.uid, "answer": "".join(t.answer).strip(), "attaches": t.attaches})
        if not interrupted and not error:
            r.last_answer_at = self.svc.clock()
        if self.turn is t:
            self.turn = None
        self.last_finished = t
        t.done.set()
        t.ended.set()
        if self.voice:
            self.svc.save_soon()  # off the pump: the next voice turn's first TEXT never waits on fsync
        else:
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
            await self._ask_brain(t)
            for text in t.steers:
                await self.brain.steer(t.internal, text)
            while True:
                await t.ended.wait()
                if t.finished or not await self._move(t):
                    break
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
        if meta is not None and meta.replayed and meta.sent_at and svc.clock() - meta.sent_at > CATCH_UP_NOTE_S:
            t.note = _catch_up_note(meta.sent_at, svc.clock())  # AG5: the model decides to ask or act
        current = Path(r.folder) if r.folder else None
        prof = svc.profile_of(self.conv)
        default = svc.default_workspace(prof)
        if prof is not None and "files_read" not in pf.granted(prof):
            # W9.5: no folders at all; the empty per-chat dir, and no router call
            route = Route(None, "new" if r.fresh_next or current is not None else "same")
        elif r.fresh_next and not r.fresh_folder and default is not None:
            route = Route(default, "new")  # a new session starts in the chat's folder
        else:
            route = await svc.router.route(t.text, folder=current, last_answer_at=r.last_answer_at,
                                           fresh=r.fresh_next, named=r.fresh_folder, default=default)
        folder, session = route.folder, route.session
        if route.question:
            opts = [{"label": o.label, "keys": o.keys} for o in route.options]
            ans = await self.ask_question(t, s.QuestionKind.ROUTE, route.question,
                                          {"options": opts, "yes": route.yes})
            if not isinstance(ans, Binding):
                raise asyncio.CancelledError
            if ans.choice is not None:
                folder, session = route.options[ans.choice].folder, route.options[ans.choice].session
            else:  # AG4: their own words go back to the model, which decides without asking again
                again = await svc.router.route(t.text, folder=current, last_answer_at=0.0, fresh=r.fresh_next,
                                               named=None, answer=(route.question, ans.steer_text),
                                               default=default)
                folder, session = (None, "new") if again.question else (again.folder, again.session)
        r.fresh_next, r.fresh_folder = False, None
        await self._enter(t, folder, session)
        return True

    async def _enter(self, t: Turn, folder: Path | None, session: str) -> None:
        """Point the conversation at `folder` and `session` ("same" | "new" | "continue:<id>"),
        with the trust check and approval question (W3.5), and get its brain ready."""
        svc, r = self.svc, self.record
        current = Path(r.folder) if r.folder else None
        fresh_session, prev_sid = False, r.session_id
        if session == "same" and folder == current:
            pass
        elif session.startswith("continue:"):
            sid = session.split(":", 1)[1]
            r.session_id, r.session_fork = sid, sid not in r.own_sessions
            r.session_updated_at = r.session_started_at = 0.0
        else:
            r.session_id, fresh_session = None, True
            r.framed.pop("", None)  # a new session starts with nothing framed (V1)
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
        cwd = folder or (svc.paths.chat_dir(self.conv) if svc.profile_of(self.conv) else svc.paths.chat_cwd)
        brain = await self.ensure_brain(cwd, sources, rebuild=rebuild)
        if fresh_session and brain is existing:
            await brain.reset()  # a reused brain may hold an older session

    # -- switch_workspace (AG1): the running model says the request belongs elsewhere ----------
    def _switch_requested(self, t: Turn, req: dict[str, Any]) -> None:
        fut: asyncio.Future[tuple[Path | None, str] | str] = asyncio.get_running_loop().create_future()
        t.switch = fut
        internal = t.internal
        try:
            t.switch_tools = max(0, int(req.get("tools_run") or 0))
        except (TypeError, ValueError):
            t.switch_tools = 1  # unknown: assume work was done, so ask

        async def decide() -> None:
            try:
                res: tuple[Path | None, str] | str = await self.svc.validate_switch(
                    self, str(req.get("folder", "")), str(req.get("mode", "")))
            except Exception:
                log.exception("conv %s: switch_workspace check failed", self.conv)
                res = "the move could not be checked"
            if not fut.done():
                fut.set_result(res)
            if t.finished or t.held is not None or self.brain is None or t.internal != internal:
                return
            if isinstance(res, str):  # refused: the turn carries on where it is
                t.switch = None
                await self.brain.steer(internal, f"Mac did not move this request: {res}. Carry on here.")
            elif not t.switch_tools:  # with work done the brain ends on its own (tools blocked); ask then
                await self.brain.interrupt(internal)  # its DONE is held; run_turn moves the request

        self.spawn(decide(), "switch")

    async def _move(self, t: Turn) -> bool:
        """The brain turn ended while a switch was pending. True when the request was re-run in
        the new folder and session (the turn goes on), False when the held end was applied."""
        fut, held = t.switch, t.held
        t.ended.clear()
        res = await fut if fut is not None else "no move"
        t.switch, t.held = None, None
        if not isinstance(res, str) and not t.stopping and t.switch_tools > 0 and not await self._move_ok(t, res[0]):
            res = "the user chose to stay"
        if isinstance(res, str) or t.stopping:
            if held is not None:
                await self._on_event(held)  # the turn ends as the brain ended it
            return False
        self.turns.pop(t.internal, None)
        t.moved = True
        await self._enter(t, *res)
        if t.stopping:
            raise asyncio.CancelledError
        assert self.brain is not None
        t.internal = self.svc.next_internal()
        self.turns[t.internal] = t
        t.answer.clear()
        t.in_brain = True
        await self._ask_brain(t)  # a new session id: the ring is framed into it once (V1)
        return True

    async def _ask_brain(self, t: Turn) -> None:
        """The user turn with its context delta (V1): only ring entries not yet framed into this
        session, recorded once the brain has accepted the turn. A refused or stopped ask never
        gets here, so its entries go next time."""
        assert self.brain is not None
        r, svc = self.record, self.svc
        key = r.session_id or ""
        msgs = ctxmod.delta(t.context, r.framed.get(key, ()))
        prof = svc.profile_of(self.conv)
        text = ctxmod.render(t.text, t.context, msgs, display_name=prof.display_name if prof else "",
                             owner_name=svc.owner_name, assistant_name=svc.voice_settings.name or "Mac")
        await self.brain.ask(t.internal, t.note + text)
        if msgs:
            ctxmod.remember(r.framed, r.session_id or key, [m.msg_ref for m in msgs])
            svc.save_soon()

    async def _move_ok(self, t: Turn, folder: Path | None) -> bool:
        """Work already ran here (tools_run > 0): moving re-runs the request from the start, so
        the user decides. An exact yes or "move" moves; "stay" stays. Any other reply also stays,
        and is then started as the next request, with this one as context (P2-1, as a restart)."""
        where = (display(folder, self.svc.paths.home) or str(folder)) if folder else "no particular folder"
        opts = [{"label": "move", "keys": ["move", "1"]}, {"label": "stay", "keys": ["stay", "here", "2"]}]
        ans = await self.ask_question(
            t, s.QuestionKind.ROUTE, f"I already started on this here. Move it to {where} and start over there?",
            {"options": opts, "yes": 0})
        if isinstance(ans, Binding) and ans.choice is None and ans.steer_text:
            await self.svc.follow_up(self, ans.steer_text, t.text, t.msg_ref,
                                     "asked whether to move it to another folder and start over")
        return isinstance(ans, Binding) and ans.choice == 0

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

    def open_confirm(self, t: Turn, summary: str, target: str, question: str = "", detail: str = "") -> None:
        self._replace()
        q = Question(_qid(), s.QuestionKind.CONFIRM, summary,
                     self.svc.clock() + self.profile.confirm_window_s + 1.0,
                     payload={"target": target, "question": question, "detail": detail}, turn_internal=t.internal)
        self.question = q
        self._set_timer(q)
        self.svc.desktop.release_conv(self.conv)  # never hold the desktop while confirming (F3)
        self.emit(s.ConfirmEvent, t, question_id=q.question_id, summary=summary, target=target,
                  question=question or None, detail=detail or None)

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
        question = None
        if q is not None:
            confirm = q.kind is s.QuestionKind.CONFIRM
            question = s.QuestionStatus(
                question_id=q.question_id, kind=q.kind, text=q.text, expires_at=q.expires_at,
                question=(q.payload.get("question") or None) if confirm else None,
                target=(q.payload.get("target") or None) if confirm else None,
                detail=(q.payload.get("detail") or None) if confirm else None)
        unacked = [] if self.voice else [
            s.UnackedTurn(turn_uid=u["turn_uid"], answer=u.get("answer", ""),
                          attaches=[s.AttachRef(**a) for a in u.get("attaches", [])]) for u in r.unacked]
        return s.ConversationStatus(
            conv=self.conv, alive=self.brain_alive(),
            folder=display(r.folder, svc.paths.home), session_title=r.session_title, paused=r.paused,
            running=running, question=question, desktop=svc.desktop.status(), unacked=unacked,
            loop_lag_ms_p99=svc.lag.p99())


def _qid() -> str:
    return os.urandom(8).hex()


_UNAVAILABLE = "Chat settings can't be read right now, so WhatsApp is off. Voice still works."


@contextlib.contextmanager
def _profile_errors() -> Any:
    try:
        yield
    except pf.ProfileError as e:
        raise ApiError(e.code, e.message) from None


def _catch_up_note(sent_at: float, now: float) -> str:
    """AG5: the model decides whether a late request still makes sense (no fixed rule asks)."""
    when = time.strftime("%I:%M %p", time.localtime(sent_at)).lstrip("0")
    mins = int(max(0.0, now - sent_at) // 60)
    ago = f"{mins} min" if mins < 120 else f"{mins // 60} h"
    return (f"[Note from Mac: the user sent this at {when}, {ago} ago, while Mac was asleep. If acting "
            "now could be stale or unwanted (time-sensitive things, sending messages, purchases), reply "
            "with one short question instead of acting; otherwise just do it.]\n\n")


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
                 lag_interval_s: float = 0.1, owner_name: str = "") -> None:
        self.paths = paths
        self.owner_name = owner_name  # macd's user_name: a member named like the owner is marked (V4)
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
        self.profiles = pf.ProfileStore(paths.state_dir / pf.DB_NAME, clock=clock)
        self.never = pf.NeverSet()
        self._never_torn: set[str] = set()
        self.notice_sink: Callable[[dict[str, Any]], None] = lambda _n: None  # the server broadcasts
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
        await self.profiles.open()
        await self.refresh_never()
        await self._seed()
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
        self.spawn(offload.warm())

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
        if not c.voice and not r.paused and self.clock() - run.started_at < RESUME_WINDOW_S:
            # Owner, 2026-11-12: a restart must not lose the request. Same session, same request, a note
            # that says so; the chat sees an ordinary turn start instead of a "pick it up?" question.
            t = self._new_turn(c, AGENTD_EPOCH, self._next_internal + 1, run.text, run.msg_ref)
            t.note = RESUMED_NOTE
            log.info("conv %s: resuming the turn cut by the restart", c.conv)
            self.spawn(self._start_turn(c, t, None))
            return
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
        await self.profiles.close()

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
    # -- profiles (W9) ----------------------------------------------------------------------
    def profile_of(self, conv: str) -> s.Profile | None:
        channel, _, chat_id = conv.partition(":")
        return self.profiles.get(channel, chat_id) if chat_id else None

    def chat_ok(self, p: s.Profile | None) -> bool:
        return p is not None and p.enabled and not self.never.blocks(p.chat_id)

    def chat_check(self, conv: str) -> None:
        """Live grant (P2): every chat op is checked against the profile cache."""
        if not self.profiles.available:
            raise ApiError(s.ErrorCode.PROFILES_UNAVAILABLE, _UNAVAILABLE)
        if conv == s.VOICE_CONV or not self.chat_ok(self.profile_of(conv)):
            raise ApiError(s.ErrorCode.FORBIDDEN_CONVERSATION, "Mac is not set up to answer in that chat.")

    def live_chat_convs(self) -> list[str]:
        if not self.profiles.available:
            return []
        return [f"{p.channel}:{p.chat_id}" for p in self.profiles.rows() if self.chat_ok(p)]

    def live_caps(self, conv: str) -> frozenset[str]:
        p = self.profile_of(conv)
        return pf.granted(p) if self.profiles.available and self.chat_ok(p) else frozenset()

    def default_workspace(self, p: s.Profile | None) -> Path | None:
        """The chat's folder, if it is still an approved workspace (W9.5)."""
        if p is None or not p.default_workspace:
            return None
        path = resolve_user_path(p.default_workspace, self.paths.home)
        return path if path is not None and str(path) in self.workspaces.approved else None

    def _approved_ws(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        path = resolve_user_path(raw, self.paths.home)
        if path is None or str(path) not in self.workspaces.approved:
            raise pf.ProfileError(s.ErrorCode.WORKSPACE_NOT_APPROVED,
                                  "That folder is not an approved workspace. Approve it first.")
        return display(path, self.paths.home)

    async def refresh_never(self, add: frozenset[str] = frozenset()) -> None:
        """Re-read the never-set (fixed paths, read-only) and make every JID in it sticky in the
        profile store (I1: emptying pairing.json or config.toml unblocks nothing). A row whose
        chat became never-listed, or every row when the set is unreadable, is shown disabled:
        teardown, a new rev and a notice (I3)."""
        loaded = await asyncio.to_thread(pf.load_never_set, self.paths.chat_config, self.paths.pairing)
        st = self.profiles
        bumped = False
        if st.available:
            try:
                bumped = await st.add_never((loaded.jids if loaded.ok else frozenset()) | add)
            except Exception:  # the in-memory set below still blocks them
                log.warning("profiles: could not store never-listed chats", exc_info=True)
        self.never = dataclasses.replace(loaded, jids=loaded.jids | st.never_jids | add)
        if not st.available:
            return
        for p in st.rows():
            conv = f"{p.channel}:{p.chat_id}"
            if not self.never.blocks(p.chat_id):
                self._never_torn.discard(conv)
            elif p.enabled and conv not in self._never_torn:
                self._never_torn.add(conv)
                log.warning("profiles: %s is never-listed now; Mac stops answering there", conv)
                if not bumped:
                    with contextlib.suppress(Exception):
                        await st.bump()
                    bumped = True
                await self.teardown(conv, lambda p=p: self.notify_profile(p.channel, p.chat_id, p))

    async def never_list(self, conv: str) -> None:
        """`pause{reason: never_listed}` (chatd saw the subject turn Timetable): sticky at once."""
        await self.refresh_never(frozenset({conv.partition(":")[2]}))

    async def teardown(self, conv: str, notice: Callable[[], None] | None = None) -> None:
        """P3 in the I2 order: (a) agentd state first (every unfinished turn of the chat is torn,
        so nothing more of it is shown, not even `done`, and it stays out of unacked; the open
        question closes skipped; confirmation tokens are revoked), (b) the notice, (c) only
        then the brain interrupt, which may be slow."""
        c = self.convs.get(conv)
        if c is None:
            if notice is not None:
                notice()
            return
        if c.question is not None:
            c.close_question(c.question, s.CloseOutcome.SKIPPED)
        c.gate.revoke_all()
        torn = [t for t in c.turns.values() if not t.finished]
        if c.turn is not None and not c.turn.finished and c.turn not in torn:
            torn.append(c.turn)
        for t in torn:
            t.torn = True
        if notice is not None:
            notice()
        for t in torn:
            await c.interrupt(t)

    def shown(self, p: s.Profile | None) -> s.Profile | None:
        """A never-listed row reads as disabled everywhere (I3), whatever is stored."""
        if p is not None and p.enabled and self.never.blocks(p.chat_id):
            return dataclasses.replace(p, enabled=False)
        return p

    def notify_profile(self, channel: str, chat_id: str, p: s.Profile | None) -> None:
        n = s.ProfileChanged(rev=self.profiles.rev, channel=channel, chat_id=chat_id, profile=self.shown(p),
                             deleted=None if p is not None else True)
        self.notice_sink(s.encode_notice(n))

    def _need_store(self) -> None:
        if not self.profiles.available:
            raise ApiError(s.ErrorCode.PROFILES_UNAVAILABLE, _UNAVAILABLE)

    async def profile_list(self, kind: s.ClientKind) -> s.ProfileListData:
        self._need_store()
        rows = [self.shown(p) for p in self.profiles.rows()]
        if kind is not s.ClientKind.APP:
            rows = [p for p in rows if self.chat_ok(p)]
        return s.ProfileListData(rev=self.profiles.rev, profiles=rows)

    async def profile_create(self, op: s.ProfileCreate) -> s.ProfileData:
        self._need_store()
        await self.refresh_never()
        with _profile_errors():
            if not op.channel or not op.chat_id or ":" in op.channel:
                raise pf.ProfileError(s.ErrorCode.BAD_ARGS, "channel and chat_id are needed.")
            extra = {k: getattr(op, k) for k in pf.EDITABLE if k != "display_name" and getattr(op, k) is not None}
            fields = pf.validate({"display_name": op.display_name, **extra})
            if self.never.blocks(op.chat_id) or pf.is_never_subject(fields["display_name"]):
                raise pf.ProfileError(s.ErrorCode.NEVER_LISTED, "I never answer in that chat.")
            if "default_workspace" in fields:
                fields["default_workspace"] = self._approved_ws(fields["default_workspace"])
            name = fields.pop("display_name")
            p = await self.profiles.create(pf.new_row(op.channel, op.chat_id, name, op.owner_only, fields, self.clock()))
        self.notify_profile(p.channel, p.chat_id, p)
        return s.ProfileData(profile=p)

    async def profile_set(self, op: s.ProfileSet) -> s.ProfileData:
        self._need_store()
        await self.refresh_never()
        with _profile_errors():
            cur = self.profiles.get(op.channel, op.chat_id)
            if cur is None:
                raise pf.ProfileError(s.ErrorCode.PROFILE_NOT_FOUND, "There are no settings for that chat.")
            patch = pf.validate(op.patch)  # capabilities: a delta, merged inside the write (CQ race)
            if (patch.get("enabled") and self.never.blocks(cur.chat_id)) or \
                    pf.is_never_subject(patch.get("display_name", "")):
                raise pf.ProfileError(s.ErrorCode.NEVER_LISTED, "I never answer in that chat.")
            if "default_workspace" in patch:
                patch["default_workspace"] = self._approved_ws(patch["default_workspace"])
            p = await self.profiles.update(op.channel, op.chat_id, patch, expect_rev=op.rev)
        await self._after_write(p.channel, p.chat_id, p)
        return s.ProfileData(profile=p)

    async def profile_delete(self, op: s.ProfileDelete) -> None:
        self._need_store()
        with _profile_errors():
            await self.profiles.delete(op.channel, op.chat_id)
        await self._after_write(op.channel, op.chat_id, None)

    async def _after_write(self, channel: str, chat_id: str, p: s.Profile | None) -> None:
        if self.chat_ok(p):
            self.notify_profile(channel, chat_id, p)
        else:
            await self.teardown(f"{channel}:{chat_id}", lambda: self.notify_profile(channel, chat_id, p))

    async def set_persona(self, conv: str, text: str) -> None:
        """The model's `set_persona`, after a bound yes: this chat only (bound at brain build)."""
        cur = self.profile_of(conv)
        if cur is None or not self.chat_ok(cur):
            raise pf.ProfileError(s.ErrorCode.PROFILE_NOT_FOUND, "This chat has no settings.")
        patch = pf.validate({"persona": text})
        p = await self.profiles.update(cur.channel, cur.chat_id, patch)
        self.notify_profile(p.channel, p.chat_id, p)

    async def profile_command(self, op: s.ProfileCommand) -> s.ProfileCommandData:
        """W9.3 owner commands (parsed by chatd). `add` and `allow` open a profile question;
        only a bound yes applies them, as a patch on the row as it is then."""
        self._need_store()
        c = self.conversation(op.conv)
        channel = op.conv.partition(":")[0]
        if op.command == "disallow":
            cap = self._capability(op.capability)
            cur = self.profile_of(op.conv)
            if cur is not None and pf.caps_dict(cur)[cap]:
                with _profile_errors():
                    p = await self.profiles.update(cur.channel, cur.chat_id, {"capabilities": {cap: False}})
                self.notify_profile(p.channel, p.chat_id, p)
            return s.ProfileCommandData(applied=True)
        if op.command == "allow":
            cap = self._capability(op.capability)
            q = c.open_question(s.QuestionKind.PROFILE, f"Turn on {s.CAPABILITY_WORDS[cap]} in this chat? "
                                "Reply yes to this message.", {"command": "allow", "capability": cap})
            return s.ProfileCommandData(question_id=q.question_id)
        subject, cid = (op.subject or "").strip(), (op.chat_id or "").strip()
        if not subject or not cid or op.owner_only is None:
            raise ApiError(s.ErrorCode.BAD_ARGS, "add needs subject, chat_id and owner_only.")
        await self.refresh_never()
        if self.never.blocks(cid) or pf.is_never_subject(subject):
            raise ApiError(s.ErrorCode.NEVER_LISTED, "I never answer in that chat.")
        if self.profiles.get(channel, cid) is not None:
            raise ApiError(s.ErrorCode.PROFILE_EXISTS, f"I already answer in “{subject[:80]}”.")
        q = c.open_question(s.QuestionKind.PROFILE, f"Answer in “{subject[:80]}” too? Reply yes to this message.",
                            {"command": "add", "channel": channel, "chat_id": cid, "subject": subject[:80],
                             "owner_only": op.owner_only})
        return s.ProfileCommandData(question_id=q.question_id)

    @staticmethod
    def _capability(raw: str | None) -> str:
        if raw not in s.CAPABILITIES:
            raise ApiError(s.ErrorCode.BAD_ARGS, f"capability must be one of {', '.join(s.CAPABILITIES)}.")
        return raw

    async def _apply_profile_question(self, c: Conversation, payload: dict[str, Any]) -> None:
        """A bound yes: re-check the never-set and existence, then patch the row as it is now (N6)."""
        await self.refresh_never()
        cmd = payload.get("command")
        result: dict[str, Any] = {"command": cmd, "applied": False}
        try:
            if cmd == "add":
                channel, cid, subject = str(payload["channel"]), str(payload["chat_id"]), str(payload["subject"])
                if self.never.blocks(cid) or pf.is_never_subject(subject):
                    result["message"] = f"I never answer in “{subject}”."
                elif self.profiles.get(channel, cid) is not None:
                    result["message"] = f"I already answer in “{subject}”."
                else:
                    row = pf.new_row(channel, cid, subject, bool(payload.get("owner_only")), {}, self.clock())
                    p = await self.profiles.create(row)
                    self.notify_profile(p.channel, p.chat_id, p)
                    result.update(applied=True, message=f"Done. I answer in “{subject}” now, with everything "
                                                        "turned on. Turn things off in Mac ▸ WhatsApp.")
            elif cmd == "allow":
                cap, cur = str(payload["capability"]), self.profile_of(c.conv)
                if cur is not None and self.chat_ok(cur):
                    p = await self.profiles.update(cur.channel, cur.chat_id, {"capabilities": {cap: True}})
                    self.notify_profile(p.channel, p.chat_id, p)
                    w = s.CAPABILITY_WORDS[cap]
                    result.update(applied=True, message=f"{w[0].upper()}{w[1:]} is on in this chat.")
        except pf.ProfileError as e:
            result["message"] = e.message
        c.emit(s.StatusEvent, data={"profile_command": result})

    async def _seed(self) -> None:
        """W9.10, once: Interview prep (and the old `chats` key) from pairing.json and chatd's
        config. Never Timetable. Never again, even if the table is later emptied."""
        st = self.profiles
        if not st.available or st.seeded_from is not None:
            return
        pairing = await asyncio.to_thread(pf.read_pairing, self.paths.pairing)
        if st.cache:
            await st.mark_seeded(json.dumps({"sources": [], "at": self.clock(), "note": "rows existed"}))
            return
        if not self.never.ok:
            return  # never-set unreadable (N4): seed nothing, try again at the next start
        resumes = [a for a in self.workspaces.approved if Path(a).name == "Resume"]
        approved = display(Path(resumes[0]), self.paths.home) if len(resumes) == 1 else None
        for row in pf.seed_rows(self.never, pairing, approved, self.clock()):
            with contextlib.suppress(pf.ProfileError):
                await st.create(row)
        sources = [n for n, ok in (("pairing.json", pairing is not None), ("config.toml", self.never.config is not None))
                   if ok]
        await st.mark_seeded(json.dumps({"sources": sources, "at": self.clock()}))
        log.info("profiles: seeded %d chat(s) from %s", len(st.cache), ", ".join(sources))

    async def validate_switch(self, c: Conversation, folder: str, mode: str) -> tuple[Path | None, str] | str:
        """AG1, deterministic: the folder exists under home and is not skipped, one match only,
        and `continue` picks a session that folder really has. Approval and the hash check run
        as at any turn start. Returns (folder, session) or why not."""
        if c.voice:
            return "voice requests do not move between folders"
        if c.turn is not None and c.turn.moved:
            return "this request already moved once"
        if mode not in ("continue", "new"):
            return "mode must be continue or new"
        current = Path(c.record.folder) if c.record.folder else None
        cands = await self.router.candidates()
        if folder.strip().lower() == "general":
            path = None
        else:
            hits = await self.router.find_folder(folder.strip(), cands)
            hits = [h for h in hits if under(h, self.paths.home) and not is_skipped(h, self.paths.home)]
            if not hits:
                return f"there is no folder called {folder} in the home folder"
            if len(hits) > 1:
                listed = ", ".join(display(h, self.paths.home) or str(h) for h in hits[:4])
                return f"several folders match {folder} ({listed}); give the path"
            path = hits[0]
        if mode == "new":
            session = "new"
        elif path == current:
            session = "same"
        else:
            sessions = next((cd.sessions for cd in cands if cd.path == path), [])
            session = f"continue:{sessions[0][0]}" if sessions and path is not None else "new"
        if path == current and session == "same":
            return "the request is already in that folder"
        return path, session

    def wake_turn(self, c: Conversation, internal: int) -> Turn:
        """The brain's own query (a background task finished): a turn with a new uid, so chatd
        adopts it and posts the answer as a new message."""
        uid = s.turn_uid(self.boot, c.conv, AGENTD_EPOCH, internal)
        t = Turn(internal, uid, AGENTD_EPOCH, internal, "", time.monotonic())
        t.in_brain = True
        c.uids.add(uid)
        c.turn = t
        c.turns[internal] = t
        c.record.running = RunningTurn(t.uid, t.text, self.clock())
        self.save_soon()
        return t

    def next_internal(self) -> int:
        self._next_internal += 1
        return self._next_internal

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
        t = Turn(self.next_internal(), uid, epoch, client_turn, text, time.monotonic(), msg_ref)
        c.uids.add(uid)
        return t

    async def follow_up(self, c: Conversation, text: str, before: str, msg_ref: str, what: str) -> None:
        """AG4: the user's free-text answer is the next request; the one asked about is context."""
        t = self._new_turn(c, AGENTD_EPOCH, self._next_internal + 1, text, msg_ref)
        t.note = (f'[Note from Mac: Mac was working on "{_short(before, 200)}" and {what}. The user answered '
                  "with the message below.]\n\n")
        await self._start_turn(c, t, None)

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
        context = None
        if op.context is not None:
            if c.voice:
                raise ApiError(s.ErrorCode.BAD_ARGS, "context is for chat conversations only.")
            try:
                context = ctxmod.narrow(ctxmod.validate(op.context, self.paths.chat_media),
                                        "read_chat" in self.live_caps(op.conv))
            except ctxmod.ContextError as e:
                raise ApiError(s.ErrorCode.BAD_ARGS, str(e)) from None
        meta = op.meta
        t = self._new_turn(c, epoch, op.turn, op.text, meta.msg_ref if meta else "")
        t.context = context
        await self._start_turn(c, t, meta)
        return s.AskData(turn_uid=t.uid)

    async def _start_turn(self, c: Conversation, t: Turn, meta: s.AskMeta | None) -> None:
        before = c.turn if c.turn is not None and not c.turn.finished else None
        c.turn = t
        c.turns[t.internal] = t
        c.recent.append(t.text)
        c.record.running = RunningTurn(t.uid, t.text, self.clock(), t.msg_ref)
        if c.voice:
            self.save_soon()  # voice: never on the path to the first TEXT; a lost notice fails quiet
        else:
            await self.persist()  # chat: durable before `accepted` (chatd marks the message handled)
        self.events.emit("agentd.turn", conv=c.conv, turn_uid=t.uid)
        t.task = c.spawn(c.run_turn(t, meta, before), f"turn{t.internal}")

    def continue_turn(self, c: Conversation, last: Turn, text: str, *, internal: int = 0) -> Turn:
        """A new query on a finished voice turn: same client turn and uid, new brain turn
        (unless the brain already runs it under `internal`)."""
        if not internal:
            internal = self.next_internal()
        t = Turn(internal, last.uid, last.epoch, last.client_turn, text or last.text, time.monotonic())
        c.turn = t
        c.turns[internal] = t
        c.record.running = RunningTurn(t.uid, t.text, self.clock())
        self.save_soon()
        return t

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
        if t is None and c.voice:
            last = c.last_finished
            if (last is not None and not last.stopping and last.epoch == epoch and last.client_turn == op.turn
                    and c.running() is None):
                # Local ClaudeBrain semantics: a steer for a just-finished voice turn starts a new
                # query on the same turn. Its events carry the same turn and turn_uid.
                c.recent.append(op.text)
                cont = self.continue_turn(c, last, op.text)
                cont.task = c.spawn(c.run_turn(cont, None, None), f"turn{cont.internal}")
                return s.AcceptedData(accepted=True)
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
        elif q.kind is s.QuestionKind.PROFILE:
            c.close_question(q, outcome, b)
            if b.outcome is s.ReplyOutcome.APPROVED:
                await self._apply_profile_question(c, q.payload)
        elif q.kind is s.QuestionKind.RESTART:
            c.close_question(q, outcome, b)
            again = b.outcome is s.ReplyOutcome.APPROVED or (b.outcome is s.ReplyOutcome.ANSWERED and b.steer_text)
            if again and c.running() is None:
                before, ref = str(q.payload.get("text", "")), str(q.payload.get("msg_ref", ""))
                if b.outcome is s.ReplyOutcome.APPROVED:
                    await self._start_turn(c, self._new_turn(c, AGENTD_EPOCH, self._next_internal + 1, before, ref),
                                           None)
                else:
                    await self.follow_up(c, b.steer_text, before, ref, "restarted, then asked whether to pick it up")
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
    async def reset(self, conv: str, hard: bool = False) -> None:
        c = self.conversation(conv)
        if hard:  # "stop everything": the running turn, any open question, then the brain and its agents
            await c.interrupt(c.running())
            if c.question is not None:
                c.close_question(c.question, s.CloseOutcome.SKIPPED)
            c.record.fresh_next, c.record.last_answer_at = True, 0.0
            await self.persist()
        if c.brain is not None:
            await c.brain.reset(hard=hard)
        c.check_alive()

    async def warm(self, conv: str) -> None:
        c = self.conversation(conv)
        try:
            brain = await self.ensure_voice_brain(c) if c.voice else c.brain
            if brain is not None:
                await brain.warm()
        except Exception:
            log.warning("warm failed", exc_info=True)  # never fails (Brain.warm)
        c.check_alive()

    async def sleep(self, conv: str) -> None:
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.sleep()
        c.check_alive()

    async def set_context(self, conv: str, locked: bool) -> None:
        c = self.conversation(conv)
        c.locked = locked
        if c.brain is not None:
            await c.brain.set_context(locked)
        c.check_alive()

    async def set_language(self, conv: str, lang: str) -> None:
        self.voice_settings.language = lang
        c = self.conversation(conv)
        if c.brain is not None:
            await c.brain.set_language(lang)
        c.check_alive()

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
        c.check_alive()

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
        if value and reason == "never_listed":
            await self.never_list(conv)
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
        path = self._ws_path(raw)
        await self.workspaces.remove(path)
        if not self.profiles.available:
            return
        for p in self.profiles.rows():  # W9.5: a removed workspace is nobody's folder any more
            if p.default_workspace and resolve_user_path(p.default_workspace, self.paths.home) == path:
                with _profile_errors():
                    np = await self.profiles.update(p.channel, p.chat_id, {"default_workspace": None})
                self.notify_profile(np.channel, np.chat_id, np)


__all__ = ["AgentPaths", "ApiError", "Conversation", "GapError", "Service"]
