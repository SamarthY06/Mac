"""ChatRunner: everything the user sees in a chat (docs/whatsapp/ARCHITECTURE.md W5.1-W5.5).

One runner drives one channel adapter and one agentd connection (kind=chat). It owns presentation;
agentd owns every decision (D9).

- **Timeline.** 👀 on the request as soon as agentd accepts it; one status message, a reply to the
  request, that says where and what (never how: no tool names, no narration), edited in place at
  most once per `min_edit_interval_s`, in whole minutes; at the end the status is edited INTO the
  answer, files follow with no caption, and 👀 becomes ✅ (❌ on an error, ✋ when stopped).
- **Budget per task** (U27): one reaction slot, one status message, the files, at most one question;
  never more than 6 new messages per task.
- **Commands** with or without "/", exact match only: stop, new chat [in X] / start fresh,
  status / what are you doing, help. Any other text goes to the model: an open question gets the
  raw reply (agentd binds it), mid-turn text is a steer (➕), anything else is an ask.
- **No spam.** One coalesced catch-up per wake, the loop breaker, a restart notice once per uid,
  hello and goodbye once per marker, and the governor in every adapter.
- **Stream position** (V3, B1). After every side effect, `stream.json` holds per conversation
  `{boot, last_applied_seq, running_turn_uid, status_msg_ref, sent_attach_ids, ...}`. A restart
  subscribes with `after: {boot, seq}` and continues the running turn on the same status message;
  `stop` is `interrupt{conv}` (the running turn). On `gap` it rebuilds from `status.unacked`, and
  every finished turn is `ack`ed once applied.

All work on chat state runs in one pump task, in arrival order: agentd events, inbound messages,
timers and the post-connect resync. So there is no lock and no interleaving.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import functools
import logging
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from agentd import schema as s
from agentd.stores import atomic_write_json, read_json
from chatd import context as cx
from chatd import format as fmt
from chatd.agent_client import BACKOFF_S, AgentClient, AgentError, AgentUnavailable
from chatd.channels.base import (
    Cap,
    ChannelAdapter,
    ChannelStatus,
    ChatChanged,
    ContextMessage,
    DestinationRefused,
    InboundMessage,
    LinkState,
    MessageRef,
)
from chatd.context import Ring
from chatd.governor import BudgetExceeded

log = logging.getLogger(__name__)

STATE_FILE = "stream.json"
CATCH_UP_WINDOW_S = 5.0  # replays are gathered this long, then handled as one
STATUS_AFTER_S = 3.0  # the status message appears by now even without a folder or progress yet
TASK_NEW_MESSAGES = 6  # per task; beyond that, edit the status or drop
LOOP_TURNS, LOOP_WINDOW_S = 3, 60.0  # more than 3 Mac-like messages in 60 s pauses the chat
EDIT_ROLLOVER_S = 120.0  # a fresh status message this long before the edit window closes
KEEP = 200  # applied uids, notices and attach ids remembered
NEVER_SUBJECT = "timetable"
CLOSE_GRACE_S = 2.0
FORBIDDEN = s.ErrorCode.FORBIDDEN_CONVERSATION.value

EYES, CHECK, CROSS, HAND, PLUS = "👀", "✅", "❌", "✋", "➕"

# The four commands are exact-match fast paths (W5.3). Every other text goes to the model.
_STOP = {"stop"}
_NEW = re.compile(r"^(?:new chat|start fresh)(?: in (?P<folder>.+))?$")
_STATUS = {"status", "what are you doing"}
_HELP = {"help"}

# The three owner profile commands (W9.3), exact patterns forwarded as `profile_command`.
_ADD = re.compile(r"^add group (?P<subject>.+)$")
_CAP = re.compile(r"^(?P<cmd>allow|disallow) (?P<cap>.+)$")
# a capability by its key ("files_read"), the key with spaces ("files read") or its plain words
_CAP_NAMES = {**{k: k for k in s.CAPABILITIES}, **{k.replace("_", " "): k for k in s.CAPABILITIES},
              **{w.casefold(): k for k, w in s.CAPABILITY_WORDS.items()}}


def never_subject(subject: str) -> bool:
    """A group subject that names the never-list (NFKC, format characters dropped, trimmed,
    case-folded): "Timetable", " TIMETABLE", "Time\u200btable"."""
    t = "".join(c for c in unicodedata.normalize("NFKC", subject) if unicodedata.category(c) != "Cf")
    return t.strip().casefold() == NEVER_SUBJECT


def strip_trigger(text: str, word: str = "Mac") -> str | None:
    """The prefix trigger (W9.6): the trimmed text starts with `word`, any case, then the end, a
    space or one of `, : ! .`; the word and that separator are stripped. "Mac, find jobs" ->
    "find jobs"; None for "Macintosh", "Mac's" and our own "Mac · " echo."""
    m = re.match(rf"\s*{re.escape(word)}(?=$|[\s,:!.])(?!\s*·)[,:!.]?\s*", text, re.IGNORECASE)
    return text[m.end():].strip() if m else None


def parse_command(text: str) -> tuple[str, str] | None:
    """("stop" | "new" | "status" | "help" | "add" | "allow" | "disallow", argument) for a command,
    with or without "/". `add` is only a candidate: it is a command when exactly one group matches."""
    t = " ".join(text.strip().lower().replace("’", "'").split())
    t = t.lstrip("/").rstrip(".!? ").strip()
    if t in _STOP:
        return ("stop", "")
    if t in _STATUS:
        return ("status", "")
    if t in _HELP:
        return ("help", "")
    if _ADD.match(t):
        orig = re.match(r"\s*/?\s*add\s+group\s+(.+?)\s*$", text, re.IGNORECASE | re.DOTALL)
        return ("add", orig.group(1) if orig else "")
    m = _CAP.match(t)
    if m and m.group("cap").casefold() in _CAP_NAMES:  # "allow it" is not a command: the model's
        return (m.group("cmd"), _CAP_NAMES[m.group("cap").casefold()])
    m = _NEW.match(t)
    if m:
        folder = (m.group("folder") or "").strip()
        orig = re.search(r"\bin (.+?)[.!?]*$", text.strip(), re.IGNORECASE)  # keep the folder's own spelling
        return ("new", orig.group(1).strip() if folder and orig else folder)
    return None


def confirm_text(ev: s.ConfirmEvent) -> str:
    """Summary and the model's question (AG10), then "To: <target>", then the detail after a blank
    line (`detail`: set_persona's new text). The fallback question only when the summary asks nothing."""
    summary, detail = ev.summary.rstrip(), ev.detail or ""
    q = str(getattr(ev, "question", "") or "")
    if not q and not summary.endswith("?"):
        q = fmt.CONFIRM_QUESTION
    text = f"{summary} {q}".rstrip()
    if ev.target and ev.target.lower() not in text.lower():
        text += f"\nTo: {ev.target}"
    if detail.strip():
        text += f"\n\n{detail.strip()}"
    return text


def _norm(text: str) -> str:
    return " ".join(fmt.strip_prefix(text).lower().split())


@dataclass
class Turn:
    uid: str
    text: str = ""  # the request text
    request: str = ""  # the request's message id ("" for a turn this runner did not start)
    status: str = ""  # status message id
    status_sent_at: float = 0.0
    shown: str = ""  # text on the status message now
    last_edit: float = 0.0
    started_at: float = 0.0
    folder: str = ""  # display name ("" = general)
    say: str = ""
    note: str = ""  # a progress line that wins over `say` (resume failed, waiting for the desktop)
    waiting_since: float = 0.0
    answer: list[str] = field(default_factory=list)
    attaches: list[dict[str, str]] = field(default_factory=list)  # pending {attach_id, path}
    stopped: bool = False
    reacted: str = ""
    new_msgs: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Turn:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChatState:
    conv: str
    chat_id: str
    boot: str = ""
    last_applied_seq: int = 0
    turn: Turn | None = None
    sent_attach_ids: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)  # finished turns fully shown
    notified: list[str] = field(default_factory=list)  # restart notices handed out (uids)
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)  # qid -> {msg, kind, target, at}
    restart_msg: str = ""  # the message carrying the restart notice; the restart question binds to it
    last: dict[str, Any] = field(default_factory=dict)  # {text, folder, at} of the last finished task
    markers: list[str] = field(default_factory=list)  # hello / goodbye, written before the send (N-d)
    loop_noted: bool = False
    prompts: list[str] = field(default_factory=list)  # message ids of every question prompt Mac posted
    members: int = 0  # the chat's member count last seen (a rise in a shared chat pauses it, W9.7)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["running_turn_uid"] = self.turn.uid if self.turn else ""
        d["status_msg_ref"] = self.turn.status if self.turn else ""
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], conv: str, chat_id: str) -> ChatState:
        known = {f.name for f in fields(cls)} - {"conv", "chat_id", "turn"}
        st = cls(conv, chat_id, **{k: v for k, v in d.items() if k in known})
        if isinstance(d.get("turn"), dict):
            st.turn = Turn.from_dict(d["turn"])
        return st

    def remember(self, name: str, value: str) -> None:
        items: list[str] = getattr(self, name)
        items.append(value)
        del items[:-KEEP]


Job = Callable[[], Awaitable[None]]


class ChatRunner:
    def __init__(self, adapter: ChannelAdapter, *, sock: Path, token: str, state_dir: Path | None,
                 clock: Callable[[], float] = time.time,
                 catch_up_window_s: float = CATCH_UP_WINDOW_S, status_after_s: float = STATUS_AFTER_S,
                 backoff: tuple[float, ...] = BACKOFF_S, assistant_name: str | None = None) -> None:
        self.adapter = adapter
        self.assistant_name = assistant_name or ""  # never taggable (V4); None: the voice config's name
        # The chats come from agentd only (W9.8): a read-only copy of this channel's enabled
        # profiles, from `profile_list` after every hello, then `profile_changed`. Empty until then.
        self.profiles: dict[str, s.Profile] = {}  # chat id -> enabled profile
        self.profiles_rev = -1
        self.chats: dict[str, str] = {}  # conv -> chat id, one per enabled profile
        self.path = Path(state_dir) / STATE_FILE if state_dir is not None else None  # None: test mode
        self.clock = clock
        self.catch_up_window_s = catch_up_window_s
        self.status_after_s = status_after_s
        self.client = AgentClient(sock, token, [], on_event=self._on_event, on_connect=self._on_connect,
                                  on_notice=self._on_notice, backoff=backoff)  # agentd grants the live set
        self.state: dict[str, ChatState] = {}
        self._saved: dict[str, Any] = {}  # stream.json entries of chats not enabled now (kept for later)
        self.rings: dict[str, Ring] = {}  # chat id -> recent messages (memory only, W10.2)
        self.members: dict[str, list[dict[str, Any]]] = {}  # chat id -> [{jid, number, name}] (W10.4)
        self._unprefixed: set[str] = set()  # quoted replies let through without the prefix: never commands
        self.link: ChannelStatus | None = None
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._timers: dict[tuple[str, str], asyncio.TimerHandle] = {}
        self._turn_no = 0
        self._pending: dict[str, list[InboundMessage]] = {}  # arrived while agentd was away
        self._batch: dict[str, list[InboundMessage]] = {}  # replays being gathered
        self._recent_out: collections.deque[str] = collections.deque(maxlen=50)
        self._own_like: dict[str, collections.deque[float]] = {}
        self.fatal: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------------------------------------
    async def start(self) -> None:
        if not self.assistant_name:
            self.assistant_name = await asyncio.to_thread(_assistant_name)
        raw = await asyncio.to_thread(read_json, self.path, {}) if self.path else {}
        saved = raw.get("convs", {}) if isinstance(raw, dict) else {}
        self._saved = {k: v for k, v in saved.items() if isinstance(v, dict)}
        self._tasks.append(asyncio.create_task(self._pump(), name="chatd:pump"))
        if hasattr(self.adapter, "on_connected"):  # after every bridge start or `connected` (W9.7, W9.8)
            self.adapter.on_connected = self._bridge_connected
        await self.adapter.start(self._sink, self._on_link, self._on_chat_changed, on_context=self._context_sink)
        self._tasks.append(asyncio.create_task(self._run_client(), name="chatd:agent"))

    async def close(self) -> None:
        for h in self._timers.values():
            h.cancel()
        self._timers.clear()
        await self.client.close()
        with contextlib.suppress(TimeoutError):  # finish the job in hand: a sent message is saved with its id
            await asyncio.wait_for(self._q.join(), CLOSE_GRACE_S)
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        await self.adapter.close()

    async def settle(self) -> None:
        """Wait until every queued job has run (tests)."""
        await self._q.join()

    async def _run_client(self) -> None:
        try:
            await self.client.run()
        except AgentError as e:
            log.error("chatd: agentd refused this client (%s); stopping", e.code)
            self.fatal = e

    async def _pump(self) -> None:
        while True:
            job = await self._q.get()
            try:
                await job()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("chatd: a job failed")
            finally:
                self._q.task_done()

    def _enqueue(self, fn: Callable[..., Awaitable[None]], *args: Any) -> None:
        self._q.put_nowait(functools.partial(fn, *args))

    def _later(self, conv: str, name: str, delay: float, fn: Callable[..., Awaitable[None]], *args: Any) -> None:
        key = (conv, name)
        old = self._timers.pop(key, None)
        if old is not None:
            old.cancel()

        def fire() -> None:
            self._timers.pop(key, None)
            self._enqueue(fn, *args)

        self._timers[key] = asyncio.get_running_loop().call_later(max(0.0, delay), fire)

    def _cancel(self, conv: str, *names: str) -> None:
        for n in names:
            h = self._timers.pop((conv, n), None)
            if h is not None:
                h.cancel()

    async def _save(self) -> None:
        if self.path is not None:
            snap = {"version": 1, "convs": {**self._saved, **{c: st.to_dict() for c, st in self.state.items()}}}
            await asyncio.to_thread(atomic_write_json, self.path, snap)

    async def _call(self, op: str, **args: Any) -> Any:
        """An op whose failure only costs a retry later (ack, question_posted, pause)."""
        try:
            return await self.client.call(op, **args)
        except AgentUnavailable:
            return None
        except AgentError as e:
            log.info("chatd: %s failed (%s)", op, e.code)
            if e.code == FORBIDDEN and "conv" in args:
                self._enqueue(self._forbidden, args["conv"])
            return None

    # -- agentd connection -------------------------------------------------------------------------------
    def _on_event(self, ev: s.Event) -> None:
        self._enqueue(self._apply, ev)

    async def _on_connect(self, hello: s.HelloData) -> None:
        notices: dict[str, list[s.InterruptedTurn]] = {}
        for it in hello.interrupted:
            notices.setdefault(it.conv, []).append(it)
        data: s.ProfileListData = await self.client.call("profile_list")
        self.profiles_rev = data.rev
        self._retarget({p.chat_id: p for p in data.profiles if p.channel == self.adapter.name and p.enabled})
        await self._push_allowed()
        for conv in list(self.chats):
            await self._subscribe(conv, notices.get(conv, []))
        self._enqueue(self._flush_pending)
        if self.link is not None and self.link.state is LinkState.CONNECTED:
            self._enqueue(self._check_members)  # members may have changed while agentd was away

    async def _subscribe(self, conv: str, notices: list[s.InterruptedTurn]) -> None:
        st = self.state.get(conv)
        if st is None:
            return
        after = s.Position(boot=st.boot, seq=st.last_applied_seq) if st.boot else None
        try:
            pos = await self.client.call("subscribe", conv=conv, after=after)
            fresh = after is None
        except AgentError as e:
            if e.code != s.ErrorCode.GAP.value:
                raise
            log.info("chatd: stream gap on %s; rebuilding from status", conv)
            pos = await self.client.call("subscribe", conv=conv)
            fresh = True
        if fresh:  # live from here; with `after`, the replay moves the position itself
            if st.boot == pos.boot:
                st.last_applied_seq = max(st.last_applied_seq, pos.seq)
            else:
                st.boot, st.last_applied_seq = pos.boot, pos.seq
        snap_at = self.clock()
        status = await self.client.call("status", conv=conv)
        self._enqueue(self._resync, conv, status, notices, snap_at)

    # -- profiles (W9.6-W9.8) ----------------------------------------------------------------------------
    def _retarget(self, enabled: dict[str, s.Profile]) -> list[str]:
        """Make the enabled profiles this runner's chats; returns the conversations that are new.
        A chat that left keeps its stream.json entry for when it comes back."""
        old = set(self.profiles)
        self.profiles = enabled
        for cid in old - enabled.keys():
            conv = f"{self.adapter.name}:{cid}"
            st = self.state.pop(conv, None)
            if st is not None:
                self._saved[conv] = st.to_dict()
            self.chats.pop(conv, None)
            self._pending.pop(conv, None)
            self._batch.pop(conv, None)
            self._cancel(conv, "flush", "status", "catch_up")
            log.info("chatd: %s is no longer enabled", conv)
        added = []
        for cid in enabled.keys() - old:
            conv = f"{self.adapter.name}:{cid}"
            d = self._saved.pop(conv, None)
            self.state[conv] = ChatState.from_dict(d, conv, cid) if isinstance(d, dict) else ChatState(conv, cid)
            self.chats[conv] = cid
            added.append(conv)
        return added

    async def _push_allowed(self) -> None:
        """The adapter's allowlist is the enabled profiles (W9.8 step 4), its context set those with
        `read_chat` (W10.6). A chat that left the context set loses its ring and members at once.
        Under test mode the adapter keeps its fixed lists and only logs."""
        reading = self._reading()
        for cid in set(self.rings) - reading:
            self.rings.pop(cid, None)
        for cid in set(self.members) - reading:
            self.members.pop(cid, None)
        try:
            await self.adapter.set_allowed(frozenset(self.profiles), frozenset(reading))
        except Exception:
            log.exception("chatd: updating the channel's allowlist failed")
        for cid in reading - set(self.members):
            await self._refresh_members(cid)

    def _reading(self) -> set[str]:
        return {cid for cid, p in self.profiles.items() if p.capabilities.read_chat}

    # -- group context (W10) -----------------------------------------------------------------------------
    def _ring_add(self, chat_id: str, msg_ref: str, **kw: Any) -> None:
        if chat_id in self._reading():
            self.rings.setdefault(chat_id, Ring()).add(msg_ref, at=kw.pop("at", None) or self.clock(), **kw)

    async def _on_context(self, cm: ContextMessage) -> None:
        """Context is data only: it enters the ring and nothing else (W10.1). Never the parser,
        reply, steer or ask."""
        if cm.deleted:  # deleted on WhatsApp: never sent to the model again
            self.rings.setdefault(cm.chat_id, Ring()).remove(cm.ref.message_id)
        elif cm.chat_id in self.profiles:  # the adapter emits context only for its context set
            self.rings.setdefault(cm.chat_id, Ring()).add_context(cm)

    async def _refresh_members(self, chat_id: str) -> None:
        try:
            self.members[chat_id] = list(await self.adapter.members(chat_id))
        except Exception as e:  # noqa: BLE001 (members are a convenience: plain-text names without them)
            log.info("chatd: members unavailable (%s)", type(e).__name__)

    def _context(self, st: ChatState, msg: InboundMessage) -> s.AskContext | None:
        """`ask.context`: members and the whole ring (minus the trigger) when the chat has
        `read_chat`; the quoted message always (your choice)."""
        reading = st.chat_id in self._reading()
        ring = self.rings.get(st.chat_id) if reading else None
        names = [cx.display_name(m) for m in self.members.get(st.chat_id, [])][:s.CONTEXT_MAX_MEMBERS] if reading else []
        msgs = ring.messages(exclude=msg.ref.message_id) if ring is not None else []
        q = cx.quoted(msg.reply_to) if not msg.reaction else None
        if not (names or msgs or q):
            return None
        return s.AskContext(members=names or None, messages=msgs or None, quoted=q)

    async def _maybe_history(self, st: ChatState, msg: InboundMessage) -> None:
        """After the first accepted ask with an empty ring, ask the phone for 30 earlier messages;
        they arrive later as history context (W10.3). The adapter and bridge own the bounds (context
        set, once an hour, budget) and answer False when they refuse."""
        ring = self.rings.get(st.chat_id)
        if ring is not None and (len(ring) or ring.fresh):
            return
        try:
            await self.adapter.request_history(st.chat_id, msg.ref, cx.HISTORY_COUNT)
        except Exception as e:  # noqa: BLE001 (history is best effort; the ring fills from live messages)
            log.info("chatd: history request failed (%s)", type(e).__name__)

    def _mentions(self, st: ChatState, text: str) -> tuple[str, tuple[str, ...]]:
        members = self.members.get(st.chat_id) if st.chat_id in self._reading() else None
        if not members:
            return text, ()
        me = self.link.account if self.link is not None else ""
        own = {str(m.get("name") or "").strip().casefold() for m in members if me and m.get("jid") == me}
        others = [m for m in members if not me or m.get("jid") != me]  # never tags the owner's own account
        reserved = cx.RESERVED_NAMES | {self.assistant_name.strip().casefold()} | own
        return cx.resolve_mentions(text, others, frozenset(reserved - {""}))

    def _on_notice(self, notice: Any) -> None:
        if isinstance(notice, s.ProfileChanged):
            self._enqueue(self._profile_changed, notice)

    async def _profile_changed(self, n: s.ProfileChanged) -> None:
        """Ignore rev <= ours, apply rev + 1, re-list on a bigger jump (W9.8 step 3)."""
        if n.rev <= self.profiles_rev:
            return
        if n.rev > self.profiles_rev + 1:
            await self._relist()
            return
        self.profiles_rev = n.rev
        if n.channel != self.adapter.name:
            return
        enabled = dict(self.profiles)
        if n.deleted or n.profile is None or not n.profile.enabled:
            enabled.pop(n.chat_id, None)
        else:
            enabled[n.chat_id] = n.profile
        await self._apply_profiles(enabled)

    async def _relist(self) -> None:
        try:
            data: s.ProfileListData = await self.client.call("profile_list")
        except AgentError as e:
            log.info("chatd: profile_list failed (%s)", e.code)
            return
        self.profiles_rev = data.rev
        await self._apply_profiles({p.chat_id: p for p in data.profiles
                                    if p.channel == self.adapter.name and p.enabled})

    async def _apply_profiles(self, enabled: dict[str, s.Profile]) -> None:
        added = self._retarget(enabled)
        await self._push_allowed()
        await self._save()
        for conv in added:
            try:
                await self._subscribe(conv, [])
            except AgentError as e:
                log.info("chatd: subscribing %s failed (%s)", conv, e.code)

    async def _bridge_connected(self) -> None:
        """The adapter re-sent its allowlist after a bridge start or `connected` (W9.8 step 5);
        now compare the members (W9.7)."""
        self._enqueue(self._check_members)
        for cid in self._reading():
            self._enqueue(self._refresh_members, cid)

    async def _check_members(self) -> None:
        try:
            chats = await self.adapter.list_chats()
        except Exception as e:  # noqa: BLE001 (the bridge may be gone again; the next connect retries)
            log.info("chatd: list_chats failed (%s)", type(e).__name__)
            return
        for c in chats:
            cid = str(c.get("id", ""))
            if cid in self.profiles:
                await self._members(cid, bool(c.get("owner_only", False)), c.get("members"))

    async def _members(self, chat_id: str, owner_only: bool, members: Any) -> None:
        """A member was added (owner-only -> shared, or more members in a shared chat): pause the
        chat until the owner acts in the app (W9.7). A member leaving changes nothing."""
        conv = f"{self.adapter.name}:{chat_id}"
        st, p = self.state.get(conv), self.profiles.get(chat_id)
        if st is None or p is None:
            return
        n = members if isinstance(members, int) and not isinstance(members, bool) else 0
        added = (not owner_only and not p.shared) or (p.shared and bool(st.members) and n > st.members)
        if n:
            st.members = n
        await self._save()
        if added:
            log.warning("chatd: %s has a new member; pausing it", conv)
            await self._pause(conv, "members_changed")

    async def _resync(self, conv: str, status: s.ConversationStatus, notices: list[s.InterruptedTurn],
                      snap_at: float) -> None:
        """After (re)connecting and the replay: restart notices, finished turns not yet shown
        (V3), the running turn (B1) and the open question."""
        st = self.state.get(conv)
        if st is None:
            return
        for it in notices:
            await self._restart_notice(st, it.uid, it.text)
        for u in status.unacked:
            if u.turn_uid not in st.applied:
                t = st.turn if st.turn and st.turn.uid == u.turn_uid else Turn(uid=u.turn_uid, started_at=self.clock())
                t.answer = [u.answer] if u.answer else []
                for a in u.attaches:
                    if a.attach_id not in st.sent_attach_ids and all(p["attach_id"] != a.attach_id for p in t.attaches):
                        t.attaches.append({"attach_id": a.attach_id, "path": a.path})
                await self._finish(st, t, "done")
            else:
                await self._call("ack", conv=conv, turn_uid=u.turn_uid)
        r = status.running
        if r is not None and (st.turn is None or st.turn.uid != r.uid):
            st.turn = Turn(uid=r.uid, text=r.text, started_at=self.clock() - r.minutes * 60,
                           folder=fmt.folder_name(r.folder))
        elif r is None and st.turn is not None and st.turn.started_at <= snap_at:
            st.turn = None  # it ended while we were away and left nothing to show
        q = status.question
        for qid, rec in list(st.questions.items()):
            if (q is None or qid != q.question_id) and float(rec.get("at", 0)) <= snap_at:
                st.questions.pop(qid)  # closed while we were away
        if q is not None:
            await self._post_question(st, q.question_id, q.kind.value, q.text)
        if not status.paused:
            st.loop_noted = False
        await self._save()

    async def _flush_pending(self) -> None:
        for conv, msgs in list(self._pending.items()):
            self._pending.pop(conv, None)
            if len(msgs) == 1:
                await self._inbound(msgs[0])
            else:
                await self._catch_up_now(conv, msgs)

    # -- events --------------------------------------------------------------------------------------------
    def _turn(self, st: ChatState, uid: str | None) -> Turn | None:
        if not uid or uid in st.applied:
            return None
        if st.turn is not None and st.turn.uid != uid and st.turn.request:
            return Turn(uid=uid)  # the tail of a turn stopped before our newer one: shown by nobody
        if st.turn is None or st.turn.uid != uid:  # a turn this runner did not start: adopt it
            st.turn = Turn(uid=uid, started_at=self.clock())
        return st.turn

    async def _apply(self, ev: s.Event) -> None:
        st = self.state.get(ev.conv)
        if st is None or (ev.boot == st.boot and ev.seq <= st.last_applied_seq):
            return
        st.boot, st.last_applied_seq = ev.boot, ev.seq
        t = self._turn(st, getattr(ev, "turn_uid", None))
        match ev:
            case s.StatusEvent() if isinstance(ev.data.get("compacted"), dict):
                c = ev.data["compacted"]  # Claude compacted the session: logged, never posted (W10.7)
                log.info("chatd: %s compacted (%s, %s -> %s tokens)", st.conv, c.get("trigger"), c.get("pre_tokens"),
                         c.get("post_tokens"))
            case s.StatusEvent() if isinstance(ev.data.get("profile_command"), dict):
                said = str(ev.data["profile_command"].get("message") or "")  # agentd's result of a bound yes
                if said:
                    await self._send(st, None, said)
            case s.TextEvent() if t is not None:
                if ev.text and not ev.tool_follows:
                    t.answer.append(ev.text)
            case s.ToolStartEvent() if t is not None:
                t.answer.clear()  # the text so far was a step line; tools are never shown
            case s.ToolEndEvent() if t is not None and t.waiting_since:
                t.note, t.waiting_since = "", 0.0  # the desktop wait is over
                await self._progress(st, t)
            case s.SayEvent() if t is not None:
                t.say, t.note, t.waiting_since = ev.text.strip(), "", 0.0
                await self._progress(st, t)
            case s.StatusEvent() if t is not None:
                await self._status_event(st, t, ev.data)
            case s.ConfirmEvent():
                await self._post_question(st, ev.question_id, "confirm", confirm_text(ev), ev.target)
            case s.QuestionEvent():
                await self._post_question(st, ev.question_id, ev.kind.value, ev.text)
            case s.QuestionClosedEvent():
                await self._close_question(st, ev.question_id, ev.outcome)
            case s.AttachEvent() if t is not None:
                if ev.attach_id not in st.sent_attach_ids and all(a["attach_id"] != ev.attach_id for a in t.attaches):
                    t.attaches.append({"attach_id": ev.attach_id, "path": ev.path})
            case s.DoneEvent() if t is not None:
                await self._finish(st, t, "stopped" if ev.data.interrupted else "done")
            case s.ErrorEvent() if t is not None:
                await self._finish(st, t, "error", ev.kind.value, ev.message)
            case s.InterruptedEvent():
                await self._restart_notice(st, ev.uid, ev.text)
        await self._save()

    async def _status_event(self, st: ChatState, t: Turn, data: dict[str, Any]) -> None:
        if "folder" in data:
            t.folder = fmt.folder_name(str(data.get("folder") or ""))
            await self._progress(st, t, create=bool(t.folder))  # 👀 is the ack; no bare "On it"
        if data.get("session_fallback"):
            t.note = fmt.RESUME_FAILED
            await self._progress(st, t)
        if data.get("waiting_for") == "desktop":
            t.waiting_since = t.waiting_since or self.clock()
            t.note = fmt.DESKTOP_WAIT
            await self._progress(st, t)

    # -- the status message -------------------------------------------------------------------------------
    def _compose(self, t: Turn) -> str:
        now = self.clock()
        head = fmt.ON_IT + (f", in {t.folder}" if t.folder else "")
        m = fmt.minutes(now - t.started_at)
        line = t.note or t.say
        if line:
            return f"{head}\n{line} · {m} min" if m else f"{head}\n{line}"
        return f"{head} · {m} min" if m else head

    async def _progress(self, st: ChatState, t: Turn, *, create: bool = False) -> None:
        if t is not st.turn:
            return  # a stopped turn's tail: never shown
        if t.status:
            await self._edit_status(st, t)
        elif create or t.say or t.note:
            await self._open_status(st, t)

    async def _open_status(self, st: ChatState, t: Turn) -> None:
        if t.status:
            return
        text = self._compose(t)
        ref = await self._send(st, t, text, reply_to=t.request)
        if ref is not None:
            now = self.clock()
            t.status, t.status_sent_at, t.shown, t.last_edit = ref.message_id, now, fmt.outbound(text), now
            await self._save()  # the status id is on disk before anything else can happen

    async def _status_due(self, conv: str, uid: str) -> None:
        st = self.state.get(conv)
        t = st.turn if st is not None else None
        if st is not None and t is not None and t.uid == uid and not t.status and not st.questions:
            await self._open_status(st, t)
            await self._save()

    async def _flush_status(self, conv: str, uid: str) -> None:
        st = self.state.get(conv)
        if st is not None and st.turn is not None and st.turn.uid == uid and st.turn.status:
            await self._edit_status(st, st.turn)
            await self._save()

    async def _edit_status(self, st: ChatState, t: Turn, text: str | None = None, *, final: bool = False) -> bool:
        """Show `text` (default: the composed progress) on the status message. Progress edits are
        throttled to one per `min_edit_interval_s` and only on change; near the end of the edit
        window a fresh status message takes over. A final text is never throttled."""
        body = fmt.outbound(text if text is not None else self._compose(t))
        if t.status and body == t.shown:
            return True
        lim, now = self.adapter.limits, self.clock()
        if not final and now - t.last_edit < lim.min_edit_interval_s:
            self._later(st.conv, "flush", t.last_edit + lim.min_edit_interval_s - now, self._flush_status,
                        st.conv, t.uid)
            return False
        self._cancel(st.conv, "flush")
        can_edit = Cap.EDIT in self.adapter.caps and lim.edit_window_s > 0
        margin = 0.0 if final else EDIT_ROLLOVER_S
        ok = False
        if can_edit and t.status and now - t.status_sent_at < lim.edit_window_s - margin:
            ok = await self._edit(st, t.status, body)
        if not ok:
            if not final and (not can_edit or not t.status):
                return False  # progress without edits is dropped, never a stream of messages
            ref = await self._send(st, t, body, reply_to=t.request, final=final)
            if ref is None:
                return False
            t.status, t.status_sent_at = ref.message_id, now
        t.shown, t.last_edit = body, now
        return True

    # -- the end of a turn -------------------------------------------------------------------------------
    async def _finish(self, st: ChatState, t: Turn, how: str, kind: str = "", message: str = "") -> None:
        if t.uid in st.applied:
            return
        self._cancel(st.conv, "flush", "status")
        lim = self.adapter.limits
        answer = fmt.render("".join(t.answer).strip(), lim.text_format)
        quiet = not (t.status or t.request or answer or t.attaches)  # a turn nobody saw: nothing to say
        emoji = {"done": CHECK, "stopped": HAND, "error": CROSS}[how]
        if how == "done" and not quiet:
            if answer or not t.attaches or t.status:
                chunks, full = fmt.plan_answer(answer, lim.max_text) if answer else ([fmt.DONE], None)
                if full is not None:
                    log.warning("chatd: the answer was too long for chat; sending it as a file (AG21)")
                first, tagged = self._mentions(st, chunks[0])
                if tagged and t.status:
                    # WhatsApp pings nobody for a tag added by an edit: an answer that tags someone is a new
                    # message, and the status stays as it was (no edit, so the answer is never shown twice)
                    ref = await self._send(st, t, first, reply_to=t.request, final=True, mentions=tagged)
                    answer_id = ref.message_id if ref else ""
                else:
                    await self._show_final(st, t, first, tagged)
                    answer_id = t.status
                for c in chunks[1:]:
                    c, tagged = self._mentions(st, c)
                    await self._send(st, t, c, mentions=tagged)
                if full is not None:
                    await self._send_md(st, t, full)
                if answer and answer_id:  # Mac's reply joins the ring for later asks (W10.2)
                    self._ring_add(st.chat_id, answer_id, mac=True, text=answer)
        elif how == "stopped" and t.status:
            await self._show_final(st, t, fmt.STOPPED)
        elif how == "error":
            await self._show_final(st, t, fmt.error_line(kind, message))
        for a in list(t.attaches):
            if a["attach_id"] not in st.sent_attach_ids and await self._send_file(st, t, Path(a["path"])):
                st.remember("sent_attach_ids", a["attach_id"])
                await self._save()
            t.attaches.remove(a)
        if t.request:
            await self._react(st, t, emoji)
        if how == "done" and t.text:
            st.last = {"text": t.text, "folder": t.folder, "at": self.clock()}
        st.remember("applied", t.uid)
        if st.turn is t or (st.turn is not None and st.turn.uid == t.uid):
            st.turn = None
        await self._save()
        await self._call("ack", conv=st.conv, turn_uid=t.uid)

    async def _show_final(self, st: ChatState, t: Turn, text: str, mentions: tuple[str, ...] = ()) -> None:
        """The final text on the status message, or a new message. Tags only ever go on a new message."""
        if t.status:
            await self._edit_status(st, t, text, final=True)
        else:
            ref = await self._send(st, t, text, reply_to=t.request, final=True, mentions=mentions)
            if ref is not None:
                t.status, t.status_sent_at, t.shown = ref.message_id, self.clock(), fmt.outbound(text)

    async def _send_md(self, st: ChatState, t: Turn, full: str) -> None:
        if self.path is None:
            return
        out = self.path.parent / "outbox" / f"answer-{abs(hash(t.uid)) % 10**8}.md"
        await asyncio.to_thread(_write_text, out, full)
        await self._send_file(st, t, out)

    # -- questions -------------------------------------------------------------------------------------------
    async def _post_question(self, st: ChatState, qid: str, kind: str, text: str, target: str = "") -> None:
        if qid in st.questions:
            return
        mid = ""
        if kind == "restart" and st.restart_msg:  # the stopped turn's status becomes agentd's question
            if await self._edit(st, st.restart_msg, text):
                mid = st.restart_msg
            st.restart_msg = ""
        if not mid:
            ref = await self._send(st, st.turn, text, final=True)  # a question is never dropped
            if ref is None:
                return
            mid = ref.message_id
        if st.turn is not None and st.turn.status:
            st.turn.status = ""  # UX3: the answer lands below the question, never above it
        st.questions[qid] = {"msg": mid, "kind": kind, "target": target, "at": self.clock(), "text": text}
        if mid not in st.prompts:
            st.remember("prompts", mid)
        await self._save()
        await self._call("question_posted", conv=st.conv, question_id=qid, msg_ref=mid)

    async def _close_question(self, st: ChatState, qid: str, outcome: s.CloseOutcome) -> None:
        q = st.questions.pop(qid, None)
        if q is None:
            return
        text = None
        if outcome is s.CloseOutcome.SKIPPED:
            text = fmt.SKIPPED
        elif q["kind"] == "confirm":
            text = {s.CloseOutcome.APPROVED: fmt.APPROVED, s.CloseOutcome.DENIED: fmt.DECLINED,
                    s.CloseOutcome.EXPIRED: fmt.NO_ANSWER, s.CloseOutcome.CHANGED: fmt.CHANGING}.get(outcome)
        elif outcome is s.CloseOutcome.EXPIRED:  # profile, folder, route, restart: stop asking
            text = fmt.NO_ANSWER_LEFT
        elif outcome is s.CloseOutcome.DENIED:
            text = fmt.DECLINED
        if text is not None:  # the question stays readable; the outcome goes below it (a record in groups)
            asked = str(q.get("text") or "")
            await self._edit(st, q["msg"], f"{fmt.prefixed(asked)}\n\n{fmt.strip_prefix(text)}" if asked else text)
        t = st.turn
        if t is not None and not t.status and not st.questions and t.request:
            self._later(st.conv, "status", self.status_after_s, self._status_due, st.conv, t.uid)

    async def _restart_notice(self, st: ChatState, uid: str, text: str) -> None:
        """agentd restarted under a running turn: ❌ once, and the turn's status message is kept for
        agentd's restart question, which follows (its text is agentd's, never chatd's)."""
        if uid in st.notified:
            return
        st.remember("notified", uid)
        t = st.turn if st.turn is not None and st.turn.uid == uid else None
        if t is not None:
            st.restart_msg = t.status
            if t.request:
                await self._react(st, t, CROSS)
            self._cancel(st.conv, "flush", "status")
            st.turn = None
        await self._save()

    # -- inbound ---------------------------------------------------------------------------------------------
    async def _context_sink(self, cm: ContextMessage) -> None:
        self._enqueue(self._on_context, cm)

    async def _sink(self, msg: InboundMessage) -> None:
        gated = self._gate(msg)
        if gated is not None:
            self._enqueue(self._inbound, gated)

    def _gate(self, msg: InboundMessage) -> InboundMessage | None:
        """The trigger (W9.6), before everything else (catch-up, loop breaker, commands). A chat
        without an enabled profile gets nothing. In a prefix chat (every shared chat) only these
        count: text starting with the profile's prefix word (stripped), and a reply or 👍 on one
        of Mac's own question prompts (agentd binds it; never a command), and a reply to any other
        message of Mac's (a steer mid-turn, else a follow-up ask). The rest is ignored
        silently: no reaction, no reply, no log line with text, not marked as handled."""
        p = self.profiles.get(msg.chat_id)
        if p is None or msg.channel != self.adapter.name:
            return None
        if not (p.shared or p.trigger == "prefix"):
            return msg
        st = self.state.get(f"{msg.channel}:{msg.chat_id}")
        on_prompt = (st is not None and msg.reply_to is not None
                     and msg.reply_to.ref.message_id in st.prompts)
        if msg.reaction:
            return msg if on_prompt else None
        rest = strip_trigger(msg.text, p.prefix or "Mac") if not msg.unsupported else None
        if rest is not None and (rest or msg.attachments):
            return dataclasses.replace(msg, text=rest)
        if on_prompt and msg.text.strip():
            self._unprefixed.add(msg.ref.message_id)
            return msg
        if msg.reply_to is not None and msg.reply_to.from_mac and (msg.text.strip() or msg.attachments):
            return msg  # a reply to one of Mac's messages is for Mac: mid-turn it joins the task (➕)
        log.debug("chatd: %s ignored (no prefix)", msg.ref.message_id)
        if msg.text.strip() or msg.attachments:  # your own non-trigger message: context (W10.2)
            media = {"kind": msg.attachments[0].kind, "name": msg.attachments[0].name} if msg.attachments else None
            self._ring_add(msg.chat_id, msg.ref.message_id, owner=True, at=msg.sent_at, text=msg.text.strip(),
                           media=media)
        return None

    def _on_link(self, status: ChannelStatus) -> None:
        self.link = status
        log.info("chatd: channel %s is %s", self.adapter.name, status.state.value)

    async def _on_chat_changed(self, change: ChatChanged) -> None:
        if change.chat_id in self.profiles and never_subject(change.subject):
            self._enqueue(self._never_listed, change.chat_id)  # renamed to Timetable: never again
            return
        self._enqueue(self._members, change.chat_id, change.owner_only, change.extra.get("members"))
        if change.chat_id in self._reading():
            self._enqueue(self._refresh_members, change.chat_id)

    async def _never_listed(self, chat_id: str) -> None:
        """A profiled group renamed to a never subject: agentd makes it a sticky never-list entry,
        tears it down and disables it (pause reason never_listed); then re-list."""
        conv = f"{self.adapter.name}:{chat_id}"
        log.warning("chatd: %s was renamed to a never-listed subject; stopping there", conv)
        await self._call("pause", conv=conv, value=True, reason="never_listed")
        await self._relist()

    async def _pause(self, conv: str, reason: str) -> None:
        await self._call("pause", conv=conv, value=True, reason=reason)

    async def _inbound(self, msg: InboundMessage) -> None:
        conv = f"{msg.channel}:{msg.chat_id}"
        st = self.state.get(conv)
        if st is None:
            return
        if not self.client.connected.is_set():
            self._pending.setdefault(conv, []).append(msg)  # handled once agentd is back (not marked: R4)
            return
        if msg.replayed:
            self._batch.setdefault(conv, []).append(msg)
            if (conv, "catch_up") not in self._timers:
                self._later(conv, "catch_up", self.catch_up_window_s, self._catch_up_due, conv)
            return
        try:
            await self._live(st, msg)
        except AgentUnavailable:
            self._pending.setdefault(conv, []).append(msg)
        except AgentError as e:
            if e.code != FORBIDDEN:
                raise
            await self._forbidden(conv)

    async def _forbidden(self, conv: str) -> None:
        """I3: agentd refuses this chat now (never-listed or disabled without a notice we saw):
        re-list, which drops it and narrows the allowlist. Nothing is said in that chat."""
        log.warning("chatd: agentd refused %s; re-listing the profiles", conv)
        await self._relist()

    async def _handled(self, msg: InboundMessage) -> None:
        with contextlib.suppress(Exception):
            await self.adapter.mark_handled(msg.ref)

    async def _live(self, st: ChatState, msg: InboundMessage) -> None:
        if msg.unsupported:
            if msg.unsupported in ("voice", "audio", "video"):
                await self._send(st, None, fmt.VOICE_NOTE, reply_to=msg.ref.message_id)
            await self._handled(msg)
            return
        if msg.reaction:
            if st.questions and msg.reply_to is not None:
                await self.client.call("reply", conv=st.conv, reaction=msg.reaction,
                                       reaction_on=msg.reply_to.ref.message_id)
            await self._handled(msg)
            return
        text = msg.text.strip()
        if not text and not msg.attachments:
            await self._handled(msg)
            return
        unprefixed = msg.ref.message_id in self._unprefixed
        self._unprefixed.discard(msg.ref.message_id)
        p = self.profiles.get(st.chat_id)
        word = p.prefix if p is not None and p.prefix else "Mac"
        # P2-3: "Mac, stop" is a command in every chat, the trigger word stripped (a prefix chat's gate did it already)
        cmd = None if unprefixed else parse_command(strip_trigger(text, word) or text)
        if cmd is not None and await self._command(st, msg, *cmd):
            await self._handled(msg)
            return
        if await self._loop_breaker(st, text):
            await self._handled(msg)
            return
        if st.questions and text:
            quoted = msg.reply_to.ref.message_id if msg.reply_to else None
            r = await self.client.call("reply", conv=st.conv, text=text, quoted_ref=quoted)
            if r.bound:
                await self._handled(msg)
                return
        t = st.turn
        if t is not None and await self._steer(st, msg, text, t.uid):
            return  # mid-turn text (a question about progress included) is the model's to answer
        await self._ask(st, msg, text)

    async def _steer(self, st: ChatState, msg: InboundMessage, text: str, uid: str | None) -> bool:
        try:
            r = await self.client.call("steer", conv=st.conv, turn_uid=uid, text=text)
        except AgentUnavailable:
            raise
        except AgentError as e:
            if e.code == FORBIDDEN:
                raise
            return False
        if not r.accepted:
            return False
        await self._react_on(st, msg.ref.message_id, PLUS)
        await self._handled(msg)
        return True

    async def _ask(self, st: ChatState, msg: InboundMessage, text: str, *, replayed: bool = False) -> None:
        self._turn_no += 1
        atts = [s.AskAttachment(path=str(a.path), kind=a.kind, name=a.name or a.path.name) for a in msg.attachments]
        meta = s.AskMeta(replayed=replayed, sent_at=msg.sent_at, msg_ref=msg.ref.message_id)
        context = self._context(st, msg)
        try:
            data = await self.client.call("ask", conv=st.conv, turn=self._turn_no, text=text or "(see the file)",
                                          attachments=atts or None, meta=meta, context=context)
        except AgentUnavailable:
            raise
        except AgentError as e:
            if e.code == FORBIDDEN:
                raise
            if e.code == s.ErrorCode.BUSY.value and await self._steer(st, msg, text, None):
                return
            log.info("chatd: ask not accepted (%s)", e.code)  # paused: silent by design
            await self._handled(msg)
            return
        t = Turn(uid=data.turn_uid, text=text, request=msg.ref.message_id, started_at=self.clock())
        st.turn = t
        await self._react(st, t, EYES)
        await self._handled(msg)  # R4: only after agentd accepted it
        await self._maybe_history(st, msg)
        await self._save()
        self._later(st.conv, "status", self.status_after_s, self._status_due, st.conv, t.uid)

    async def _catch_up_due(self, conv: str) -> None:
        await self._catch_up_now(conv, self._batch.pop(conv, []))

    async def _catch_up_now(self, conv: str, msgs: list[InboundMessage]) -> None:
        """Everything missed while asleep becomes one ask: the latest message, with the earlier
        ones as context, marked replayed with its send time. The model decides whether to act or
        ask first (AG5)."""
        st = self.state.get(conv)
        if st is None:
            return
        texts = []
        for m in msgs:
            if m.unsupported or m.reaction or not m.text.strip() or parse_command(m.text):
                await self._handled(m)
            else:
                texts.append(m)
        if not texts:
            return
        latest, earlier = texts[-1], texts[:-1]
        if not earlier and not latest.replayed:
            await self._inbound(latest)
            return
        text = latest.text.strip()
        if earlier:
            said = "\n".join(f'"{m.text.strip()}"' for m in earlier)
            text = f"While I was offline you also wrote:\n{said}\n\nYour latest message:\n{text}"
        try:
            await self._ask(st, latest, text, replayed=any(m.replayed for m in texts))
        except AgentUnavailable:
            self._pending.setdefault(conv, []).extend(texts)
            return
        for m in earlier:
            await self._handled(m)

    async def _loop_breaker(self, st: ChatState, text: str) -> bool:
        """More than 3 messages in 60 s that look like Mac's own: pause the chat, say so once."""
        if not (text.startswith(fmt.PREFIX.strip()) or _norm(text) in self._recent_out):
            return False
        now = self.clock()
        times = self._own_like.setdefault(st.conv, collections.deque())
        times.append(now)
        while times and now - times[0] > LOOP_WINDOW_S:
            times.popleft()
        if len(times) <= LOOP_TURNS:
            return False
        if not st.loop_noted:
            st.loop_noted = True
            await self._call("pause", conv=st.conv, value=True, reason="loop")
            await self._send(st, None, fmt.LOOP_PAUSED)
            await self._save()
        return True

    # -- commands ------------------------------------------------------------------------------------------
    async def _command(self, st: ChatState, msg: InboundMessage, cmd: str, arg: str) -> bool:
        """Run an exact command. False: it was not one after all (an `add group` that names no
        group), and the text goes to the model as a normal message."""
        conv, mid = st.conv, msg.ref.message_id
        if cmd == "help":
            await self._send(st, None, self._help(st), reply_to=mid)
        elif cmd == "stop":
            r = await self.client.call("interrupt", conv=conv)  # the running turn (B1)
            t = st.turn
            if not r.stopped:
                await self._send(st, None, fmt.NOTHING_TO_STOP, reply_to=mid)
            elif t is not None:
                t.stopped = True
                if t.status:
                    await self._edit_status(st, t, fmt.STOPPED, final=True)
                if t.request:
                    await self._react(st, t, HAND)
                else:
                    await self._react_on(st, mid, HAND)
            else:
                await self._react_on(st, mid, HAND)
            await self._save()
        elif cmd == "new":
            if st.turn is not None:
                await self._send(st, None, fmt.STOP_FIRST, reply_to=mid)
                return True
            await self.client.call("new_session", conv=conv, folder=arg or None)
            self.rings[st.chat_id] = Ring(fresh=True)  # a fresh start carries no earlier messages
            await self._send(st, None, fmt.FRESH_IN.format(folder=arg) if arg else fmt.FRESH, reply_to=mid)
        elif cmd == "status":
            data: s.ConversationStatus = await self.client.call("status", conv=conv)
            await self._send(st, None, self._status_reply(st, data), reply_to=mid)
        elif cmd == "add":
            return await self._add_group(st, mid, arg)
        elif cmd in ("allow", "disallow"):
            await self._capability(st, mid, cmd, arg)
        return True

    def _help(self, st: ChatState) -> str:
        """The prefix variant where a bare "stop" is ignored; the folder example is this chat's own."""
        p = self.profiles.get(st.chat_id)
        folder = fmt.folder_name(p.default_workspace or "") if p is not None else ""
        folder = folder or fmt.HELP_FOLDER
        if p is not None and (p.shared or p.trigger == "prefix"):
            return fmt.HELP_PREFIX.format(w=p.prefix or "Mac", folder=folder)
        return fmt.HELP.format(folder=folder)

    async def _add_group(self, st: ChatState, mid: str, subject: str) -> bool:
        """`add group <subject>`: exactly one group whose trimmed, case-folded subject matches
        (W9.3); agentd re-checks the never-set and asks the quoted-yes question."""
        try:
            chats = await self.adapter.list_chats()
        except Exception as e:  # noqa: BLE001 (any bridge failure is the same answer to the owner)
            log.info("chatd: list_chats failed (%s)", type(e).__name__)
            await self._send(st, None, fmt.GROUPS_UNAVAILABLE, reply_to=mid)
            return True
        want = subject.strip().casefold()
        hits = [c for c in chats if not c.get("is_self") and str(c.get("subject", "")).strip().casefold() == want]
        if not hits:
            return False  # P2-2: "add group of five people to the list" is the model's
        if len(hits) > 1:
            await self._send(st, None, fmt.MANY_GROUPS.format(subject=fmt.short(subject.strip())), reply_to=mid)
            return True
        c = hits[0]
        await self._profile_command(st, mid, command="add", subject=str(c.get("subject", "")),
                                    chat_id=str(c.get("id", "")), owner_only=bool(c.get("owner_only", False)))
        return True

    async def _capability(self, st: ChatState, mid: str, cmd: str, cap: str) -> None:
        r = await self._profile_command(st, mid, command=cmd, capability=cap)
        if r is not None and cmd == "disallow" and r.applied:
            await self._send(st, None, fmt.CAPABILITY_OFF.format(name=fmt.capitalized(s.CAPABILITY_WORDS[cap])), reply_to=mid)

    async def _profile_command(self, st: ChatState, mid: str, **args: Any) -> s.ProfileCommandData | None:
        """`add` and `allow` answer with a question that follows as a `question` event."""
        try:
            return await self.client.call("profile_command", conv=st.conv, **args)
        except AgentUnavailable:
            raise
        except AgentError as e:
            if e.code == FORBIDDEN:
                raise
            log.info("chatd: profile_command %s refused (%s)", args.get("command"), e.code)
            text = {s.ErrorCode.NEVER_LISTED.value: fmt.NEVER_LISTED,
                    s.ErrorCode.PROFILE_EXISTS.value: fmt.PROFILE_EXISTS}.get(e.code, fmt.PROFILE_FAILED)
            await self._send(st, None, text, reply_to=mid)
            return None

    def _status_reply(self, st: ChatState, data: s.ConversationStatus) -> str:
        r = data.running
        if r is not None:
            where = f" in {fmt.folder_name(r.folder)}" if fmt.folder_name(r.folder) else ""
            when = f"{r.minutes} min in" if r.minutes else "just started"
            t = st.turn
            say = (t.note or t.say).rstrip(" .") if t is not None and t.uid == r.uid else ""
            # the model's own latest progress line (AG24); the request only when it said nothing yet
            out = f"{say}{where}, {when}." if say else f'Working on "{fmt.short(r.text)}"{where}, {when}.'
        else:
            out = "Nothing running."
            last = st.last
            if last.get("text"):
                where = f" in {last['folder']}" if last.get("folder") else ""
                out += f' Last: "{fmt.short(str(last["text"]))}"{where}, {fmt.ago(self.clock() - float(last["at"]))}.'
        if data.question is not None:
            out += " I'm waiting for your answer above."
        if data.desktop is not None and data.desktop.holder == "voice":
            out += " " + fmt.DESKTOP_HELD
        return out

    # -- hello and goodbye (N-d) ----------------------------------------------------------------------------
    async def say_once(self, chat_id: str, marker: str, text: str) -> bool:
        """Send `text` once per `marker` (e.g. "hello:<jid>:<device>"); the marker is written
        before the send, so a crash can lose it but never repeat it."""
        st = self.state.get(f"{self.adapter.name}:{chat_id}")
        if st is None or marker in st.markers:
            return False
        st.remember("markers", marker)
        await self._save()
        return await self._send(st, None, text) is not None

    # -- outbound helpers ----------------------------------------------------------------------------------
    def _ref(self, st: ChatState, mid: str) -> MessageRef | None:
        return MessageRef(st.chat_id, mid) if mid else None

    async def _send(self, st: ChatState, t: Turn | None, text: str, *, reply_to: str = "",
                    final: bool = False, mentions: tuple[str, ...] = ()) -> MessageRef | None:
        if t is not None and t.new_msgs >= TASK_NEW_MESSAGES and not final:
            log.warning("chatd: task message budget spent; dropped a message")
            return None
        body = fmt.outbound(text)
        try:
            ref = await self.adapter.send_text(st.chat_id, body, self._ref(st, reply_to), mentions=mentions)
        except (BudgetExceeded, DestinationRefused) as e:
            log.warning("chatd: send refused (%s)", type(e).__name__)
            return None
        except Exception:
            log.exception("chatd: send failed")
            return None
        if t is not None:
            t.new_msgs += 1
        self._recent_out.append(_norm(body))
        await self._save()
        return ref

    async def _edit(self, st: ChatState, mid: str, text: str) -> bool:
        body = fmt.outbound(text)
        try:
            ok = await self.adapter.edit_text(MessageRef(st.chat_id, mid), body)  # never tags: edits ping nobody
        except (BudgetExceeded, DestinationRefused) as e:
            log.warning("chatd: edit refused (%s)", type(e).__name__)
            return False
        except Exception:
            log.exception("chatd: edit failed")
            return False
        if ok:
            self._recent_out.append(_norm(body))
            await self._save()
        return ok

    async def _send_file(self, st: ChatState, t: Turn, path: Path) -> bool:
        if t.new_msgs >= TASK_NEW_MESSAGES:
            log.warning("chatd: task message budget spent; a file was not sent")
            return False
        try:
            await self.adapter.send_file(st.chat_id, path)  # no caption (U7)
        except (BudgetExceeded, DestinationRefused) as e:
            log.warning("chatd: file refused (%s)", type(e).__name__)
            return False
        except Exception:
            log.exception("chatd: file send failed")
            return False
        t.new_msgs += 1
        return True

    async def _react(self, st: ChatState, t: Turn, emoji: str) -> None:
        if t.request and t.reacted != emoji:
            await self._react_on(st, t.request, emoji)
            t.reacted = emoji

    async def _react_on(self, st: ChatState, mid: str, emoji: str) -> None:
        if Cap.REACT not in self.adapter.caps:
            return
        try:
            await self.adapter.react(MessageRef(st.chat_id, mid), emoji)
        except Exception:
            log.warning("chatd: reaction failed", exc_info=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(text, encoding="utf-8")


def _assistant_name() -> str:
    """The assistant's name from the voice config (`name`, default "Mac"), read only, never written."""
    from macd import config as mcfg

    try:
        return mcfg.load(write_default=False).name
    except Exception:  # noqa: BLE001 (an unreadable config: the default name is reserved anyway)
        return "Mac"
