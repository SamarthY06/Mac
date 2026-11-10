"""An in-memory channel for tests and dry runs. It keeps the adapter rules (R1-R5) exactly like
a real channel, and shows what the user would see: `messages` (current text after edits),
`reactions` (per message, newest last) and `wire` (every outbound write, in order).

Tests drive it with `inject(...)`, which applies the inbound checks (W5.2) the way a bridge does.
Logs carry ids, kinds and lengths only, never message text.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from chatd.channels.base import (
    Attachment,
    Cap,
    ChannelStatus,
    ChatChanged,
    ChatChangedCallback,
    InboundMessage,
    Limits,
    LinkState,
    MessageRef,
    MessageSink,
    Quoted,
    StatusCallback,
)
from chatd.channels.registry import ChannelContext
from chatd.format import PREFIX

log = logging.getLogger(__name__)


@dataclass
class Write:
    op: Literal["send_text", "edit", "react", "typing", "send_file"]
    chat_id: str
    message_id: str = ""
    text: str = ""
    reply_to: str = ""
    path: str = ""
    at: float = 0.0


@dataclass
class FakeMessage:
    chat_id: str
    message_id: str
    text: str
    sent_at: float
    mine: bool  # sent by Mac
    reply_to: str = ""
    path: str = ""  # a file message
    caption: str = ""
    history: list[str] = field(default_factory=list)  # every text it showed, oldest first


class FakeChannel:
    def __init__(self, cfg: dict[str, Any], ctx: ChannelContext) -> None:
        self.name: str = cfg.get("name", "fake")
        self.caps: Cap = cfg.get("caps", Cap.EDIT | Cap.REACT | Cap.TYPING | Cap.FILES)
        self.limits = Limits(max_text=cfg.get("max_text", 4000), edit_window_s=cfg.get("edit_window_s", 900.0),
                             min_edit_interval_s=cfg.get("min_edit_interval_s", 4.0),
                             text_format=cfg.get("text_format", "whatsapp"))
        self.ctx = ctx
        self.trusted_devices: tuple[int, ...] = tuple(cfg.get("trusted_devices", (0,)))
        self._linked = bool(cfg.get("linked", True))
        self._status = ChannelStatus(LinkState.UNLINKED)
        self._ids = itertools.count(1)
        self.sink: MessageSink | None = None
        self.on_status: StatusCallback | None = None
        self.on_chat_changed: ChatChangedCallback | None = None
        self.wire: list[Write] = []
        self.messages: dict[str, FakeMessage] = {}
        self.reactions: dict[str, list[str]] = {}
        self.downloads = 0  # attachments fetched (never for voice, audio, video, stickers)
        self.chats: list[dict[str, Any]] = list(cfg.get("groups", []))  # what list_chats returns
        self.allowed_sets: list[frozenset[str]] = []  # every set_allowed call (recorded, not applied)
        self.on_connected: Callable[[], Awaitable[None]] | None = None  # after every (re)connect

    # -- lifecycle ---------------------------------------------------------------------------------
    async def start(self, sink: MessageSink, on_status: StatusCallback,
                    on_chat_changed: ChatChangedCallback) -> None:
        self.sink, self.on_status, self.on_chat_changed = sink, on_status, on_chat_changed
        self._set(LinkState.CONNECTED if self._linked else LinkState.UNLINKED)

    async def close(self) -> None:
        self.sink = None

    def status(self) -> ChannelStatus:
        return self._status

    def _set(self, state: LinkState, **kw: Any) -> None:
        self._status = ChannelStatus(state, **kw)
        if self.on_status is not None:
            self.on_status(self._status)

    async def link(self) -> None:
        self._set(LinkState.PAIRING, detail="Scan the QR code on your phone.", qr="fake-qr-1")

    async def complete_pairing(self, account: str = "owner") -> None:  # test driver
        self._linked = True
        self._set(LinkState.CONNECTED, account=account)

    async def reconnect(self) -> None:  # test driver: the transport came back (W9.8 step 5)
        self._set(LinkState.CONNECTED, account="owner")
        if self.on_connected is not None:
            await self.on_connected()

    async def set_allowed(self, chat_ids: frozenset[str]) -> None:
        """Test-only adapter: the allowlist is fixed by its context (like the bridge under --test)."""
        self.allowed_sets.append(frozenset(chat_ids))

    async def go_offline(self) -> None:  # test driver
        self._set(LinkState.OFFLINE, detail="Offline.")

    async def unlink(self) -> None:
        self._linked = False
        self._set(LinkState.UNLINKED)

    async def list_chats(self) -> list[dict[str, Any]]:
        return list(self.chats)

    # -- inbound (what a bridge would deliver) ---------------------------------------------------
    async def inject(self, text: str = "", *, chat_id: str, msg_id: str | None = None, from_owner: bool = True,
                     device: int = 0, replayed: bool = False, sent_at: float | None = None,
                     unsupported: Literal["", "voice", "audio", "video", "sticker"] = "",
                     reaction: str | None = None, reply_to: str | None = None,
                     attachments: tuple[Attachment, ...] = ()) -> bool:
        """Apply the W5.2 checks and deliver to the sink. Returns whether it was delivered."""
        msg_id = msg_id or f"in-{next(self._ids)}"
        sent_at = self.ctx.clock() if sent_at is None else sent_at
        key = f"{chat_id}:{msg_id}"
        why = ""
        if not self.ctx.allowed(chat_id):
            why = "chat"
        elif not from_owner or device not in self.trusted_devices:
            why = "not owner"
        elif (msg_id in self.messages and self.messages[msg_id].mine) or text.strip().startswith(PREFIX.strip()):
            why = "echo"
        elif self.ctx.too_old(sent_at):
            why = "old"
        elif key in self.ctx.seen:
            why = "handled"
        if why or self.sink is None:
            log.debug("fake: inbound %s dropped (%s)", msg_id, why or "not started")
            return False
        if not unsupported:
            self.downloads += len(attachments)
        quoted = None
        if reply_to:
            q = self.messages.get(reply_to)
            quoted = Quoted(MessageRef(chat_id, reply_to), q.text if q else "", from_owner=not (q and q.mine))
        self.messages.setdefault(msg_id, FakeMessage(chat_id, msg_id, text, sent_at, mine=False))
        msg = InboundMessage(channel=self.name, chat_id=chat_id, ref=MessageRef(chat_id, msg_id), text=text,
                             attachments=() if unsupported else attachments, reply_to=quoted,
                             reaction=reaction, unsupported=unsupported, sent_at=sent_at, replayed=replayed)
        log.debug("fake: inbound %s delivered (len %d)", msg_id, len(text))
        await self.sink(msg)
        return True

    async def change_chat(self, change: ChatChanged) -> None:  # test driver
        if self.on_chat_changed is not None and self.ctx.allowed(change.chat_id):
            await self.on_chat_changed(change)

    async def mark_handled(self, ref: MessageRef) -> None:
        await self.ctx.seen.add(f"{ref.chat_id}:{ref.message_id}")

    # -- outbound (R1, R2: destination check, then the governor, then I/O) ---------------------------
    def _new_id(self) -> str:
        n = next(self._ids)
        return f"dry-{n}" if self.ctx.dry_run else f"out-{n}"

    async def send_text(self, chat_id: str, text: str, reply_to: MessageRef | None = None) -> MessageRef:
        self.ctx.check_destination(chat_id)
        await self.ctx.governor.charge("send", chat_id)
        mid = self._new_id()
        if not self.ctx.dry_run:
            rt = reply_to.message_id if reply_to else ""
            self.messages[mid] = FakeMessage(chat_id, mid, text, self.ctx.clock(), True, rt, history=[text])
            self.wire.append(Write("send_text", chat_id, mid, text, rt, at=self.ctx.clock()))
        return MessageRef(chat_id, mid)

    async def edit_text(self, ref: MessageRef, text: str) -> bool:
        self.ctx.check_destination(ref.chat_id)
        m = self.messages.get(ref.message_id)
        if self.ctx.dry_run:
            await self.ctx.governor.charge("edit", ref.chat_id)
            return True
        if (m is None or not m.mine or not self.limits.edit_window_s
                or self.ctx.clock() - m.sent_at > self.limits.edit_window_s):
            return False  # past the window (or not ours): the caller sends a new message
        await self.ctx.governor.charge("edit", ref.chat_id)
        m.text = text
        m.history.append(text)
        self.wire.append(Write("edit", ref.chat_id, ref.message_id, text, at=self.ctx.clock()))
        return True

    async def react(self, ref: MessageRef, emoji: str) -> None:
        self.ctx.check_destination(ref.chat_id)
        await self.ctx.governor.charge("react", ref.chat_id)
        if not self.ctx.dry_run:
            self.reactions.setdefault(ref.message_id, []).append(emoji)
            self.wire.append(Write("react", ref.chat_id, ref.message_id, emoji, at=self.ctx.clock()))

    async def set_typing(self, chat_id: str, on: bool) -> None:
        self.ctx.check_destination(chat_id)
        await self.ctx.governor.charge("typing", chat_id)
        if not self.ctx.dry_run:
            self.wire.append(Write("typing", chat_id, text="on" if on else "off", at=self.ctx.clock()))

    async def send_file(self, chat_id: str, path: Path, caption: str = "") -> MessageRef:
        self.ctx.check_destination(chat_id)
        await self.ctx.governor.charge("send", chat_id)
        mid = self._new_id()
        if not self.ctx.dry_run:
            self.messages[mid] = FakeMessage(chat_id, mid, caption, self.ctx.clock(), True, path=str(path),
                                             caption=caption)
            self.wire.append(Write("send_file", chat_id, mid, caption, path=str(path), at=self.ctx.clock()))
        return MessageRef(chat_id, mid)

    # -- views for tests -------------------------------------------------------------------------------
    def sent(self, chat_id: str | None = None) -> list[FakeMessage]:
        """Mac's own messages, in send order."""
        return [m for m in self.messages.values() if m.mine and (chat_id is None or m.chat_id == chat_id)]

    def texts(self) -> list[str]:
        """Every text Mac ever showed: sends, edits and captions."""
        return [w.text for w in self.wire if w.op in ("send_text", "edit", "send_file") and w.text]


def create(cfg: dict[str, Any], ctx: ChannelContext) -> FakeChannel:
    return FakeChannel(cfg, ctx)
