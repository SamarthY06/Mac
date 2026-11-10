"""The terminal channel behind `chatd chat` (W7.4): you type, Mac answers in the same terminal,
through the same ChatRunner, governor and agentd API as WhatsApp.

Every line you type is an owner message in one chat (`chat_id`, default "local"). Edits print
the new text on a line starting with "~", reactions print as "[👀]", files as "📎 <path>".
A typed line starting with "Mac ·" is treated as an echo and dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from chatd.channels.base import (
    Cap,
    ChannelStatus,
    ChatChangedCallback,
    ContextCallback,
    InboundMessage,
    Limits,
    LinkState,
    MessageRef,
    MessageSink,
    StatusCallback,
)
from chatd.channels.registry import ChannelContext
from chatd.format import PREFIX

log = logging.getLogger(__name__)


class CliChannel:
    name = "cli"
    caps = Cap.EDIT | Cap.REACT | Cap.FILES

    def __init__(self, cfg: dict[str, Any], ctx: ChannelContext) -> None:
        self.ctx = ctx
        self.chat_id: str = cfg.get("chat_id", "local")
        self.limits = Limits(edit_window_s=86400.0, min_edit_interval_s=cfg.get("min_edit_interval_s", 1.0),
                             text_format="plain")
        self._input: asyncio.StreamReader | None = cfg.get("input")
        self._out: TextIO = cfg.get("output") or sys.stdout
        self._ids = itertools.count(1)
        self._sent: dict[str, float] = {}  # our message id -> sent time
        self._status = ChannelStatus(LinkState.OFFLINE)
        self._on_status: StatusCallback | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self, sink: MessageSink, on_status: StatusCallback,
                    on_chat_changed: ChatChangedCallback, *, on_context: ContextCallback | None = None) -> None:
        self._on_status = on_status  # a terminal has no other members: no context
        if self._input is None:
            loop = asyncio.get_running_loop()
            self._input = asyncio.StreamReader()
            await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(self._input), sys.stdin)
        self._set(LinkState.CONNECTED, account=self.chat_id)
        self._task = asyncio.create_task(self._read(sink), name="cli:read")

    async def _read(self, sink: MessageSink) -> None:
        assert self._input is not None
        try:
            while line := await self._input.readline():
                text = line.decode("utf-8", "replace").strip()
                if not text or text.startswith(PREFIX.strip()) or not self.ctx.allowed(self.chat_id):
                    continue
                mid = f"in-{next(self._ids)}"
                await sink(InboundMessage(channel=self.name, chat_id=self.chat_id,
                                          ref=MessageRef(self.chat_id, mid), text=text, sent_at=time.time()))
        finally:
            self._set(LinkState.OFFLINE, detail="The terminal closed.")

    def _set(self, state: LinkState, **kw: Any) -> None:
        self._status = ChannelStatus(state, **kw)
        if self._on_status is not None:
            self._on_status(self._status)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    def status(self) -> ChannelStatus:
        return self._status

    async def mark_handled(self, ref: MessageRef) -> None:
        await self.ctx.seen.add(f"{ref.chat_id}:{ref.message_id}")

    def _print(self, line: str) -> None:
        if not self.ctx.dry_run:
            self._out.write(line + "\n")
            self._out.flush()

    async def send_text(self, chat_id: str, text: str, reply_to: MessageRef | None = None,
                        mentions: tuple[str, ...] = ()) -> MessageRef:
        self.ctx.check_destination(chat_id)
        await self.ctx.governor.charge("send", chat_id)
        mid = f"out-{next(self._ids)}"
        self._sent[mid] = self.ctx.clock()
        self._print(text)
        return MessageRef(chat_id, mid)

    async def edit_text(self, ref: MessageRef, text: str, mentions: tuple[str, ...] = ()) -> bool:
        self.ctx.check_destination(ref.chat_id)
        sent = self._sent.get(ref.message_id)
        if sent is None or self.ctx.clock() - sent > self.limits.edit_window_s:
            return False
        await self.ctx.governor.charge("edit", ref.chat_id)
        self._print("~ " + text.replace("\n", "\n  "))
        return True

    async def react(self, ref: MessageRef, emoji: str) -> None:
        self.ctx.check_destination(ref.chat_id)
        await self.ctx.governor.charge("react", ref.chat_id)
        self._print(f"[{emoji}]")

    async def set_typing(self, chat_id: str, on: bool) -> None:
        self.ctx.check_destination(chat_id)  # no typing in a terminal

    async def send_file(self, chat_id: str, path: Path, caption: str = "") -> MessageRef:
        self.ctx.check_destination(chat_id)
        await self.ctx.governor.charge("send", chat_id)
        mid = f"out-{next(self._ids)}"
        self._sent[mid] = self.ctx.clock()
        self._print(f"📎 {path}" + (f"  {caption}" if caption else ""))
        return MessageRef(chat_id, mid)

    async def link(self) -> None:
        self._set(LinkState.CONNECTED, account=self.chat_id)

    async def unlink(self) -> None:
        self._set(LinkState.UNLINKED)

    async def list_chats(self) -> list[dict[str, Any]]:
        return [{"id": self.chat_id, "subject": "Terminal", "owner_only": True}]

    async def set_allowed(self, chat_ids: frozenset[str], context_ids: frozenset[str] = frozenset()) -> None:
        """The terminal is one fixed local chat; nothing to change."""

    async def members(self, chat_id: str) -> list[dict[str, Any]]:
        return []  # one person: nobody to tag

    async def request_history(self, chat_id: str, before: MessageRef, count: int = 30) -> bool:
        return False


def create(cfg: dict[str, Any], ctx: ChannelContext) -> CliChannel:
    return CliChannel(cfg, ctx)
