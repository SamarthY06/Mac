"""The chat channel adapter contract (docs/whatsapp/ARCHITECTURE.md W5.1).

A channel (WhatsApp first; later Telegram, Slack, a CLI) is one module under chatd/channels/
with `create(cfg, ctx) -> ChannelAdapter`. chatd's ChatRunner talks only to this Protocol, and
every adapter must pass the adapter contract suite (W7.1, cases C1-C9).

Rules every adapter keeps:
- R1. Every outbound call goes through the governor.
- R2. Never-listed or non-allowlisted destinations raise before any I/O.
- R3. Only owner messages on trusted devices are delivered, never our own echoes (device id,
      sent id, "Mac · " prefix).
- R4. `mark_handled` is called only after agentd has accepted the request.
- R5. Prompts are plain text. Binding of replies to questions happens in agentd.

Contract only: no behaviour lives here.
"""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


class Cap(enum.Flag):
    NONE = 0
    EDIT = enum.auto()
    REACT = enum.auto()
    TYPING = enum.auto()
    FILES = enum.auto()


@dataclass(frozen=True)
class Limits:
    max_text: int = 4000
    edit_window_s: float = 0.0  # 0 = edits not supported
    min_edit_interval_s: float = 4.0
    max_file_mb: int = 100
    text_format: Literal["plain", "whatsapp", "markdown", "slack"] = "plain"


@dataclass(frozen=True)
class MessageRef:
    chat_id: str
    message_id: str


@dataclass(frozen=True)
class Attachment:
    """Images and documents only; audio, voice, video and stickers are never downloaded."""

    kind: Literal["image", "document"]
    path: Path
    mime: str
    name: str = ""
    size: int = 0


@dataclass(frozen=True)
class Quoted:
    ref: MessageRef
    text: str
    from_owner: bool


@dataclass(frozen=True)
class InboundMessage:
    """An owner message (R3). `reaction` is set for a reaction event (then `reply_to` names the
    message reacted to); `unsupported` flags media that was not downloaded."""

    channel: str
    chat_id: str
    ref: MessageRef
    text: str = ""
    attachments: tuple[Attachment, ...] = ()
    reply_to: Quoted | None = None
    reaction: str | None = None
    unsupported: Literal["", "voice", "audio", "video", "sticker"] = ""
    sent_at: float = 0.0
    replayed: bool = False


class LinkState(str, enum.Enum):
    UNLINKED = "unlinked"
    PAIRING = "pairing"
    CONNECTED = "connected"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass(frozen=True)
class ChannelStatus:
    """What `status()` reports and `on_status` receives. W5.1 names the type without fields;
    this is the minimal shape (flagged to COORD in SPIKES.md)."""

    state: LinkState
    detail: str = ""  # plain words for the app, e.g. "Scan the QR code on your phone."
    qr: str = ""  # current pairing code while PAIRING
    account: str = ""  # the linked own id while CONNECTED


@dataclass(frozen=True)
class ChatChanged:
    """A chat's membership or subject changed (owner-only -> shared pauses the chat, W5.2)."""

    chat_id: str
    subject: str = ""
    owner_only: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


MessageSink = Callable[[InboundMessage], Awaitable[None]]
StatusCallback = Callable[[ChannelStatus], None]
ChatChangedCallback = Callable[[ChatChanged], Awaitable[None]]


@runtime_checkable
class ChannelAdapter(Protocol):
    name: str
    caps: Cap
    limits: Limits

    async def start(self, sink: MessageSink, on_status: StatusCallback,
                    on_chat_changed: ChatChangedCallback) -> None: ...
    async def close(self) -> None: ...
    def status(self) -> ChannelStatus: ...
    async def mark_handled(self, ref: MessageRef) -> None: ...
    async def send_text(self, chat_id: str, text: str, reply_to: MessageRef | None = None) -> MessageRef: ...
    async def edit_text(self, ref: MessageRef, text: str) -> bool: ...
    async def react(self, ref: MessageRef, emoji: str) -> None: ...
    async def set_typing(self, chat_id: str, on: bool) -> None: ...
    async def send_file(self, chat_id: str, path: Path, caption: str = "") -> MessageRef: ...
    async def link(self) -> None: ...
    async def unlink(self) -> None: ...
    async def list_chats(self) -> list[dict[str, Any]]: ...
    async def set_allowed(self, chat_ids: frozenset[str]) -> None:
        """The allowlist from agentd's enabled profiles (W9.8). The adapter keeps the last set and
        re-applies it after every restart of its transport. Under test mode: logged, not applied."""


class DestinationRefused(Exception):
    """Raised by an adapter (R2) before any I/O for a never-listed or non-allowlisted chat."""
