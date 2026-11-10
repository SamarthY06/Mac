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
class QuotedMedia:
    """An image or document in a quoted message (W10.5). `path` is set only after the bytes were
    read (25 MB cap) and resolved with realpath inside the media dir."""

    kind: str
    name: str = ""
    mime: str = ""
    size: int = 0
    path: Path | None = None


@dataclass(frozen=True)
class Quoted:
    """The message a trigger replies to (W10.5). `kind == "other_chat"`: a quote from another
    chat, with no text, links or media."""

    ref: MessageRef
    text: str
    from_owner: bool
    sender: str = ""  # jid
    sender_name: str = ""
    from_mac: bool = False
    at: float = 0.0
    links: tuple[str, ...] = ()
    media: QuotedMedia | None = None
    kind: str = ""


@dataclass(frozen=True)
class ContextMessage:
    """A message in a chat of the context set that is not a trigger (W10.1): another member's, or
    on-demand history (`history`). Data only: it never starts, steers or answers anything."""

    channel: str
    chat_id: str
    ref: MessageRef
    sender: str = ""  # jid
    sender_name: str = ""
    from_owner: bool = False
    at: float = 0.0
    text: str = ""
    media: dict[str, Any] | None = None  # {kind, name?}; never downloaded
    quoted_msg: str = ""
    replayed: bool = False
    history: bool = False


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
ContextCallback = Callable[[ContextMessage], Awaitable[None]]


@runtime_checkable
class ChannelAdapter(Protocol):
    name: str
    caps: Cap
    limits: Limits

    async def start(self, sink: MessageSink, on_status: StatusCallback,
                    on_chat_changed: ChatChangedCallback, *, on_context: ContextCallback | None = None) -> None: ...
    async def close(self) -> None: ...
    def status(self) -> ChannelStatus: ...
    async def mark_handled(self, ref: MessageRef) -> None: ...
    async def send_text(self, chat_id: str, text: str, reply_to: MessageRef | None = None,
                        mentions: tuple[str, ...] = ()) -> MessageRef: ...
    async def edit_text(self, ref: MessageRef, text: str, mentions: tuple[str, ...] = ()) -> bool: ...
    async def react(self, ref: MessageRef, emoji: str) -> None: ...
    async def set_typing(self, chat_id: str, on: bool) -> None: ...
    async def send_file(self, chat_id: str, path: Path, caption: str = "") -> MessageRef: ...
    async def link(self) -> None: ...
    async def unlink(self) -> None: ...
    async def list_chats(self) -> list[dict[str, Any]]: ...
    async def set_allowed(self, chat_ids: frozenset[str], context_ids: frozenset[str] = frozenset()) -> None:
        """The allowlist from agentd's enabled profiles (W9.8) and the context set, the subset with
        `read_chat` (W10.6). The adapter keeps the last pair and re-applies it after every restart
        of its transport. Under test mode: logged, not applied."""
    async def members(self, chat_id: str) -> list[dict[str, Any]]:
        """`[{jid, number, name}]` of an allowed chat (W10.4); raises DestinationRefused otherwise."""
    async def request_history(self, chat_id: str, before: MessageRef, count: int = 30) -> bool:
        """Ask for on-demand history (W10.3); results arrive as `history` context messages. False
        when the adapter can't (rate, dry run, not in the context set, unsupported)."""


class DestinationRefused(Exception):
    """Raised by an adapter (R2) before any I/O for a never-listed or non-allowlisted chat."""
