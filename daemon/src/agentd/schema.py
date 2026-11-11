"""The agentd API, version 1.1, as one schema module (docs/agentd-api.md is the source text).

Every request op, response payload, event and error code of the contract is a dataclass or an
enum here, and `tests/agentd/test_schema.py` checks this module against the markdown file. Change
both together, and bump API_VERSION for any change to the wire format.

Wire rules applied by the codec:
- one JSON object per line, UTF-8, at most MAX_LINE_BYTES (a longer line or bad JSON is `bad_frame`);
- unknown fields are ignored when decoding (requests, responses and events alike);
- optional fields whose value is None are left out when encoding;
- ints and bools are kept apart (`true` is never a valid `turn`).

The module has no behaviour beyond encoding, decoding and validation.
"""

from __future__ import annotations

import enum
import json
import types
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from functools import cache
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

API_VERSION = "1.2"
MAX_LINE_BYTES = 1 << 20  # 1 MiB per line
REPLAY_BUFFER_EVENTS = 2_000  # per conversation
CLIENT_QUEUE_EVENTS = 10_000  # per connection, then slow_client


# ----------------------------------------------------------------------------- enums


class ClientKind(str, enum.Enum):
    VOICE = "voice"
    CHAT = "chat"
    APP = "app"
    TEST = "test"  # fake agentd only; the real agentd rejects it


class ErrorCode(str, enum.Enum):
    """Response error codes (api §7)."""

    BAD_FRAME = "bad_frame"
    HELLO_REQUIRED = "hello_required"
    UNAUTHORIZED = "unauthorized"
    VERSION_MISMATCH = "version_mismatch"
    FORBIDDEN = "forbidden"
    FORBIDDEN_CONVERSATION = "forbidden_conversation"
    UNKNOWN_OP = "unknown_op"
    BAD_ARGS = "bad_args"
    BUSY = "busy"
    STALE_QUESTION = "stale_question"
    PAUSED = "paused"
    PROFILE_NOT_FOUND = "profile_not_found"
    PROFILE_EXISTS = "profile_exists"
    PROFILE_CONFLICT = "profile_conflict"
    NEVER_LISTED = "never_listed"
    WORKSPACE_NOT_APPROVED = "workspace_not_approved"
    PROFILES_UNAVAILABLE = "profiles_unavailable"
    GAP = "gap"
    SLOW_CLIENT = "slow_client"
    INTERNAL = "internal"


CLOSING_CODES = frozenset({ErrorCode.BAD_FRAME, ErrorCode.HELLO_REQUIRED, ErrorCode.UNAUTHORIZED,
                           ErrorCode.VERSION_MISMATCH, ErrorCode.SLOW_CLIENT})


class EventErrorKind(str, enum.Enum):
    """`kind` of the `error` event (api §6). The brain's BrainErrorKind plus `paused`."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    CLI_MISSING = "cli_missing"
    TIMEOUT = "timeout"
    INTERNAL = "internal"
    PAUSED = "paused"


class QuestionKind(str, enum.Enum):
    """The single question-kind enum. The `question` event never uses CONFIRM (an outward
    confirmation is its own `confirm` event); `status.question.kind` may."""

    CONFIRM = "confirm"
    FOLDER_APPROVAL = "folder_approval"
    ROUTE = "route"
    CATCH_UP = "catch_up"
    RESTART = "restart"
    PROFILE = "profile"  # 1.1: add group / allow; a quoted yes or a 👍 only, 10 min


class ReplyOutcome(str, enum.Enum):
    APPROVED = "approved"
    DENIED = "denied"
    CHANGED = "changed"
    ANSWERED = "answered"


class CloseOutcome(str, enum.Enum):
    APPROVED = "approved"
    DENIED = "denied"
    CHANGED = "changed"
    ANSWERED = "answered"
    EXPIRED = "expired"
    SKIPPED = "skipped"  # replaced by a newer question


class SchemaError(ValueError):
    """A frame or message that does not match the contract. `code` is the ErrorCode to answer
    with (bad_frame for framing, unknown_op, bad_args for everything else)."""

    def __init__(self, message: str, code: ErrorCode = ErrorCode.BAD_ARGS) -> None:
        super().__init__(message)
        self.code = code


# ----------------------------------------------------------------------------- shared shapes


@dataclass(kw_only=True)
class Position:
    boot: str
    seq: int


@dataclass(kw_only=True)
class AskAttachment:
    path: str
    kind: Literal["image", "document"]
    name: str


@dataclass(kw_only=True)
class AskMeta:
    replayed: bool = False
    sent_at: float = 0.0
    msg_ref: str = ""


# ask.context (1.2, W10): untrusted chat content, validated and framed by agentd, never trimmed
CONTEXT_MAX_MESSAGES = 50
CONTEXT_MAX_CHARS = 8_000  # all message texts together
CONTEXT_MAX_TEXT = 1_000  # one message text
CONTEXT_MAX_MEMBERS = 256
CONTEXT_MAX_LINKS = 10
FRAMED_REFS_PER_SESSION = 200  # V1: msg_refs already framed into one session


@dataclass(kw_only=True)
class ContextMedia:
    kind: str
    name: str | None = None
    path: str | None = None  # quoted only: realpath inside the chat media dir


@dataclass(kw_only=True)
class ContextMessage:
    msg_ref: str
    sender: str
    owner: bool
    mac: bool
    at: float
    text: str | None = None
    media: ContextMedia | None = None
    earlier: bool | None = None  # on-demand history (W10.3)


@dataclass(kw_only=True)
class QuotedMessage:
    sender: str
    owner: bool
    mac: bool
    at: float
    text: str | None = None
    links: list[str] = field(default_factory=list)
    media: ContextMedia | None = None


@dataclass(kw_only=True)
class OtherChatQuote:
    kind: Literal["other_chat"]  # a quote from another chat: no content (V3)


@dataclass(kw_only=True)
class AskContext:
    members: list[str] | None = None
    messages: list[ContextMessage] | None = None
    quoted: QuotedMessage | OtherChatQuote | None = None


@dataclass(kw_only=True)
class InterruptedTurn:
    conv: str
    uid: str
    text: str


@dataclass(kw_only=True)
class AttachRef:
    attach_id: str
    path: str
    caption: str = ""


@dataclass(kw_only=True)
class UnackedTurn:
    turn_uid: str
    answer: str
    attaches: list[AttachRef] = field(default_factory=list)


@dataclass(kw_only=True)
class RunningStatus:
    uid: str
    text: str
    minutes: int
    folder: str | None = None


@dataclass(kw_only=True)
class QuestionStatus:
    question_id: str
    kind: QuestionKind
    text: str
    expires_at: float | None = None
    question: str | None = None  # confirm only: the model's own question (ConfirmEvent.question)
    target: str | None = None  # confirm only: re-posted prompts can show "To: <target>"
    detail: str | None = None  # confirm only (set_persona): the full new text, below the prompt


@dataclass(kw_only=True)
class DesktopStatus:
    holder: str
    what: str = ""


@dataclass(kw_only=True)
class ConversationStatus:
    conv: str
    alive: bool
    folder: str | None = None
    session_title: str = ""
    paused: str = ""  # "" | "loop" | "members_changed" | "user"
    running: RunningStatus | None = None
    question: QuestionStatus | None = None
    desktop: DesktopStatus | None = None
    unacked: list[UnackedTurn] = field(default_factory=list)
    loop_lag_ms_p99: float = 0.0


@dataclass(kw_only=True)
class Workspace:
    path: str
    name: str
    approved: bool
    settings_changed: bool


CAPABILITIES = ("files_read", "files_write", "chrome", "web", "desktop", "send_file", "outward_sends",
                "read_chat")
# the one plain-words table (UX phase 2 review): every user-facing string names a capability by these
CAPABILITY_WORDS = {"files_read": "reading files", "files_write": "writing files", "chrome": "Chrome",
                    "web": "the web", "desktop": "the desktop", "send_file": "sending files",
                    "outward_sends": "sending messages for you", "read_chat": "reading group messages"}
PERSONA_MAX_CHARS = 2000  # W9.1; agentd validates, prompts.py renders with the same limits
PERSONA_MAX_LINES = 40


@dataclass(kw_only=True)
class Capabilities:
    """Tool classes a chat may use (W9.1). A missing key is false (fails closed)."""

    files_read: bool = False
    files_write: bool = False
    chrome: bool = False
    web: bool = False
    desktop: bool = False
    send_file: bool = False
    outward_sends: bool = False
    read_chat: bool = False  # v3.5 (W10.6): no tool; the group's recent messages and members as context


@dataclass(kw_only=True)
class Profile:
    """One chat's settings (api §4.6, ARCHITECTURE W9.1)."""

    channel: str
    chat_id: str
    display_name: str
    enabled: bool
    persona: str
    trigger: Literal["all", "prefix"]
    prefix: str
    shared: bool
    capabilities: Capabilities
    confirm_files: bool
    rev: int
    created_at: float
    updated_at: float
    default_workspace: str | None = None  # None: no folder (left out on the wire)


# ----------------------------------------------------------------------------- ops (api §4)


@dataclass(kw_only=True)
class Op:
    OP: ClassVar[str] = ""


@dataclass(kw_only=True)
class Hello(Op):
    OP: ClassVar[str] = "hello"
    api: str
    token: str
    client: str
    client_epoch: str
    conversations: list[str] = field(default_factory=list)


# 4.1 turns
@dataclass(kw_only=True)
class Ask(Op):
    OP: ClassVar[str] = "ask"
    conv: str
    turn: int
    text: str
    attachments: list[AskAttachment] | None = None
    meta: AskMeta | None = None
    context: AskContext | None = None  # chat only (1.2)


@dataclass(kw_only=True)
class Steer(Op):
    """Voice: `turn` (required). Chat: `turn_uid`, or neither for the running turn."""

    OP: ClassVar[str] = "steer"
    conv: str
    text: str
    turn: int | None = None
    turn_uid: str | None = None


@dataclass(kw_only=True)
class Interrupt(Op):
    """Voice: `turn` (required). Chat: `turn_uid`, or neither for the running turn."""

    OP: ClassVar[str] = "interrupt"
    conv: str
    turn: int | None = None
    turn_uid: str | None = None


@dataclass(kw_only=True)
class NewSession(Op):
    OP: ClassVar[str] = "new_session"
    conv: str
    folder: str | None = None


@dataclass(kw_only=True)
class Reset(Op):
    OP: ClassVar[str] = "reset"
    conv: str
    hard: bool = False  # stop everything: the running turn, queued asks and background agents; fresh session next


@dataclass(kw_only=True)
class Warm(Op):
    OP: ClassVar[str] = "warm"
    conv: str


@dataclass(kw_only=True)
class Sleep(Op):
    OP: ClassVar[str] = "sleep"
    conv: str


@dataclass(kw_only=True)
class Handoff(Op):
    OP: ClassVar[str] = "handoff"
    conv: str


@dataclass(kw_only=True)
class Cancel(Op):
    OP: ClassVar[str] = "cancel"
    conv: str


@dataclass(kw_only=True)
class StopOther(Op):
    OP: ClassVar[str] = "stop_other"


# 4.2 questions and confirmations
@dataclass(kw_only=True)
class Confirm(Op):
    OP: ClassVar[str] = "confirm"
    conv: str
    question_id: str
    approved: bool


@dataclass(kw_only=True)
class RevokeAll(Op):
    OP: ClassVar[str] = "revoke_all"
    conv: str


@dataclass(kw_only=True)
class QuestionPosted(Op):
    OP: ClassVar[str] = "question_posted"
    conv: str
    question_id: str
    msg_ref: str


@dataclass(kw_only=True)
class Reply(Op):
    OP: ClassVar[str] = "reply"
    conv: str
    text: str | None = None
    quoted_ref: str | None = None
    reaction: str | None = None
    reaction_on: str | None = None


# 4.3 context and settings
@dataclass(kw_only=True)
class SetContext(Op):
    OP: ClassVar[str] = "set_context"
    conv: str
    locked: bool


@dataclass(kw_only=True)
class SetLanguage(Op):
    OP: ClassVar[str] = "set_language"
    conv: str
    lang: str


@dataclass(kw_only=True)
class SetName(Op):
    OP: ClassVar[str] = "set_name"
    conv: str
    name: str


@dataclass(kw_only=True)
class VoiceSettings(Op):
    OP: ClassVar[str] = "voice_settings"
    model: str
    chrome: bool
    setting_sources: list[str]
    name: str
    language: str


@dataclass(kw_only=True)
class ConnectCheck(Op):
    OP: ClassVar[str] = "connect_check"
    conv: str
    deep: bool | None = None


# 4.4 streams and status
@dataclass(kw_only=True)
class Subscribe(Op):
    OP: ClassVar[str] = "subscribe"
    conv: str
    after: Position | None = None


@dataclass(kw_only=True)
class Ack(Op):
    OP: ClassVar[str] = "ack"
    conv: str
    turn_uid: str


@dataclass(kw_only=True)
class Status(Op):
    OP: ClassVar[str] = "status"
    conv: str | None = None


@dataclass(kw_only=True)
class Pause(Op):
    OP: ClassVar[str] = "pause"
    conv: str
    value: bool
    reason: Literal["loop", "user", "members_changed", "never_listed"] | None = None


# 4.5 workspaces
@dataclass(kw_only=True)
class WsList(Op):
    OP: ClassVar[str] = "ws_list"


@dataclass(kw_only=True)
class WsApprove(Op):
    OP: ClassVar[str] = "ws_approve"
    path: str


@dataclass(kw_only=True)
class WsRemove(Op):
    OP: ClassVar[str] = "ws_remove"
    path: str


# 4.6 profiles (1.1)
@dataclass(kw_only=True)
class ProfileList(Op):
    OP: ClassVar[str] = "profile_list"


@dataclass(kw_only=True)
class ProfileCreate(Op):
    """`fields`: any other profile field except `rev` and the times, validated by agentd."""

    OP: ClassVar[str] = "profile_create"
    channel: str
    chat_id: str
    display_name: str
    owner_only: bool
    enabled: bool | None = None
    persona: str | None = None
    trigger: Literal["all", "prefix"] | None = None
    prefix: str | None = None
    shared: bool | None = None
    default_workspace: str | None = None
    capabilities: dict[str, Any] | None = None
    confirm_files: bool | None = None


@dataclass(kw_only=True)
class ProfileSet(Op):
    OP: ClassVar[str] = "profile_set"
    channel: str
    chat_id: str
    rev: int
    patch: dict[str, Any]


@dataclass(kw_only=True)
class ProfileDelete(Op):
    OP: ClassVar[str] = "profile_delete"
    channel: str
    chat_id: str


@dataclass(kw_only=True)
class ProfileCommand(Op):
    """An owner command chatd parsed (W9.3). `add`: subject, chat_id, owner_only; `allow` and
    `disallow`: capability."""

    OP: ClassVar[str] = "profile_command"
    conv: str
    command: Literal["add", "allow", "disallow"]
    subject: str | None = None
    chat_id: str | None = None
    owner_only: bool | None = None
    capability: str | None = None


OPS: dict[str, type[Op]] = {c.OP: c for c in (
    Hello, Ask, Steer, Interrupt, NewSession, Reset, Warm, Sleep, Handoff, Cancel, StopOther,
    Confirm, RevokeAll, QuestionPosted, Reply, SetContext, SetLanguage, SetName, VoiceSettings,
    ConnectCheck, Subscribe, Ack, Status, Pause, WsList, WsApprove, WsRemove,
    ProfileList, ProfileCreate, ProfileSet, ProfileDelete, ProfileCommand)}


# ----------------------------------------------------------------------------- response data


@dataclass(kw_only=True)
class Empty:
    pass


@dataclass(kw_only=True)
class HelloData:
    api: str
    server: str
    kind: ClientKind
    boot: str
    conversations: list[str] = field(default_factory=list)
    interrupted: list[InterruptedTurn] = field(default_factory=list)


@dataclass(kw_only=True)
class AskData:
    turn_uid: str


@dataclass(kw_only=True)
class AcceptedData:  # steer
    accepted: bool


@dataclass(kw_only=True)
class StoppedData:  # interrupt, cancel
    stopped: bool


@dataclass(kw_only=True)
class StopOtherData:
    stopped: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class ConfirmData:
    applied: bool


@dataclass(kw_only=True)
class ReplyData:
    bound: bool
    outcome: ReplyOutcome | None = None


@dataclass(kw_only=True)
class ConnectCheckData:
    accepted: bool
    check_id: str


@dataclass(kw_only=True)
class SubscribeData:
    boot: str
    seq: int


@dataclass(kw_only=True)
class StatusAll:
    """`status` without `conv` (app): every conversation. With `conv` the data is one
    ConversationStatus, as in the api §4.4 example."""

    conversations: list[ConversationStatus] = field(default_factory=list)


@dataclass(kw_only=True)
class PauseData:
    paused: bool | str


@dataclass(kw_only=True)
class WsListData:
    workspaces: list[Workspace]


@dataclass(kw_only=True)
class WsApproveData:
    settings_hash: str


@dataclass(kw_only=True)
class ProfileListData:
    rev: int
    profiles: list[Profile]


@dataclass(kw_only=True)
class ProfileData:  # profile_create, profile_set
    profile: Profile


@dataclass(kw_only=True)
class ProfileCommandData:
    question_id: str | None = None  # add, allow: the profile question that follows
    applied: bool | None = None  # disallow


@dataclass(kw_only=True)
class GapData:
    """`data` of a `gap` error response."""

    boot: str
    oldest_seq: int


RESPONSE_DATA: dict[str, Any] = {
    "hello": HelloData, "ask": AskData, "steer": AcceptedData, "interrupt": StoppedData,
    "new_session": Empty, "reset": Empty, "warm": Empty, "sleep": Empty, "handoff": AcceptedData,
    "cancel": StoppedData, "stop_other": StopOtherData, "confirm": ConfirmData, "revoke_all": Empty,
    "question_posted": Empty, "reply": ReplyData, "set_context": Empty, "set_language": Empty,
    "set_name": Empty, "voice_settings": Empty, "connect_check": ConnectCheckData,
    "subscribe": SubscribeData, "ack": Empty, "status": ConversationStatus | StatusAll,
    "pause": PauseData, "ws_list": WsListData, "ws_approve": WsApproveData, "ws_remove": Empty,
    "profile_list": ProfileListData, "profile_create": ProfileData, "profile_set": ProfileData,
    "profile_delete": Empty, "profile_command": ProfileCommandData,
}


@dataclass(kw_only=True)
class OkResponse:
    id: int
    data: Any = None  # a response-data dataclass, a list (ws_list) or a raw dict


@dataclass(kw_only=True)
class ErrorResponse:
    id: int | None  # None for frame-level errors (bad_frame, slow_client) where no id is known
    code: ErrorCode
    error: str
    data: dict[str, Any] | None = None  # e.g. GapData for `gap`


# ----------------------------------------------------------------------------- events (api §6)


@dataclass(kw_only=True)
class Event:
    EV: ClassVar[str] = ""
    conv: str
    boot: str
    seq: int


@dataclass(kw_only=True)
class TurnEvent(Event):
    turn: int
    turn_uid: str


@dataclass(kw_only=True)
class TextEvent(TurnEvent):
    EV: ClassVar[str] = "text"
    text: str
    tool_follows: bool | None = None


@dataclass(kw_only=True)
class SayEvent(TurnEvent):
    EV: ClassVar[str] = "say"
    text: str


@dataclass(kw_only=True)
class ToolStartEvent(TurnEvent):
    EV: ClassVar[str] = "tool_start"
    tool: str
    tool_use_id: str
    narration: str | None = None  # voice only; chat must not show it


@dataclass(kw_only=True)
class ToolEndEvent(TurnEvent):
    EV: ClassVar[str] = "tool_end"
    tool: str
    tool_use_id: str
    is_error: bool = False


@dataclass(kw_only=True)
class ConfirmEvent(TurnEvent):
    EV: ClassVar[str] = "confirm"
    question_id: str
    summary: str
    target: str
    question: str | None = None  # the model's own question ("Delete the draft to Bob?"), AG10/AG18
    detail: str | None = None  # set_persona only: the full new text; chatd shows it after a blank line


@dataclass(kw_only=True)
class QuestionEvent(Event):
    EV: ClassVar[str] = "question"
    question_id: str
    kind: QuestionKind
    text: str
    expires_at: float

    def __post_init__(self) -> None:
        if self.kind == QuestionKind.CONFIRM:
            raise SchemaError("question.kind cannot be confirm (outward confirmations are `confirm` events)")


@dataclass(kw_only=True)
class QuestionClosedEvent(Event):
    EV: ClassVar[str] = "question_closed"
    question_id: str
    outcome: CloseOutcome


@dataclass(kw_only=True)
class AttachEvent(TurnEvent):
    EV: ClassVar[str] = "attach"
    attach_id: str
    path: str
    caption: str = ""


@dataclass(kw_only=True)
class StatusEvent(Event):
    """Informational keys of `data`: chrome_connected, session_fallback, notify{title, body, path?},
    waiting_for ("desktop"), folder, session_title, alive; results of long ops: handoff_done{ok},
    connect_check{check_id, ...}."""

    EV: ClassVar[str] = "status"
    data: dict[str, Any] = field(default_factory=dict)
    turn: int | None = None
    turn_uid: str | None = None


@dataclass(kw_only=True)
class DoneData:
    duration_s: float
    interrupted: bool | None = None
    cost: float | None = None


@dataclass(kw_only=True)
class DoneEvent(TurnEvent):
    EV: ClassVar[str] = "done"
    data: DoneData


@dataclass(kw_only=True)
class ErrorEvent(Event):
    EV: ClassVar[str] = "error"
    kind: EventErrorKind
    message: str
    turn: int | None = None
    turn_uid: str | None = None


@dataclass(kw_only=True)
class InterruptedEvent(Event):
    EV: ClassVar[str] = "interrupted"
    uid: str
    text: str


EVENTS: dict[str, type[Event]] = {c.EV: c for c in (
    TextEvent, SayEvent, ToolStartEvent, ToolEndEvent, ConfirmEvent, QuestionEvent,
    QuestionClosedEvent, AttachEvent, StatusEvent, DoneEvent, ErrorEvent, InterruptedEvent)}


# connection-level notices: no conv, no seq, never replayed (api §4.6)
@dataclass(kw_only=True)
class ProfileChanged:
    EV: ClassVar[str] = "profile_changed"
    rev: int  # the store-wide profiles_rev
    channel: str
    chat_id: str
    profile: Profile | None = None
    deleted: bool | None = None


NOTICES: dict[str, type[Any]] = {ProfileChanged.EV: ProfileChanged}


# ----------------------------------------------------------------------------- policy (api §3)

_VOICE_OPS = frozenset({"ask", "steer", "interrupt", "confirm", "revoke_all", "reset", "warm", "sleep",
                        "handoff", "set_context", "set_language", "set_name", "voice_settings",
                        "connect_check", "status", "subscribe", "stop_other"})
_CHAT_OPS = frozenset({"ask", "steer", "interrupt", "reply", "question_posted", "new_session", "reset",
                       "set_context", "status", "subscribe", "ack", "pause", "ws_list", "profile_list",
                       "profile_command"})
_APP_OPS = frozenset({"status", "cancel", "pause", "ws_list", "ws_approve", "ws_remove", "profile_list",
                      "profile_create", "profile_set", "profile_delete"})

KIND_OPS: dict[ClientKind, frozenset[str]] = {
    ClientKind.VOICE: _VOICE_OPS | {"hello"},
    ClientKind.CHAT: _CHAT_OPS | {"hello"},
    ClientKind.APP: _APP_OPS | {"hello"},
    ClientKind.TEST: frozenset(OPS),  # the fake agentd grants what it likes; the real one rejects TEST
}

VOICE_CONV = "voice:local"
CHAT_CONFIG_PATH = "~/.mac-agent/chat/config.toml"  # fixed; never taken from a request


def check_op_for_kind(kind: ClientKind, op: Op) -> None:
    """Raise SchemaError(forbidden) if `kind` may not send `op`, and bad_args for the per-kind
    argument rules (voice steer/interrupt address a `turn`)."""
    if op.OP not in KIND_OPS[kind]:
        raise SchemaError(f"{kind.value} clients may not use {op.OP}", ErrorCode.FORBIDDEN)
    if kind == ClientKind.VOICE and isinstance(op, Steer | Interrupt) and op.turn is None:
        raise SchemaError(f"voice {op.OP} needs turn")


def turn_uid(boot: str, conv: str, client_epoch: str, turn: int) -> str:
    return f"{boot}:{conv}:{client_epoch}:{turn}"


# ----------------------------------------------------------------------------- codec


@cache
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _coerce(tp: Any, v: Any, path: str) -> Any:
    if tp is Any:
        return v
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = get_args(tp)
        if v is None:
            if type(None) in args:
                return None
            raise SchemaError(f"{path}: must not be null")
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, v, path)
            except SchemaError as e:
                errors.append(str(e))
        raise SchemaError("; ".join(errors) or f"{path}: bad value")
    if origin is Literal:
        if any(v == a and type(v) is type(a) for a in get_args(tp)):
            return v
        raise SchemaError(f"{path}: must be one of {list(get_args(tp))}")
    if origin is list:
        if not isinstance(v, list):
            raise SchemaError(f"{path}: must be a list")
        (et,) = get_args(tp) or (Any,)
        return [_coerce(et, x, f"{path}[{i}]") for i, x in enumerate(v)]
    if origin is dict or tp is dict:
        if not isinstance(v, dict):
            raise SchemaError(f"{path}: must be an object")
        return dict(v)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(v)
        except ValueError:
            raise SchemaError(f"{path}: must be one of {[m.value for m in tp]}") from None
    if is_dataclass(tp):
        if not isinstance(v, dict):
            raise SchemaError(f"{path}: must be an object")
        return from_dict(tp, v, path)
    if tp is bool:
        if isinstance(v, bool):
            return v
        raise SchemaError(f"{path}: must be true or false")
    if tp is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise SchemaError(f"{path}: must be an integer")
    if tp is float:
        if isinstance(v, int | float) and not isinstance(v, bool):
            return v
        raise SchemaError(f"{path}: must be a number")
    if tp is str:
        if isinstance(v, str):
            return v
        raise SchemaError(f"{path}: must be a string")
    raise SchemaError(f"{path}: unsupported schema type {tp!r}", ErrorCode.INTERNAL)


def from_dict(cls: type, d: dict[str, Any], path: str = "") -> Any:
    """Build dataclass `cls` from a JSON object. Unknown keys are ignored."""
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        where = f"{path}.{f.name}" if path else f.name
        if f.name in d:
            kwargs[f.name] = _coerce(hints[f.name], d[f.name], where)
        elif f.default is MISSING and f.default_factory is MISSING:
            raise SchemaError(f"{where}: missing")
    return cls(**kwargs)


def to_json(v: Any) -> Any:
    """Dataclass / enum / list tree -> plain JSON values. None-valued fields are left out."""
    if is_dataclass(v) and not isinstance(v, type):
        out = {}
        for f in fields(v):
            x = getattr(v, f.name)
            if x is not None:
                out[f.name] = to_json(x)
        return out
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, list | tuple):
        return [to_json(x) for x in v]
    if isinstance(v, dict):
        return {k: to_json(x) for k, x in v.items()}
    return v


def parse_line(line: bytes | str) -> dict[str, Any]:
    """One NDJSON line -> a JSON object, or SchemaError(bad_frame)."""
    raw = line.encode() if isinstance(line, str) else line
    if len(raw) > MAX_LINE_BYTES:
        raise SchemaError("line longer than 1 MiB", ErrorCode.BAD_FRAME)
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SchemaError(f"invalid JSON: {e}", ErrorCode.BAD_FRAME) from None
    if not isinstance(obj, dict):
        raise SchemaError("a frame must be a JSON object", ErrorCode.BAD_FRAME)
    return obj


def encode_line(obj: dict[str, Any]) -> bytes:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_LINE_BYTES:
        raise SchemaError("encoded line longer than 1 MiB", ErrorCode.INTERNAL)
    return raw


# requests
def encode_request(req_id: int, op: Op) -> dict[str, Any]:
    return {"id": req_id, "op": op.OP, **to_json(op)}


def decode_request(obj: dict[str, Any]) -> tuple[int, Op]:
    req_id = obj.get("id")
    if not isinstance(req_id, int) or isinstance(req_id, bool):
        raise SchemaError("id: must be an integer")
    name = obj.get("op")
    cls = OPS.get(name) if isinstance(name, str) else None
    if cls is None:
        raise SchemaError(f"unknown op {name!r}", ErrorCode.UNKNOWN_OP)
    return req_id, from_dict(cls, {k: v for k, v in obj.items() if k not in ("id", "op")})


# responses
def encode_ok(req_id: int, data: Any = None) -> dict[str, Any]:
    return {"id": req_id, "ok": True, "data": {} if data is None else to_json(data)}


def encode_error(req_id: int | None, code: ErrorCode, error: str, data: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"id": req_id, "ok": False, "code": ErrorCode(code).value, "error": error}
    if data is not None:
        out["data"] = to_json(data)
    return out


def decode_response(obj: dict[str, Any], op: str | None = None) -> OkResponse | ErrorResponse:
    """Decode a response. With `op`, `data` is typed per RESPONSE_DATA; without, it stays raw."""
    ok = obj.get("ok")
    if ok is True:
        req_id = _coerce(int, obj.get("id"), "id")
        data = obj.get("data", {})
        if op is not None:
            data = _coerce(RESPONSE_DATA[op], data, "data")
        return OkResponse(id=req_id, data=data)
    if ok is False:
        return from_dict(ErrorResponse, obj)
    raise SchemaError("ok: must be true or false")


# events
def encode_event(ev: Event) -> dict[str, Any]:
    return {"ev": ev.EV, **to_json(ev)}


def decode_event(obj: dict[str, Any]) -> Event:
    name = obj.get("ev")
    cls = EVENTS.get(name) if isinstance(name, str) else None
    if cls is None:
        raise SchemaError(f"unknown event {name!r}")
    return from_dict(cls, {k: v for k, v in obj.items() if k != "ev"})


def is_event(obj: dict[str, Any]) -> bool:
    return "ev" in obj


def encode_notice(n: Any) -> dict[str, Any]:
    return {"ev": n.EV, **to_json(n)}


def decode_notice(obj: dict[str, Any]) -> Any:
    name = obj.get("ev")
    cls = NOTICES.get(name) if isinstance(name, str) else None
    if cls is None:
        raise SchemaError(f"unknown notice {name!r}")
    return from_dict(cls, {k: v for k, v in obj.items() if k != "ev"})
