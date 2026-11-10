"""The conversation record (W2.4), persisted in `~/.mac-agent/agent/state/conversations.json`.

Only persisted kinds of question live here; an outward confirmation is memory only (fail closed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

UNACKED_MAX = 20  # finished chat turns kept until `ack`; the oldest go first
OWN_SESSIONS_MAX = 50


@dataclass
class RunningTurn:
    uid: str  # f"{boot}:{conv}:{client_epoch}:{turn}"
    text: str
    started_at: float  # wall time
    msg_ref: str = ""
    folder: str | None = None


@dataclass
class OpenQuestion:
    question_id: str
    kind: str  # folder_approval | route | catch_up | restart
    text: str
    expires_at: float  # wall time
    prompt_ref: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationRecord:
    conv: str
    folder: str | None = None  # absolute path; None = general
    session_id: str | None = None
    session_fork: bool = False  # the session is not Mac's own: resume it as a fork
    session_started_at: float = 0.0
    session_updated_at: float = 0.0
    own_sessions: list[str] = field(default_factory=list)  # sessions Mac created (resumed, not forked)
    session_title: str = ""
    paused: str = ""  # "" | "loop" | "members_changed" | "user"
    running: RunningTurn | None = None
    question: OpenQuestion | None = None
    restart_notice_for: str = ""
    last_answer_at: float = 0.0
    unacked: list[dict[str, Any]] = field(default_factory=list)
    fresh_next: bool = False  # new_session: the next ask starts a new session
    fresh_folder: str | None = None  # new_session{folder}: a folder name or path for the next ask
    voice_migrated: bool = False  # M3: the one-time copy of state.json["brain"] was made

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConversationRecord:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if isinstance(kw.get("running"), dict):
            kw["running"] = _build(RunningTurn, kw["running"])
        if isinstance(kw.get("question"), dict):
            kw["question"] = _build(OpenQuestion, kw["question"])
        return cls(**kw)

    def remember_own(self, session_id: str) -> None:
        if session_id in self.own_sessions:
            self.own_sessions.remove(session_id)
        self.own_sessions.append(session_id)
        del self.own_sessions[:-OWN_SESSIONS_MAX]

    def add_unacked(self, item: dict[str, Any]) -> None:
        self.unacked.append(item)
        del self.unacked[:-UNACKED_MAX]


def _build(cls: type, d: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})
