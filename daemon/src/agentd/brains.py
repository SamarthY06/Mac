"""The brain seam: one Brain per conversation, built from a profile chosen by token kind (W3.1).

`BrainFactory(spec) -> Brain` is the only way agentd makes a brain. Tests use `fake_factory`
(FakeBrain); the service uses `claude_factory`, which maps a spec onto ClaudeBrain's profile
arguments (builder B) and refuses to start if any of them is missing (fail closed).
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from macd.contracts import Brain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    name: str  # "voice" | "chat"
    confirm_window_s: float
    desktop_label: str


VOICE = Profile("voice", confirm_window_s=10.0, desktop_label="voice")
CHAT = Profile("chat", confirm_window_s=600.0, desktop_label="chat")


@dataclass
class SessionInfo:
    """Structurally the same as macd.brain.claude_brain.SessionInfo."""

    session_id: str | None
    updated_at: float = 0.0
    started_at: float = 0.0
    fork: bool = False


@dataclass
class BrainSpec:
    conv: str
    profile: Profile
    cwd: Path
    setting_sources: list[str]
    gate: Any  # macd.brain.gate.ConfirmationGate
    session_store: Any  # load() / save(SessionInfo), both async
    memory_file: Any  # the single MemoryFile
    desktop: Any  # DesktopHandle (polled DesktopLock protocol)
    recent_user_text: Callable[[], list[str]]
    model: str = ""
    chrome: bool = True
    name: str = "Mac"
    language: str = "en-us"
    never_chat_names: tuple[str, ...] = ()  # chat profile only (AG7, V6: voice may name any chat)
    # chat profile (W9): the live capability set (None = voice), the send_file yes, the persona
    capabilities: Callable[[], frozenset[str]] | None = None
    confirm_send_file: Callable[[], bool] | None = None  # live, like capabilities (read per send)
    chat_persona: str = ""
    display_name: str = ""
    chat_shared: bool = False
    on_set_persona: Callable[[str], Any] | None = None
    on_alive: Callable[[bool], None] | None = None  # the brain process connected or went away
    on_rename: Callable[[str], str | None] | None = None  # voice: the model renamed the assistant
    on_reset_voice: Callable[[], str | None] | None = None  # voice: back to the standard voice
    on_go_to_sleep: Callable[[], str | None] | None = None  # voice: the model ends the conversation
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[Any, ...]:
        """Changing any of these needs a new brain process."""
        return (str(self.cwd), tuple(self.setting_sources), self.model, self.chrome,
                tuple(self.never_chat_names), self.chat_persona, self.display_name, self.chat_shared)


BrainFactory = Callable[[BrainSpec], Brain]


class RecordSessionStore:
    """`session_store` backed by the conversation record (M3): agentd's conversations.json."""

    def __init__(self, get: Callable[[], SessionInfo | None], put: Callable[[SessionInfo], Any]) -> None:
        self._get = get
        self._put = put

    async def load(self) -> SessionInfo | None:
        return self._get()

    async def save(self, info: Any) -> None:
        res = self._put(SessionInfo(info.session_id, info.updated_at, info.started_at,
                                    bool(getattr(info, "fork", False))))
        if inspect.isawaitable(res):
            await res


def fake_factory(make: Callable[[BrainSpec], Brain] | None = None) -> BrainFactory:
    from macd.brain.fake_brain import FakeBrain

    return make or (lambda spec: FakeBrain())


# ClaudeBrain keyword arguments agentd needs from builder B (W3.1 profile table).
CLAUDE_ARGS = ("cwd", "session_store", "memory_file", "recent_user_text", "steer_only",
               "rotate_after_idle_min", "handoff_enabled", "on_resume_fallback", "max_turn_s",
               "write_claude_md", "setting_sources", "env_exempt_dir", "whatsapp_ui_block",
               "never_chat_names", "channel_note", "send_file", "close_own_tabs", "desktop_locks",
               "desktop_wait_s", "desktop_busy_reason", "confirm_timeout_s")


CHAT_SAFETY_ARGS = ("capabilities", "confirm_send_file", "never_chat_names")


def claude_factory(cfg: Any, chat_cfg: Any | None = None) -> BrainFactory:
    """Real brains. `cfg` is macd's Config (paths, cli); `chat_cfg` the [chat_brain] BrainConfig."""
    from macd.brain import claude_brain as cb
    from macd.brain import hooks

    params = inspect.signature(cb.ClaudeBrain.__init__).parameters
    missing = [a for a in CLAUDE_ARGS if a not in params]
    if missing:
        raise RuntimeError(f"ClaudeBrain lacks the profile arguments agentd needs: {', '.join(missing)}")

    def make(spec: BrainSpec) -> Brain:
        base = chat_cfg if (spec.profile is CHAT and chat_cfg is not None) else cfg.brain
        brain_cfg = dataclasses.replace(base, model=spec.model or base.model, chrome=spec.chrome,
                                        setting_sources=list(spec.setting_sources))
        c = dataclasses.replace(cfg, brain=brain_cfg)
        if hasattr(c, "name"):
            c = dataclasses.replace(c, name=spec.name)
        kw: dict[str, Any] = {
            "cwd": spec.cwd, "session_store": spec.session_store, "memory_file": spec.memory_file,
            "recent_user_text": spec.recent_user_text, "setting_sources": list(spec.setting_sources),
            # the transitional flock stays while local voice mode exists (W3.4); deleted with it
            "desktop_locks": [spec.desktop, *hooks.default_desktop_locks()]}
        kw.update(spec.extra)  # N5: first, so nothing in it can replace a safety argument below
        if spec.profile is CHAT:
            # safety arguments fail CLOSED: a brain that cannot take them is never built for chat
            missing = [a for a in CHAT_SAFETY_ARGS if a not in params]
            if missing or spec.capabilities is None or spec.confirm_send_file is None or not spec.never_chat_names:
                raise RuntimeError(f"refusing to build a chat brain without {', '.join(missing) or 'its policy'}")
            kw.update(cb.CHAT_PROFILE)
            kw.update(env_exempt_dir=spec.cwd, never_chat_names=list(spec.never_chat_names),
                      capabilities=spec.capabilities, confirm_send_file=spec.confirm_send_file)
            optional = {"chat_persona": spec.chat_persona, "display_name": spec.display_name,
                        "chat_shared": spec.chat_shared, "on_set_persona": spec.on_set_persona,
                        "snapshot": False}  # W10.7: a persona edit must apply to a resumed session
            kw.update({k: v for k, v in optional.items() if v is not None and k in params})
        else:  # never the never-list: by voice any chat may be named, behind a spoken confirm (V6)
            kw.update(desktop_wait_s=cb.VOICE_DESKTOP_WAIT_S, confirm_timeout_s=spec.profile.confirm_window_s)
            optional = {"on_rename": spec.on_rename, "on_reset_voice": spec.on_reset_voice,
                        "on_go_to_sleep": spec.on_go_to_sleep}
            kw.update({k: v for k, v in optional.items() if v is not None and k in params})
        if spec.on_alive is not None and "on_alive" in params:  # push liveness when B's brain offers it
            kw["on_alive"] = spec.on_alive
        return cb.ClaudeBrain(c, spec.gate, **kw)

    return make


async def close_quietly(brain: Brain, timeout: float = 10.0) -> None:
    try:
        await asyncio.wait_for(brain.close(), timeout)
    except Exception:
        log.warning("brain close failed", exc_info=True)
