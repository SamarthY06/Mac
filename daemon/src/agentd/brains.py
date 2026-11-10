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
    never_chat_names: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def key(self) -> tuple[Any, ...]:
        """Changing any of these needs a new brain process."""
        return (str(self.cwd), tuple(self.setting_sources), self.model, self.chrome)


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
            "never_chat_names": list(spec.never_chat_names),
            # the transitional flock stays while local voice mode exists (W3.4); deleted with it
            "desktop_locks": [spec.desktop, *hooks.default_desktop_locks()]}
        if spec.profile is CHAT:
            kw.update(cb.CHAT_PROFILE)
            kw["env_exempt_dir"] = spec.cwd
        else:
            kw.update(desktop_wait_s=cb.VOICE_DESKTOP_WAIT_S, confirm_timeout_s=spec.profile.confirm_window_s)
        kw.update(spec.extra)
        return cb.ClaudeBrain(c, spec.gate, **kw)

    return make


async def close_quietly(brain: Brain, timeout: float = 10.0) -> None:
    try:
        await asyncio.wait_for(brain.close(), timeout)
    except Exception:
        log.warning("brain close failed", exc_info=True)
