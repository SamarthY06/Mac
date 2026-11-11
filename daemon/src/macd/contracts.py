"""Shared contracts between macd subsystems.

This file is the single source of truth for the interfaces between the speech stack,
the audio backends, the brain and the orchestrator. Change it only deliberately: every
subsystem codes against it. See ARCHITECTURE.md for the rationale.

Conventions
- Internal audio: 16 kHz mono float32 numpy arrays in [-1, 1]. FRAME = 20 ms = 320 samples.
- TTS output may be any sample rate (Kokoro = 24 kHz); outputs resample as needed.
- Everything async runs on ONE asyncio loop. Blocking work goes to executors.
- turn_id is a monotonically increasing int; 0 means "no turn".
"""

from __future__ import annotations

import enum
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20 ms @ 16 kHz


# ----------------------------------------------------------------------------- audio


@runtime_checkable
class AudioInput(Protocol):
    """A source of 16 kHz mono float32 frames (each exactly FRAME_SAMPLES long)."""

    name: str  # "local", "remote:<identity>", "file"
    requires_wake_word: bool  # False for remote/LiveKit sources

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    def frames(self) -> AsyncIterator[np.ndarray]:
        """Async iterator of frames. Ends only when the input is closed."""
        ...


@runtime_checkable
class AudioOutput(Protocol):
    name: str

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Queue `pcm` (float32 mono) and return when it has FINISHED playing.

        If the awaiting task is cancelled, playback of this chunk stops immediately and
        CancelledError propagates. Back-to-back awaits must be gapless (< 10 ms).
        """
        ...

    def flush(self) -> None:
        """Stop all playback immediately and drop anything queued. Sync, non-blocking."""
        ...

    @property
    def aec(self) -> bool:
        """True if the input paired with this output has acoustic echo cancellation."""
        ...


# ----------------------------------------------------------------------------- speech


@dataclass(frozen=True)
class WakeHit:
    phrase: str  # normalized, e.g. "hey mac"
    sample_index: int  # absolute sample index (in the source's stream) where the hit ended
    t: float = field(default_factory=time.monotonic)


class WakeDetector(Protocol):
    def accept(self, frame: np.ndarray, sample_index: int) -> WakeHit | None:
        """Feed one frame. Returns a hit (already stage-2 verified or not; see Listener)."""
        ...

    def reset(self) -> None: ...


class VAD(Protocol):
    def accept(self, frame: np.ndarray) -> float:
        """Feed one frame; returns speech probability for this frame in [0, 1]."""
        ...

    def reset(self) -> None: ...


class STT(Protocol):
    async def transcribe(self, pcm: np.ndarray) -> str:
        """Transcribe 16 kHz float32 audio. Runs in the STT executor. Returns plain text."""
        ...


class TTS(Protocol):
    sample_rate: int

    async def synthesize(self, text: str) -> np.ndarray:
        """Return float32 mono PCM at self.sample_rate. Runs in the TTS executor."""
        ...


@dataclass(frozen=True)
class Utterance:
    text: str  # wake phrase already stripped
    source: str  # AudioInput.name
    woke: bool  # True if this utterance started with a (verified) wake word
    started_at: float  # monotonic
    ended_at: float
    audio_s: float
    window: WindowReason | None = None  # which open window (followup/confirm/ptt) captured it, if any
    same_breath: bool = False  # speech began right after the wake phrase (no pause) -> skip the wake earcon
    continuation: bool = False  # continues a capture that hit max_utterance_s -> treat as steer, no ack


# ----------------------------------------------------------------------------- brain


class BrainEventKind(str, enum.Enum):
    TEXT = "text"  # assistant text delta / block (speakable)
    TOOL_START = "tool_start"  # a tool call began; data["narration"] = spoken status or None
    TOOL_END = "tool_end"  # a tool call finished
    SAY = "say"  # model called mcp__mac__say(text): speak now
    CONFIRM = "confirm"  # model called request_confirmation(summary): await human yes/no
    STATUS = "status"  # informational (e.g. chrome connected / session id)
    ATTACH = "attach"  # mac.send_file: data = {"attach_id", "path", "caption"} (chat profile only)
    DONE = "done"  # turn finished (ResultMessage)
    ERROR = "error"  # classified error, see BrainError


class BrainErrorKind(str, enum.Enum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    CLI_MISSING = "cli_missing"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


@dataclass
class BrainEvent:
    kind: BrainEventKind
    turn_id: int
    text: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""
    is_error: bool = False
    error_kind: BrainErrorKind | None = None
    data: dict[str, Any] = field(default_factory=dict)  # cost, duration, session_id, ...


class Brain(Protocol):
    """Owner-task model: implementations run ONE long-lived task that owns the client."""

    alive: bool  # True while the underlying Claude process is connected (see warm/sleep)

    async def start(self) -> None: ...
    async def close(self) -> None: ...

    def events(self) -> AsyncIterator[BrainEvent]:
        """Single consumer stream of events for all turns (tagged with turn_id)."""
        ...

    async def ask(self, turn_id: int, text: str) -> None:
        """Begin a new turn. If a previous turn is still draining, waits for it first."""
        ...

    async def steer(self, turn_id: int, text: str) -> None:
        """Add a user message to the running turn without interrupting it."""
        ...

    async def interrupt(self, turn_id: int) -> None:
        """Interrupt turn `turn_id` (no-op if it already finished). Returns immediately;
        the stale tail is drained internally and never emitted as that turn's TEXT."""
        ...

    async def reset(self, hard: bool = False) -> None:
        """Start a fresh conversation (new session). ``hard``: stop everything now, background
        agents included, without a session handoff: the underlying process is killed."""
        ...

    async def warm(self) -> None:
        """Start the underlying Claude process now (called on wake word) so the first
        ask() is fast. Idempotent; never raises (failures surface on ask())."""
        ...

    async def sleep(self) -> None:
        """Shut the underlying Claude process down (idle shutdown), persisting the session
        so the next ask() resumes the conversation. Idempotent."""
        ...

    async def handoff(self) -> bool:
        """Before a session is retired (rotation/reset/long idle): write the durable facts and
        open tasks of this conversation into memory.md "Recent context". Bounded (~60 s);
        returns False (no-op) if the session is young or nothing happened. Never raises."""
        ...

    async def set_name(self, name: str) -> None:
        """Change the assistant's name for the next session (keeps context)."""
        ...

    async def connect_check(self) -> dict[str, Any]:
        """Verify the user's Claude + Chrome setup (bounded ~25 s; may start the CLI). Returns
        {"auth": ok|not_signed_in|api_key|cli_missing|error, "auth_detail", "chrome":
        connected|not_connected|unavailable_api_key|unknown, "chrome_detail", "cli_version",
        "session_ok"}. Answers from the last known result while a turn is running."""
        ...

    async def set_context(self, locked: bool) -> None:
        """Tell the brain about the Mac's state (screen locked → tools that need the UI won't work)."""
        ...

    async def set_language(self, lang: str) -> None:
        """Change the language Mac answers in (Kokoro lang code, e.g. "hi", "fr-fr")."""
        ...


# ----------------------------------------------------------------------------- state


class State(str, enum.Enum):
    OFF = "off"  # listening switched off by the user: no wake word, mic released
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CONFIRMING = "confirming"
    FOLLOWUP = "followup"
    ERROR = "error"


GateVerdict = Literal["allow", "outward", "sensitive_read", "catastrophic"]


# ----------------------------------------------------------------------------- listener


class ListenerEventKind(str, enum.Enum):
    WAKE = "wake"  # verified wake word (earcon should play)
    SPEECH_START = "speech_start"  # VAD detected user speech in an open capture
    UTTERANCE = "utterance"  # final transcript of one user utterance
    NO_SPEECH = "no_speech"  # woke / window opened but nothing was said before timeout
    FALSE_WAKE = "false_wake"  # KWS hit rejected by stage-2 verification (for metrics)
    BARGE_IN = "barge_in"  # sustained user speech while Mac talks (AEC on): pause speech now;
    # the verdict follows as UTTERANCE(window=SPEAKING) or NO_SPEECH(window=SPEAKING)


@dataclass(frozen=True)
class ListenerEvent:
    kind: ListenerEventKind
    source: str
    utterance: Utterance | None = None
    wake: WakeHit | None = None
    window: WindowReason | None = None  # for NO_SPEECH: which window expired
    t: float = field(default_factory=time.monotonic)


class WindowReason(str, enum.Enum):
    FOLLOWUP = "followup"
    CONFIRM = "confirm"
    PTT = "ptt"  # push-to-talk: capture next utterance immediately
    SPEAKING = "speaking"  # capture while Mac is talking: any words with AEC (barge-in), stop words only without
    TURN = "turn"  # the conversation partner's mic during a voice turn: no wake word needed, stays open


class Listener(Protocol):
    """One Listener per AudioInput. Owns KWS -> verify -> VAD -> STT for that source.

    Capture semantics:
    - Always runs KWS (unless muted). A verified hit emits WAKE, then captures the
      utterance (including speech in the same breath after the wake phrase) -> UTTERANCE
      with woke=True and the wake phrase stripped.
    - open_window(): for `seconds`, any VAD-detected speech is captured WITHOUT a wake word
      -> UTTERANCE with woke=False. If nothing is said -> NO_SPEECH when it expires.
    - While `speaking` is True and the output has no AEC, window/VAD capture is suppressed
      (wake word still works, with self-trigger suppression); only bare stop words come through.
    - While `speaking` is True with AEC, speech is captured as WindowReason.SPEAKING: BARGE_IN
      after `listen.barge_in_onset_ms` of speech, then the transcript (or NO_SPEECH if empty).
    - A TURN window stays open across utterances until closed or replaced.
    """

    source_name: str
    speaking: bool

    async def run(self) -> None:
        """Long-running task; consumes the AudioInput until closed."""
        ...

    def events(self) -> AsyncIterator[ListenerEvent]: ...
    def open_window(self, seconds: float, reason: WindowReason) -> None: ...
    def close_window(self) -> None: ...
    def set_muted(self, muted: bool) -> None: ...
    def set_speaking(self, speaking: bool, current_text: str = "") -> None: ...
