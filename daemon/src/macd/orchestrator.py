"""Orchestrator: the turn state machine and input policy (ARCHITECTURE §4).

One asyncio loop, one owner per resource. The orchestrator owns *policy*; it never touches
audio directly (everything goes through SpeechQueue) and never blocks on the brain (calls
that may wait run in the Turn's task set).

Input policy:
| state               | utterance                          | action                              |
|---------------------|------------------------------------|-------------------------------------|
| IDLE / LISTENING    | woke (or PTT window / remote)      | new turn                            |
| FOLLOWUP            | any from the turn's source         | new turn (same conversation)        |
| THINKING / SPEAKING | woke: stop words                   | interrupt                           |
|                     | woke: "actually…", "no, <change>"  | interrupt + new ask                 |
|                     | woke: anything else / continuation | steer (+ "Got it." unless cont.)    |
|                     | SPEAKING window (bare stop word)   | interrupt                           |
| CONFIRMING          | affirmative                        | gate.resolve(True)                  |
|                     | bare negative / timeout            | gate.resolve(False)                 |
|                     | longer reply ("yes but …")         | gate.resolve(False) + steer         |
| OFF                 | —                                  | listeners muted; nothing arrives    |
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import itertools
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from macd import config
from macd import phrases as P
from macd.brain.errors import MESSAGES as ERROR_MESSAGES
from macd.brain.narration import describe_tool as _describe_tool
from macd.config import Config
from macd.contracts import (
    Brain,
    BrainErrorKind,
    BrainEvent,
    BrainEventKind,
    Listener,
    ListenerEvent,
    ListenerEventKind,
    State,
    Utterance,
    WindowReason,
)
from macd.eventlog import EventLog, Metrics, NullEventLog
from macd.speech.segmenter import SentenceSegmenter
from macd.speech_queue import SYSTEM_TURN, SpeechQueue
from macd.tasks import TaskSet, supervise

log = logging.getLogger(__name__)

PHONE_SESSION_DETAIL = "Phone session: local mic paused"  # shown by the app instead of "Listening…"
THINKING_AFTER_IDLE_S = 1.0  # SPEAKING → THINKING only after this much silence (no HUD flicker)
WAKE_BLIP_DELAY_S = 0.25  # the wake blip plays only if the user paused after "Hey Mac"
LONG_TURN_DONE_EARCON_S = 30.0
SPEECH_STALL_S = 30.0
HANDOFF_TIMEOUT_S = 60.0  # brain.handoff() (session notes) before reset / OFF  # an output that hasn't finished a turn's speech by then is treated as broken
CONFIRM_SPEECH_SLACK_S = 10.0  # confirm timeout armed at CONFIRM arrival, before the prompt is spoken
DescribeTool = Callable[[str, dict[str, Any]], str | None]


class Gate(Protocol):
    def resolve(self, approved: bool) -> Any: ...
    def revoke_all(self) -> Any: ...


class Segmenter(Protocol):
    def feed(self, delta: str) -> list[str]: ...
    def flush(self) -> list[str]: ...


# ----------------------------------------------------------------------------- turn


HOLD_S = 1.5  # answer text is held this long before it is spoken (step lines are dropped)
HOLD_MAX_SENTENCES = 2


@dataclass(eq=False)
class Turn:
    id: int
    text: str
    source: str  # listener source name, or "control"
    output: str  # SpeechQueue output name
    origin: str  # "voice" | "control" | "remote"
    segmenter: Segmenter
    tasks: TaskSet
    started: float = field(default_factory=time.monotonic)
    transcript_t: float = field(default_factory=time.monotonic)
    asked_t: float | None = None
    first_text_t: float | None = None
    last_event: float = field(default_factory=time.monotonic)  # last brain event (watchdog)
    first_audio: bool = False
    brain_done: bool = False
    finishing: bool = False
    errored: bool = False
    confirming: bool = False
    current_tool: str | None = None
    current_narration: str | None = None
    last_speech: float = field(default_factory=time.monotonic)
    last_heartbeat: float = 0.0
    heartbeats: int = 0
    narrated: dict[str, float] = field(default_factory=dict)  # line → when it was last spoken
    held: list[str] = field(default_factory=list)  # answer text held until we know it isn't a step line
    held_t: float | None = None
    releasing: bool = False  # this text block is the answer: stream it
    assistant: list[str] = field(default_factory=list)

    @property
    def remote(self) -> bool:
        return self.origin == "remote"


# ----------------------------------------------------------------------------- orchestrator


Subscriber = Callable[[dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        cfg: Config,
        *,
        brain: Brain,
        gate: Gate,
        speech: SpeechQueue,
        eventlog: EventLog | NullEventLog | None = None,
        metrics: Metrics | None = None,
        earcon: Callable[[str], tuple[Any, int]] | None = None,
        segmenter_factory: Callable[[], Segmenter] = SentenceSegmenter,
        describe_tool: DescribeTool = _describe_tool,
        voice_reset: Callable[[], Awaitable[Any]] | None = None,
        rename: Callable[[str], Awaitable[Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.brain = brain
        self.gate = gate
        self.speech = speech
        self.eventlog = eventlog or NullEventLog()
        self.metrics = metrics or Metrics()
        self.earcon = earcon
        self.segmenter_factory = segmenter_factory
        self.describe_tool = describe_tool
        self.voice_reset = voice_reset  # "Hey Mac, use your default voice" (wired in __main__)
        self.rename = rename  # "Hey Mac, call yourself Jarvis" → settings (wired in __main__)

        self.state = State.IDLE
        self.turn: Turn | None = None
        self.last_turn: Turn | None = None
        self._turn_ids = itertools.count(1)
        self.muted = False
        self.asleep = False
        self.off = False
        self.locked = False  # screen locked (app event); listening continues, Claude is told
        self.voice_ok = True
        self.voice_reason = ""
        self.remote: dict[str, Any] = {"active": False, "present": False, "verified": False, "pin": None,
                                       "identity": None}

        self._listeners: dict[str, Listener] = {}
        self._requires_wake: dict[str, bool] = {}
        self._primary: str | None = None
        self._listener_tasks: dict[str, list[asyncio.Task[Any]]] = {}
        self._timers: dict[str, asyncio.Task[Any]] = {}
        self._tasks = TaskSet("orch")
        self._subscribers: list[Subscriber] = []
        self._filler_rr = itertools.cycle(P.FILLER_ACTION)
        self._followup_deadline = 0.0
        self._recent_user: collections.deque[str] = collections.deque(maxlen=50)
        self._handoff_task: asyncio.Task[Any] | None = None
        self._tg: asyncio.TaskGroup | None = None
        self._stop = asyncio.Event()
        self.started = time.monotonic()

        speech.on_start = self._on_speech_start
        speech.on_idle = self._on_speech_idle

    # ================================================================== wiring

    def add_listener(self, listener: Listener, *, requires_wake: bool = True, primary: bool = False) -> None:
        name = listener.source_name
        if name in self._listeners:
            self.remove_listener(name)
        self._listeners[name] = listener
        self._requires_wake[name] = requires_wake
        if primary or (self._primary is None and requires_wake):
            self._primary = name
        listener.set_muted(self._effective_mute())
        if self._tg is not None:
            self._start_listener_tasks(name)

    def remove_listener(self, name: str) -> None:
        self._listeners.pop(name, None)
        self._requires_wake.pop(name, None)
        for t in self._listener_tasks.pop(name, []):
            t.cancel()
        if self._primary == name:
            self._primary = next((n for n, w in self._requires_wake.items() if w), None)

    def _start_listener_tasks(self, name: str) -> None:
        assert self._tg is not None
        lst = self._listeners[name]
        self._listener_tasks[name] = [
            self._tg.create_task(
                supervise(f"listener:{name}", lst.run, on_error=self._report_crash), name=f"listener:{name}"
            ),
            self._tg.create_task(
                supervise(f"listener-events:{name}", lambda: self._consume_listener(lst), on_error=self._report_crash),
                name=f"listener-events:{name}",
            ),
        ]

    def subscribe(self, cb: Subscriber) -> Callable[[], None]:
        self._subscribers.append(cb)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(cb)

        return unsubscribe

    def publish(self, event: dict[str, Any]) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                log.exception("subscriber failed")

    def _report_crash(self, name: str, exc: BaseException) -> None:
        self.metrics.incr("subsystem_crashes")
        self.eventlog.emit("subsystem_crash", subsystem=name, error=repr(exc))
        self.publish({"event": "error", "kind": "internal", "source": name,
                      "message": f"{name} crashed: {exc!r}; restarting"})

    def set_voice(self, ok: bool, reason: str = "") -> None:
        """Whether speech models loaded; surfaced on every state event (UX#9)."""
        self.voice_ok, self.voice_reason = ok, reason

    # ================================================================== lifecycle

    async def run(self) -> None:
        """Run all orchestrator-owned subsystem tasks until `close()`."""
        own: list[asyncio.Task[Any]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                self._tg = tg
                own.append(tg.create_task(
                    supervise("speech", self.speech.run, on_error=self._report_crash, restart_on_exit=True),
                    name="speech"))
                own.append(tg.create_task(
                    supervise("brain-events", self._consume_brain, on_error=self._report_crash,
                              restart_on_exit=True), name="brain-events"))
                for name in list(self._listeners):
                    self._start_listener_tasks(name)
                self.eventlog.emit("orchestrator_start", listeners=sorted(self._listeners))
                self._set_state(State.OFF if self.off else State.IDLE, "start")
                await self._stop.wait()
                self._shutdown_tasks(own)
        finally:
            self._tg = None
            self._shutdown_tasks(own)

    def _shutdown_tasks(self, own: list[asyncio.Task[Any]]) -> None:
        if self.turn is not None:
            self.turn.tasks.cancel()
        for t in list(self._timers.values()):
            t.cancel()
        self._timers.clear()
        self._tasks.cancel()
        for tasks in self._listener_tasks.values():
            for t in tasks:
                t.cancel()
        for t in own:
            t.cancel()

    def close(self) -> None:
        self._stop.set()

    # ================================================================== state & events

    def _set_state(self, state: State, reason: str = "") -> None:
        if state == self.state and reason != "start":
            return
        prev = self.state
        self.state = state
        tid = self.turn.id if self.turn else None
        self.eventlog.emit("state", turn_id=tid, state=state.value, prev=prev.value, reason=reason)
        self.publish(self._state_event(prev=prev.value, reason=reason))

    def _state_event(self, *, prev: str, reason: str) -> dict[str, Any]:
        detail = self.voice_reason or (PHONE_SESSION_DETAIL if self.remote["present"] else "")
        return {"event": "state", "state": self.state.value, "prev": prev, "turn_id": self.turn.id if self.turn else None,
                "reason": reason, "muted": self.muted, "locked": self.locked, "voice": self.voice_ok,
                "voice_reason": detail}

    def _activity(self, kind: str, turn_id: int | None = None, **data: Any) -> None:
        """Log + publish one activity record (the HUD's "what's happening" line)."""
        self.eventlog.emit("activity", turn_id=turn_id, kind=kind, **data)
        self.publish({"event": "activity", "kind": kind, "turn_id": turn_id, "text": "", **data})

    def _transcript(self, role: str, text: str, *, turn_id: int | None, source: str, kind: str = "",
                    woke: bool = False, remote: bool = False) -> None:
        self.publish({"event": "transcript", "role": role, "text": text, "kind": kind, "turn_id": turn_id,
                      "source": source, "woke": woke, "remote": remote, "final": True, "delta": False})

    def _timer(self, name: str, delay: float, cb: Callable[[], Any]) -> None:
        self._cancel_timer(name)

        async def fire() -> None:
            await asyncio.sleep(delay)
            self._timers.pop(name, None)
            res = cb()
            if asyncio.iscoroutine(res):
                await res

        self._timers[name] = asyncio.create_task(fire(), name=f"timer:{name}")

    def _cancel_timer(self, name: str) -> None:
        t = self._timers.pop(name, None)
        if t is not None and t is not asyncio.current_task():
            t.cancel()

    # ================================================================== speech helpers

    def _output_for_source(self, source: str) -> str:
        if source.startswith("remote") and self.speech.has_output("remote"):
            return "remote"
        return self.speech.default_output

    def _play_earcon(self, name: str, output: str, *, priority: bool = False,
                     on_play: Callable[[], None] | None = None) -> None:
        if self.earcon is None:
            return
        try:
            pcm, sr = self.earcon(name)
        except KeyError:
            return
        self.speech.play_pcm(SYSTEM_TURN, pcm, sr, "earcon", output=output, priority=priority, on_play=on_play)

    def _speak(self, turn: Turn, text: str, kind: str = "answer", *, priority: bool = False) -> None:
        def on_play(t: Turn = turn) -> None:
            t.last_speech = time.monotonic()
            if not t.first_audio:
                t.first_audio = True
                self.metrics.observe_since("transcript_to_first_audio", t.transcript_t)

        text = P.dedash(text)
        if self.speech.say(turn.id, text, kind, output=turn.output, priority=priority, on_play=on_play):  # type: ignore[arg-type]
            turn.last_speech = time.monotonic()
            if kind in ("answer", "say", "confirm"):
                self._transcript("assistant", text, turn_id=turn.id, source=turn.source, kind=kind,
                                 remote=turn.remote)

    def _say_system(self, text: str, output: str | None = None, *, priority: bool = False) -> None:
        out = output or (self.turn.output if self.turn else self.speech.default_output)
        self.speech.say(SYSTEM_TURN, P.dedash(text), "say", output=out, priority=priority)

    def _on_speech_start(self, text: str, kind: str) -> None:
        for lst in self._listeners.values():
            with contextlib.suppress(Exception):
                lst.set_speaking(True, text)
        self._cancel_timer("thinking")
        if self.turn is not None and self.state == State.THINKING and kind in ("answer", "say", "confirm"):
            self._set_state(State.SPEAKING, "speech")

    def _on_speech_idle(self) -> None:
        for lst in self._listeners.values():
            with contextlib.suppress(Exception):
                lst.set_speaking(False, "")
        if self.turn is not None and self.state == State.SPEAKING and not self.turn.brain_done:
            self._timer("thinking", THINKING_AFTER_IDLE_S, self._back_to_thinking)

    def _back_to_thinking(self) -> None:
        turn = self.turn
        if turn is not None and self.state == State.SPEAKING and not turn.brain_done and not self.speech.speaking \
                and self.speech.pending(turn.id) == 0:
            self._set_state(State.THINKING, "speech idle")

    def _filler(self, text: str) -> str:
        if not self.brain.alive:
            return P.SLOW_FILLER
        if P.is_question(text):
            return P.FILLER_SHORT_QUESTION if len(text.split()) <= 5 else P.FILLER_QUESTION
        return next(self._filler_rr)

    # ================================================================== windows

    def _listener_for(self, source: str) -> Listener | None:
        lst = self._listeners.get(source)
        if lst is not None:
            return lst
        if self.remote["present"]:
            remote = next((lst for n, lst in self._listeners.items() if n.startswith("remote")), None)
            if remote is not None:
                return remote
        return self._listeners.get(self._primary) if self._primary else None

    def _open_window(self, source: str, seconds: float, reason: WindowReason) -> str | None:
        lst = self._listener_for(source)
        if lst is None:
            return None
        lst.open_window(seconds, reason)
        return lst.source_name

    def _close_windows(self) -> None:
        for lst in self._listeners.values():
            with contextlib.suppress(Exception):
                lst.close_window()

    # ================================================================== listener input

    async def _consume_listener(self, lst: Listener) -> None:
        async for ev in lst.events():
            try:
                await self.on_listener_event(ev)
            except Exception:
                log.exception("listener event handling failed: %s", ev.kind)

    async def on_listener_event(self, ev: ListenerEvent) -> None:
        kind = ev.kind
        if kind == ListenerEventKind.WAKE:
            self._on_wake(ev)
        elif kind == ListenerEventKind.FALSE_WAKE:
            self.metrics.incr("false_wakes")
            self.eventlog.emit("false_wake", source=ev.source)
        elif kind == ListenerEventKind.SPEECH_START:
            self._on_speech_detected(ev)
        elif kind == ListenerEventKind.NO_SPEECH:
            self._on_no_speech(ev)
        elif kind == ListenerEventKind.UTTERANCE and ev.utterance is not None:
            await self._on_utterance(ev, ev.utterance)

    def _local_suppressed(self, source: str) -> bool:
        return self.remote["present"] and not source.startswith("remote")

    def _on_wake(self, ev: ListenerEvent) -> None:
        self.metrics.incr("wakes")
        self.eventlog.emit("wake", source=ev.source, phrase=ev.wake.phrase if ev.wake else "")
        self._activity("wake", self.turn.id if self.turn else None, source=ev.source)
        if self._local_suppressed(ev.source):
            return
        if self.cfg.brain.warm_on_wake:
            self._tasks.spawn(self.brain.warm(), "brain.warm")
        wake_t = ev.wake.t if ev.wake else ev.t
        output = self._output_for_source(ev.source)
        if self.turn is None or self.state in (State.IDLE, State.FOLLOWUP, State.LISTENING):
            self._cancel_timer("followup")
            # Blip only if the user paused after "Hey Mac": a SPEECH_START/UTTERANCE in the same
            # breath cancels it (UX#15).
            self._timer("wake-blip", WAKE_BLIP_DELAY_S, lambda: self._play_earcon(
                "wake", output, on_play=lambda: self.metrics.observe_since("wake_to_earcon", wake_t)))
            if self.turn is None:
                self._set_state(State.LISTENING, "wake")
        elif self.speech.speaking:
            self._play_earcon("wake", output, priority=True)  # "I heard you" while talking (UX#2)

    def _on_speech_detected(self, ev: ListenerEvent) -> None:
        self._cancel_timer("wake-blip")
        if self._local_suppressed(ev.source):
            return
        if self.turn is None and self.state in (State.IDLE, State.FOLLOWUP) and ev.window is not None:
            self._cancel_timer("followup")
            self._set_state(State.LISTENING, "speech")

    def _on_no_speech(self, ev: ListenerEvent) -> None:
        if self.state == State.CONFIRMING and self.turn is not None and self.turn.confirming:
            self._resolve_confirm(self.turn, "timeout", "no_speech")
            return
        if self.turn is None and self.state in (State.LISTENING, State.FOLLOWUP):
            self._set_state(State.IDLE, f"no speech ({ev.window.value if ev.window else 'wake'})")

    async def _on_utterance(self, ev: ListenerEvent, utt: Utterance) -> None:
        self._cancel_timer("wake-blip")
        text = utt.text.strip()
        source = utt.source or ev.source
        window = utt.window
        if utt.audio_s > 0:
            self.metrics.observe("endpoint_to_transcript", max(0.0, (ev.t - utt.ended_at) * 1000.0))
        is_text = source == "control"
        origin = "control" if is_text else "voice"
        remote = source.startswith("remote") or (not is_text and not self._requires_wake.get(source, True))
        continuation = utt.continuation
        woke = utt.woke or window == WindowReason.PTT or remote or continuation
        self.eventlog.emit("utterance", turn_id=self.turn.id if self.turn else None, source=source, text=text,
                           woke=utt.woke, window=window.value if window else None, state=self.state.value)
        self._transcript("user", text, turn_id=self.turn.id if self.turn else None, source=source, woke=utt.woke,
                         remote=source.startswith("remote"))
        if not text:
            if self.turn is None and self.state == State.LISTENING:
                self._set_state(State.IDLE, "empty utterance")
            return
        self._recent_user.append(text)
        if self._local_suppressed(source):
            self._ignore(source, text, "remote session active")
            return
        if remote and source.startswith("remote") and not self.remote["verified"]:
            self._check_pin(source, text)
            return

        turn = self.turn
        # ---------------------------------------------------------------- bare stop word while speaking
        # DECISION 11: only stop words count here; corrections/steers still need the wake word
        # (a bystander's "actually, let's get lunch" must not redirect the turn).
        if window == WindowReason.SPEAKING:
            if not P.is_stop(text):
                self._ignore(source, text, "speaking: not a stop word")
            elif turn is not None:
                self.metrics.incr("interrupts")
                await self._cancel_turn(turn, "stop while speaking", speak=True)
            else:
                self.speech.flush_all()
            return

        # ---------------------------------------------------------------- CONFIRMING
        if turn is not None and self.state == State.CONFIRMING and turn.confirming:
            if utt.woke or window == WindowReason.CONFIRM or source == self._confirm_source(turn):
                self._resolve_confirm(turn, P.classify_confirmation(text), "utterance", text=text,
                                      confirm_source="text" if is_text else "voice")
            else:
                self._ignore(source, text, "confirming: other source")
            return

        # ---------------------------------------------------------------- remote preemption
        if remote and turn is not None and not turn.remote:
            await self._cancel_turn(turn, "remote preempt", speak=False)
            await self._start_turn(text, source, origin="remote", transcript_t=ev.t)
            return
        if remote:
            origin = "remote"

        # ---------------------------------------------------------------- mid-turn
        if turn is not None and not turn.brain_done:
            if not woke:
                self.metrics.incr("ignored_utterances")
                self._ignore(source, text, "mid-turn without wake word")
            elif continuation:
                self._steer(turn, text, ack=False)
            elif P.is_stop(text):
                self.metrics.incr("interrupts")
                await self._cancel_turn(turn, "stop words", speak=True)
            elif P.is_off_request(text):
                await self._cancel_turn(turn, "off request", speak=False)
                self._go_off_spoken(turn.output)
            elif (corr := P.correction_text(text)) is not None:
                await self._redirect(turn, text, corr, source, origin, ev.t)
            else:
                self._steer(turn, text)
            return

        # ---------------------------------------------------------------- brain done, speech still playing
        if turn is not None:
            if not woke and window != WindowReason.FOLLOWUP:
                self._ignore(source, text, "speaking tail without wake word")
                return
            await self._cancel_turn(turn, "stop words" if P.is_stop(text) else "new utterance",
                                    speak=P.is_stop(text))
            if P.is_stop(text):
                return
            await self._start_turn(text, source, origin=origin, transcript_t=ev.t)
            return

        # ---------------------------------------------------------------- no active turn
        followup_ok = window == WindowReason.FOLLOWUP or (
            self.state == State.FOLLOWUP and self.last_turn is not None and self.last_turn.source == source
        )
        if not (woke or followup_ok):
            self.metrics.incr("ignored_utterances")
            self._ignore(source, text, "no wake word")
            if self.state == State.LISTENING:
                self._set_state(State.IDLE, "ignored utterance")
            return
        if not woke and not P.is_followup_intent(text):
            # A bystander fragment ("Then") must not become a Claude turn; keep listening.
            self.metrics.incr("followup_ignored")
            self.eventlog.emit("followup_ignored", source=source, text=text)
            remaining = self._followup_deadline - time.monotonic()
            if remaining > 0.05:
                self._open_window(source, remaining, WindowReason.FOLLOWUP)
            return
        self._cancel_timer("followup")
        if P.is_stop(text):
            self.speech.flush_all()
            self._set_state(State.IDLE, "stop words")
        elif P.is_off_request(text):
            self._go_off_spoken(self._output_for_source(source))
        elif P.is_voice_reset_request(text) and self.voice_reset is not None:
            self._activity("voice_reset", None, text="Back to the default voice")
            self._tasks.spawn(self.voice_reset(), "voice-reset")
        elif (new_name := P.rename_request(text)) and self.rename is not None:
            self._activity("rename", None, text=f"Renaming to {new_name}")
            self._tasks.spawn(self.rename(new_name), "rename")
        else:
            await self._start_turn(text, source, origin=origin, transcript_t=ev.t)

    def _ignore(self, source: str, text: str, why: str) -> None:
        log.debug("ignored utterance from %s (%s): %r", source, why, text)
        self.eventlog.emit("utterance_ignored", source=source, reason=why, text=text)

    async def _redirect(self, turn: Turn, text: str, corr: str, source: str, origin: str, t: float) -> None:
        self.metrics.incr("corrections")
        self._activity("correction", turn.id, text=corr)
        await self._cancel_turn(turn, "correction", speak=False)
        await self._start_turn(text, source, origin=origin, transcript_t=t)

    # ================================================================== turns

    async def _start_turn(self, text: str, source: str, *, origin: str, transcript_t: float | None = None) -> Turn:
        if self.turn is not None:
            await self._cancel_turn(self.turn, "superseded", speak=False)
        self._cancel_timer("followup")
        self._close_windows()
        tid = next(self._turn_ids)
        turn = Turn(id=tid, text=text, source=source, output=self._output_for_source(source), origin=origin,
                    segmenter=self.segmenter_factory(), tasks=TaskSet(f"turn{tid}"),
                    transcript_t=transcript_t or time.monotonic())
        self.turn = turn
        self.speech.set_current_turn(tid)
        self.metrics.incr("turns")
        self.eventlog.emit("turn_start", turn_id=tid, text=text, source=source, origin=origin, output=turn.output)
        self._activity("turn_start", tid, text=text, source=source, origin=origin)
        self._set_state(State.THINKING, "new turn")
        self._speak(turn, self._filler(text), "filler")
        turn.tasks.spawn(self._ask(turn), "ask")
        turn.tasks.spawn(self._narration_loop(turn), "narration")
        turn.tasks.spawn(self._slow_start(turn), "slow-start")
        turn.tasks.spawn(self._watchdog(turn), "watchdog")
        return turn

    async def _ask(self, turn: Turn) -> None:
        try:
            await self.brain.ask(turn.id, turn.text)
            turn.asked_t = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("brain.ask failed")
            if self.turn is turn:
                self._on_error(turn, BrainEvent(BrainEventKind.ERROR, turn.id, error_kind=BrainErrorKind.INTERNAL,
                                                data={"exc": repr(e)}))

    def _steer(self, turn: Turn, text: str, *, ack: bool = True) -> None:
        self.metrics.incr("steers")
        self._activity("steer", turn.id, text=text)
        if ack:
            self._speak(turn, P.STEER_ACK, "filler", priority=True)
        turn.tasks.spawn(self.brain.steer(turn.id, text), "steer")

    async def _cancel_turn(self, turn: Turn, reason: str, *, speak: bool) -> None:
        if self.turn is not turn:
            return
        self.turn = None
        self.last_turn = turn
        turn.tasks.cancel()
        self.speech.cancel_turn(turn.id)
        self._cancel_timer("confirm")
        self._cancel_timer("thinking")
        if turn.confirming:
            turn.confirming = False
            with contextlib.suppress(Exception):
                self.gate.resolve(False)
        with contextlib.suppress(Exception):
            self.gate.revoke_all()
        self.eventlog.emit("turn_cancel", turn_id=turn.id, reason=reason)
        self._activity("turn_end", turn.id, reason=reason, cancelled=True, text=" ".join(turn.assistant))
        if speak:
            self._say_system(P.STOPPED, turn.output, priority=True)
        self._set_state(State.OFF if self.off else State.IDLE, reason)
        if not turn.brain_done:
            try:
                await asyncio.wait_for(self.brain.interrupt(turn.id), timeout=2.0)
            except (TimeoutError, Exception):
                log.warning("brain.interrupt(%d) failed or timed out", turn.id, exc_info=True)

    async def _finish_turn(self, turn: Turn) -> None:
        """After DONE/ERROR: wait for the turn's speech, then FOLLOWUP → IDLE."""
        if not await self.speech.drain(turn.id, timeout=SPEECH_STALL_S):
            self.speech.cancel_turn(turn.id)
            if self.turn is turn and not turn.errored:
                turn.errored = True
                self.metrics.incr("output_stalls")
                self.eventlog.emit("output_stall", turn_id=turn.id, output=turn.output)
                self.publish({"event": "error", "turn_id": turn.id, "kind": "timeout",
                              "message": f"audio output {turn.output} stalled; speech dropped"})
        if self.turn is not turn:
            return
        self.turn = None
        self.last_turn = turn
        self._cancel_timer("thinking")
        self.metrics.observe_since("turn_total", turn.started)
        full = " ".join(turn.assistant).strip()
        elapsed = time.monotonic() - turn.started
        self.eventlog.emit("turn_end", turn_id=turn.id, text=full, errored=turn.errored, duration_s=round(elapsed, 3))
        self._activity("turn_end", turn.id, text=full, errored=turn.errored, cancelled=False)
        turn.tasks.cancel()
        if not turn.errored and elapsed > LONG_TURN_DONE_EARCON_S:
            self._play_earcon("done", turn.output)
        if self.off or turn.errored or (turn.origin == "control" and not self._listeners):
            self._set_state(State.OFF if self.off else State.IDLE, "turn end")
            return
        asked_question = bool(turn.assistant) and turn.assistant[-1].rstrip().endswith("?")
        followup_s = self.cfg.listen.followup_after_question_s if asked_question else self.cfg.listen.followup_s
        if followup_s > 0 and not self._effective_mute():
            src = self._open_window(turn.source, followup_s, WindowReason.FOLLOWUP)
            if src is not None:
                self._followup_deadline = time.monotonic() + followup_s
                self._set_state(State.FOLLOWUP, "turn end")
                self._timer("followup", followup_s + 0.5, self._followup_expired)
                return
        self._set_state(State.IDLE, "turn end")

    def _followup_expired(self) -> None:
        if self.turn is None and self.state == State.FOLLOWUP:
            self._set_state(State.IDLE, "followup timeout")

    async def _watchdog(self, turn: Turn) -> None:
        """No brain event for max_turn_s + 60 s → spoken error, brain.reset() (N4)."""
        limit = float(self.cfg.brain.max_turn_s) + 60.0
        while self.turn is turn and not turn.brain_done:
            wait = turn.last_event + limit - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            log.error("turn %d: no brain event for %.0fs; resetting the brain", turn.id, limit)
            self.metrics.incr("watchdog_resets")
            self._on_error(turn, BrainEvent(BrainEventKind.ERROR, turn.id, text=P.VOICE_LOST,
                                            error_kind=BrainErrorKind.TIMEOUT, data={"watchdog": True}))
            self._tasks.spawn(self.brain.reset(), "brain.reset")
            return

    # ================================================================== brain events

    async def _consume_brain(self) -> None:
        async for ev in self.brain.events():
            try:
                self.on_brain_event(ev)
            except Exception:
                log.exception("brain event handling failed: %s", ev.kind)

    def on_brain_event(self, ev: BrainEvent) -> None:
        turn = self.turn
        if ev.kind == BrainEventKind.STATUS and (turn is None or ev.turn_id != turn.id):
            self._on_status(ev, None)
            return
        if turn is None or ev.turn_id != turn.id or turn.brain_done:
            self.metrics.incr("stale_brain_events")
            self.eventlog.emit("brain_stale", turn_id=ev.turn_id, kind=ev.kind.value, current=turn.id if turn else None)
            return
        turn.last_event = time.monotonic()
        k = ev.kind
        if k in (BrainEventKind.TEXT, BrainEventKind.SAY) and turn.first_text_t is None:
            turn.first_text_t = time.monotonic()
            self.metrics.observe_since("brain_first_text", turn.asked_t or turn.started)
        if k == BrainEventKind.TEXT:
            if ev.data.get("tool_follows"):  # the text so far was a step line: HUD only, never spoken
                self._drop_held(turn)
                return
            for s in turn.segmenter.feed(ev.text):
                self._hold(turn, s)
        elif k == BrainEventKind.SAY:
            self._flush_segmenter(turn)
            if ev.text.rstrip().endswith("?"):  # a real question for the user is spoken
                turn.assistant.append(ev.text)
                self._speak(turn, ev.text, "say")
            else:
                self._activity("narration", turn.id, text=ev.text)
        elif k == BrainEventKind.TOOL_START:
            self._drop_held(turn)
            turn.current_tool = ev.tool_name
            turn.current_narration = ev.data.get("narration") or self.describe_tool(ev.tool_name, ev.tool_input)
            self.eventlog.emit("tool_start", turn_id=turn.id, tool=ev.tool_name, tool_use_id=ev.tool_use_id,
                               input=_truncate(ev.tool_input))
            self._activity("tool_start", turn.id, tool=ev.tool_name, text=turn.current_narration or ev.tool_name)
        elif k == BrainEventKind.TOOL_END:
            turn.current_tool = None
            turn.current_narration = None
            self._activity("tool_end", turn.id, tool=ev.tool_name, is_error=ev.is_error)
        elif k == BrainEventKind.CONFIRM:
            self._on_confirm(turn, ev)
        elif k == BrainEventKind.STATUS:
            self._on_status(ev, turn)
        elif k == BrainEventKind.DONE:
            self._flush_segmenter(turn)
            turn.brain_done = True
            turn.current_tool = None
            self.eventlog.emit("brain_done", turn_id=turn.id, **_jsonable(ev.data))
            if not turn.finishing:
                turn.finishing = True
                turn.tasks.spawn(self._finish_turn(turn), "finish")
        elif k == BrainEventKind.ERROR:
            self._on_error(turn, ev)

    def _on_status(self, ev: BrainEvent, turn: Turn | None) -> None:
        self._activity("status", turn.id if turn else (ev.turn_id or None), text=ev.text, **_jsonable(ev.data))

    def _hold(self, turn: Turn, s: str) -> None:
        """Hold answer text until we know it is the answer (not a step line before a tool call).
        Released when the turn ends, or once it is clearly an answer (2+ sentences or 1.5 s)."""
        turn.held.append(s)
        now = time.monotonic()
        if turn.held_t is None:
            turn.held_t = now
            turn.tasks.spawn(self._release_later(turn, turn.held_t), "hold")
        if turn.releasing or len(turn.held) >= HOLD_MAX_SENTENCES or now - turn.held_t >= HOLD_S:
            self._release_held(turn)

    async def _release_later(self, turn: Turn, t0: float) -> None:
        await asyncio.sleep(HOLD_S)
        if self.turn is turn and turn.held_t == t0:
            self._release_held(turn)

    def _release_held(self, turn: Turn) -> None:
        turn.releasing = True
        held, turn.held = turn.held, []
        for s in held:
            turn.assistant.append(s)
            self._speak(turn, s, "answer")

    def _drop_held(self, turn: Turn) -> None:
        if turn.releasing:  # an answer already being spoken: finish its last sentence
            for s in turn.segmenter.flush():
                turn.assistant.append(s)
                self._speak(turn, s, "answer")
            dropped = []
        else:
            dropped = [*turn.held, *turn.segmenter.flush()]
        turn.held, turn.held_t, turn.releasing = [], None, False
        if dropped:
            self._activity("narration", turn.id, text=" ".join(dropped))

    def _flush_segmenter(self, turn: Turn) -> None:
        for s in turn.segmenter.flush():
            turn.held.append(s)
        self._release_held(turn)
        turn.held_t, turn.releasing = None, False

    def _on_error(self, turn: Turn, ev: BrainEvent) -> None:
        if turn.brain_done:
            return
        self._flush_segmenter(turn)
        kind = ev.error_kind or BrainErrorKind.INTERNAL
        msg = ev.text.strip() or ERROR_MESSAGES.get(kind, ERROR_MESSAGES[BrainErrorKind.INTERNAL])
        self.metrics.incr("errors")
        turn.errored = True
        turn.brain_done = True
        self.eventlog.emit("brain_error", turn_id=turn.id, error_kind=kind.value, message=msg, **_jsonable(ev.data))
        self.publish({"event": "error", "turn_id": turn.id, "kind": kind.value, "message": msg})
        self._play_earcon("error", turn.output)
        self._speak(turn, msg, "say")
        self._set_state(State.ERROR, kind.value)
        if not turn.finishing:
            turn.finishing = True
            turn.tasks.spawn(self._finish_turn(turn), "finish")

    # ================================================================== confirmation

    def _confirm_source(self, turn: Turn) -> str | None:
        lst = self._listener_for(turn.source)
        return lst.source_name if lst else None

    def _on_confirm(self, turn: Turn, ev: BrainEvent) -> None:
        if turn.confirming:
            return  # a second request while one is pending: the brain answers it (N16)
        self._flush_segmenter(turn)
        target = str(ev.data.get("target") or "")
        line = P.confirmation_line(ev.text, target)  # humanized, the raw target is never spoken
        turn.confirming = True
        if turn.first_text_t is None:
            turn.first_text_t = time.monotonic()  # no slow-start tick / "One moment" around the question
        self.metrics.incr("confirms")
        self.eventlog.emit("confirm_request", turn_id=turn.id, summary=ev.text, target=target, spoken=line)
        self._set_state(State.CONFIRMING, "confirmation requested")
        self._speak(turn, line, "confirm")  # spoken exactly once, as one item
        window = self.cfg.listen.confirm_window_s
        # Armed now, independently of speech: a stalled output can't leave us in CONFIRMING.
        self._timer("confirm", window + CONFIRM_SPEECH_SLACK_S, lambda: self._resolve_confirm(turn, "timeout", "timeout"))
        turn.tasks.spawn(self._open_confirm_window(turn), "confirm-window")

    async def _open_confirm_window(self, turn: Turn) -> None:
        # The listener's pre-roll catches an early "yes" said over the end of the prompt.
        await self.speech.drain(turn.id, timeout=CONFIRM_SPEECH_SLACK_S)
        if self.turn is not turn or not turn.confirming:
            return
        window = self.cfg.listen.confirm_window_s
        src = self._open_window(turn.source, window, WindowReason.CONFIRM)
        if src is None:
            log.warning("no listener to capture the confirmation; denying")
            self._resolve_confirm(turn, "no", "no listener")
            return
        self._timer("confirm", window + 1.0, lambda: self._resolve_confirm(turn, "timeout", "timeout"))

    def _resolve_confirm(self, turn: Turn, verdict: str, why: str, text: str = "", *,
                         confirm_source: str = "system") -> None:
        """verdict: 'yes' | 'no' | 'steer' | 'timeout' (see phrases.classify_confirmation);
        confirm_source: 'voice' | 'text' | 'command' | 'system' (timeouts, sleep)."""
        if self.turn is not turn or not turn.confirming:
            return
        turn.confirming = False
        self._cancel_timer("confirm")
        self._close_windows()
        approved = verdict == "yes"
        try:
            self.gate.resolve(approved)
        except Exception:
            log.exception("gate.resolve failed")
        self.metrics.incr("confirms_approved" if approved else "confirms_denied")
        self.eventlog.emit("confirm_result", turn_id=turn.id, approved=approved, verdict=verdict, reason=why, text=text,
                           confirm_source=confirm_source)
        self._activity("confirm_result", turn.id, approved=approved, reason=why, text=text, confirm_source=confirm_source)
        self._set_state(State.THINKING, f"confirm {why}")
        if verdict == "steer":
            self._steer(turn, text)  # "yes, but change the subject": the model revises and asks again
        elif verdict != "yes":  # approval gets no ack of ours: the model says the one word
            self._speak(turn, P.DENIED if verdict == "no" else P.CONFIRM_TIMEOUT, "filler")

    # ================================================================== narration / liveness

    async def _slow_start(self, turn: Turn) -> None:
        """Cover a slow first response: a soft tick at 3 s, "Still working on it." at 7 s (UX#5)."""
        await asyncio.sleep(P.SLOW_BLIP_S)
        if self.turn is not turn or turn.first_text_t is not None or turn.confirming:
            return
        if not await self.speech.drain(turn.id, timeout=P.SLOW_SAY_S):  # never overlap the filler
            return
        if self.turn is not turn or turn.first_text_t is not None:
            return
        self._play_earcon("tick", turn.output)
        await asyncio.sleep(max(0.0, P.SLOW_SAY_S - (time.monotonic() - turn.started)))
        if self.turn is turn and turn.first_text_t is None and not turn.confirming:
            self.metrics.incr("slow_starts")
            self._speak(turn, P.STILL_WORKING, "narration")
            turn.last_heartbeat = time.monotonic()

    async def _narration_loop(self, turn: Turn) -> None:
        silence = max(0.5, float(self.cfg.brain.narration_silence_s))
        while self.turn is turn and not turn.brain_done:
            now = time.monotonic()
            if self.speech.speaking or self.speech.pending(turn.id) > 0 or turn.confirming:
                turn.last_speech = now
            wait = turn.last_speech + silence - now
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            line = turn.current_narration
            if line and line not in turn.narrated:  # step lines go to the HUD only, never aloud
                turn.narrated[line] = now
                self._activity("narration", turn.id, text=line)
            text = self._heartbeat_text(turn, now)
            turn.last_speech = now  # rate limit even if nothing was said
            if text:
                self.metrics.incr("narrations")
                self._activity("narration", turn.id, text=text)
                self._speak(turn, text, "narration")

    def _heartbeat_text(self, turn: Turn, now: float) -> str | None:
        """One spoken "Still on it." on a long silent task; nothing more (no step-by-step talk)."""
        if (turn.heartbeats == 0 and now - turn.started >= P.HEARTBEAT_EVERY_S
                and now - turn.last_heartbeat >= P.HEARTBEAT_EVERY_S):
            turn.last_heartbeat = now
            turn.heartbeats += 1
            return P.HEARTBEATS[0]
        return None

    # ================================================================== commands (called from control.py)

    async def ask(self, text: str, source: str = "control") -> int:
        """Text input (macctl ask / the app). Goes through exactly the spoken-utterance policy:
        in CONFIRMING it answers the confirmation, mid-turn it stops/steers/redirects, otherwise
        it starts a turn. Returns the id of the turn it started or acted on."""
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        window = {State.CONFIRMING: WindowReason.CONFIRM, State.FOLLOWUP: WindowReason.FOLLOWUP}.get(self.state)
        now = time.monotonic()
        utt = Utterance(text=text, source=source, woke=True, started_at=now, ended_at=now, audio_s=0.0, window=window)
        await self._on_utterance(ListenerEvent(ListenerEventKind.UTTERANCE, source, utterance=utt), utt)
        turn = self.turn or self.last_turn
        return turn.id if turn else 0

    def recent_user_text(self, n: int = 10) -> list[str]:
        """The last `n` things the user said or typed (newest last), for the brain's memory."""
        return list(self._recent_user)[-max(0, n):] if n > 0 else []

    async def _handoff(self, reason: str) -> None:
        """Let the brain save session notes before a reset / OFF sleep. Bounded at 60 s; never
        blocks the wake path (runs as its own task; a wake during it starts a turn normally)."""
        handoff = getattr(self.brain, "handoff", None)  # Brain.handoff lands with B1's memory work
        if handoff is None:
            return
        if self._handoff_task is None or self._handoff_task.done():
            self._activity("handoff", None, text="Saving notes", reason=reason)
            self._handoff_task = self._tasks.spawn(asyncio.wait_for(handoff(), HANDOFF_TIMEOUT_S), "brain.handoff")
        try:
            await asyncio.shield(self._handoff_task)
        except TimeoutError:
            log.warning("brain.handoff timed out after %.0fs (%s)", HANDOFF_TIMEOUT_S, reason)
        except Exception:
            log.exception("brain.handoff failed (%s)", reason)

    def confirm(self, approved: bool) -> int:
        """Answer a pending confirmation from a command (the app's Yes/No buttons)."""
        turn = self.turn
        if turn is None or not turn.confirming:
            raise ValueError("nothing pending")
        self._resolve_confirm(turn, "yes" if approved else "no", "command", confirm_source="command")
        return turn.id

    def listen(self) -> str | None:
        """Push-to-talk: capture the next utterance without a wake word."""
        if self._effective_mute():
            return None
        source = self._primary or ""
        if self.remote["present"]:
            source = next((n for n in self._listeners if n.startswith("remote")), source)
        src = self._open_window(source, self.cfg.listen.no_speech_timeout_s, WindowReason.PTT)
        if src is None:
            return None
        self._cancel_timer("followup")
        self._play_earcon("listening", self._output_for_source(src))
        if self.turn is None:
            self._set_state(State.LISTENING, "push-to-talk")
        return src

    async def stop(self) -> bool:
        if self.turn is not None:
            self.metrics.incr("interrupts")
            await self._cancel_turn(self.turn, "stop command", speak=False)
            return True
        self.speech.flush_all()
        self._cancel_timer("followup")
        if self.state not in (State.IDLE, State.OFF):
            self._set_state(State.IDLE, "stop command")
        return False

    def _effective_mute(self) -> bool:
        return self.muted or self.asleep or self.off

    def _apply_mute(self) -> None:
        effective = self._effective_mute()
        for lst in self._listeners.values():
            with contextlib.suppress(Exception):
                lst.set_muted(effective)
        if effective:
            self._close_windows()

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        self._apply_mute()
        self._activity("mute", None, muted=self.muted, text="Muted" if self.muted else "Unmuted")
        self.publish(self._state_event(prev=self.state.value, reason="mute"))

    def set_off(self, off: bool, *, persist: bool = True) -> None:
        """OFF: no wake word, mic released by the app, claude process asleep. Persisted."""
        off = bool(off)
        if off == self.off:
            return
        self.off = off
        if off:
            self._cancel_timer("followup")
            self.speech.flush_all()
            self._apply_mute()
            self._set_state(State.OFF, "off")
            self._tasks.spawn(self._sleep_brain(), "brain.sleep")
        else:
            self._apply_mute()
            self._set_state(State.IDLE, "on")
            self._play_earcon("done", self.speech.default_output)
        self._activity("off" if off else "on", None, text="Listening off" if off else "Listening on")
        if persist:
            self._tasks.spawn(asyncio.to_thread(config.update_state, listening_off=off), "persist-off")

    async def _sleep_brain(self) -> None:
        """Shut Claude down for OFF even if a turn was in flight: the brain ignores sleep()
        while draining, so re-issue until it is asleep (N8)."""
        if self.turn is not None:
            await self._cancel_turn(self.turn, "off", speak=False)
        await self._handoff("off")
        for _ in range(15):
            if not self.off:
                return
            await self.brain.sleep()
            if not self.brain.alive:
                return
            await asyncio.sleep(1.0)
        log.warning("brain still alive 15 s after OFF")

    def _go_off_spoken(self, output: str) -> None:
        self._say_system(P.GOING_OFF, output)

        async def after_speech() -> None:
            await self.speech.drain(SYSTEM_TURN, timeout=10.0)
            self.set_off(True)

        self._tasks.spawn(after_speech(), "go-off")

    def say(self, text: str) -> None:
        if text.strip():
            self._say_system(text.strip())

    async def reset(self) -> None:
        if self.turn is not None:
            await self._cancel_turn(self.turn, "reset", speak=False)
        with contextlib.suppress(Exception):
            self.gate.revoke_all()
        await self._handoff("reset")
        await self.brain.reset()
        self._activity("reset", None, text="New conversation")

    async def power(self, event: str) -> None:
        if event == "will_sleep":
            self.asleep = True
            self._apply_mute()
            if self.turn is not None and self.turn.confirming:
                self._resolve_confirm(self.turn, "timeout", "sleep")
        elif event == "did_wake":
            self.asleep = False
            self._apply_mute()
        elif event in ("screen_locked", "screen_unlocked"):
            await self.set_locked(event == "screen_locked")
            return
        else:
            raise ValueError(f"unknown power event {event!r}")
        self._activity("power", None, power_event=event, text=event.replace("_", " "))

    async def set_locked(self, locked: bool) -> None:
        """Screen lock state from the app: no behaviour change here (listening, turns and speech
        continue); Claude is told so it knows UI/Chrome tools won't work until unlock."""
        changed = locked != self.locked
        self.locked = locked
        set_context = getattr(self.brain, "set_context", None)  # lands with B1's context work
        if set_context is not None:
            res = set_context(locked=locked)
            if asyncio.iscoroutine(res):
                await res
        if changed:
            self._activity("power", None, power_event="screen_locked" if locked else "screen_unlocked",
                           text="Screen locked" if locked else "Screen unlocked")
            self.publish(self._state_event(prev=self.state.value, reason="lock"))

    # ================================================================== remote (N7)

    def on_remote_session(self, pin: str | None) -> None:
        """Bridge: a session started (with its 4-digit PIN) or ended (None)."""
        self.remote.update(active=pin is not None, pin=pin, verified=False)
        if pin is None:
            self.set_remote_present(False)
        self._activity("remote_session", None, active=pin is not None, text="Phone link ready" if pin else "Phone link ended")
        self._publish_remote()

    def _publish_remote(self) -> None:
        self.publish({"event": "remote", **self.remote})

    def set_remote_present(self, present: bool, identity: str | None = None) -> None:
        if present == self.remote["present"]:
            return
        self.remote["present"] = present
        self.remote["identity"] = identity if present else None
        if not present:
            self.remote["verified"] = False
        self._activity("remote", None, present=present, text="Phone connected" if present else "Phone left")
        self._publish_remote()
        self.publish(self._state_event(prev=self.state.value, reason="remote"))  # refresh the app's status line
        if not present:
            return
        local = self.speech.default_output
        if self.turn is not None and not self.turn.remote:
            self._tasks.spawn(self._cancel_turn(self.turn, "remote joined", speak=False), "remote-preempt")
            self._say_system(P.PHONE_PREEMPTED, local)
        else:
            self._say_system(P.PHONE_CONNECTED, local)

    def _check_pin(self, source: str, text: str) -> None:
        pin = self.remote.get("pin")
        output = self._output_for_source(source)
        if pin and P.pin_matches(text, pin):
            self.remote["verified"] = True
            self._activity("remote_verified", None, text="Phone verified")
            self._publish_remote()
            self._say_system(P.PIN_OK, output)
        else:
            self.metrics.incr("remote_pin_failures")
            self._say_system(P.PIN_WRONG if P.spoken_digits(text) else P.PIN_PROMPT, output)

    # ================================================================== status

    def status(self) -> dict[str, Any]:
        turn = self.turn
        return {
            "state": self.state.value,
            "turn_id": turn.id if turn else None,
            "turn": {"text": turn.text, "source": turn.source, "tool": turn.current_tool,
                     "age_s": round(time.monotonic() - turn.started, 1)} if turn else None,
            "muted": self.muted,
            "off": self.off,
            "locked": self.locked,
            "asleep": self.asleep,
            "voice": self.voice_ok,
            "voice_reason": self.voice_reason,
            "brain_alive": self.brain.alive,
            "remote": dict(self.remote),
            "listeners": {n: {"requires_wake": self._requires_wake.get(n, True)} for n in self._listeners},
            "primary_listener": self._primary,
            "speech": self.speech.snapshot(),
            "metrics": self.metrics.snapshot(),
            "eventlog_dropped": self.eventlog.dropped,
            "uptime_s": round(time.monotonic() - self.started, 1),
        }


def _truncate(obj: Any, limit: int = 400) -> Any:
    try:
        s = json.dumps(obj, default=repr)
    except (TypeError, ValueError):
        s = repr(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in ("event", "kind", "turn_id", "type", "ts", "mono", "text"):
            k = f"data_{k}"
        out[k] = v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else repr(v)
    return out
