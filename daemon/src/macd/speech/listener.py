"""ListenerImpl: KWS -> stage-2 verify -> VAD endpointing -> STT for one AudioInput.

See contracts.Listener and ARCHITECTURE §3/§4.

Concurrency: bookkeeping and VAD (~0.1 ms) run on the event loop. KWS runs pipelined on a
dedicated single-thread executor ("kws_pool"): 80 ms batches are submitted without
waiting and their results collected on later frames, so a slow KWS step (the zipformer
encoder costs ~10 ms every 320 ms) never stalls VAD/endpointing. Only when the executor
falls more than ``MAX_KWS_INFLIGHT`` batches behind (file feeding faster than real time)
does the frame loop wait for it. STT always goes to the STT executor. All endpointing decisions are made in *sample time* (absolute sample indices
of the source), never wall-clock, so file-fed tests are deterministic at any feed speed.

Jobs that need STT (verify, utterance) are chained in capture order (each awaits its
predecessor before emitting), so events come out in the order the audio happened.

Capture model: one ``_Capture`` at a time. A woke capture stays open until its no-speech
deadline: when a segment ends in silence it is transcribed and the capture *restarts*
(same object, fresh audio) in case the transcript was just the wake phrase ("Hey Mac …
pause … open mail"). When a segment hits ``max_utterance_s`` it is transcribed and the
capture restarts with speech already in progress; that text is a ``continuation``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from macd.audio.ring import RingBuffer
from macd.config import Config
from macd.contracts import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    STT,
    VAD,
    AudioInput,
    ListenerEvent,
    ListenerEventKind,
    Utterance,
    WakeDetector,
    WakeHit,
    WindowReason,
)
from macd.eventlog import Metrics
from macd.phrases import is_stop
from macd.speech.wake import WakePhrases

log = logging.getLogger(__name__)

PREROLL_S = 2.5
VERIFY_WINDOW_S = 2.0
VERIFY_TAIL_S = 0.1  # audio after the hit before verifying (the hit already follows a trailing blank)
WAKE_LOOKBACK_S = 1.6  # utterance audio starts this far before the hit (wake phrase is stripped)
WINDOW_PREROLL_S = 0.3
WINDOW_START_FRAMES = 3  # consecutive speech frames needed to open a window capture
SAME_BREATH_S = 0.6  # speech starting within this of the hit = same breath (no wake blip needed)
STOP_CAPTURE_S = 2.0  # stop-word captures while speaking are at most this long
TRAILING_KEEP_S = 0.3
KWS_BATCH_FRAMES = 4  # feed KWS 80 ms at a time: the encoder steps every 16 frames anyway, 4x fewer thread hops
MAX_KWS_INFLIGHT = 2  # batches the executor may lag before the frame loop waits (never in real time)

def _samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


@dataclass
class _Capture:
    woke: bool
    start: int  # absolute index of the first captured sample of the current segment
    deadline: int  # woke: give up (NO_SPEECH) if no speech by then
    max_end: int
    hit: WakeHit | None = None
    window: WindowReason | None = None
    parts: list[np.ndarray] = field(default_factory=list)
    end: int = 0  # absolute index after the last captured sample
    speech_started: bool = False  # endpointing armed (silence after last_speech ends the segment)
    speech_seen: bool = False  # VAD saw speech in this segment (SPEECH_START, same_breath)
    speech_start: int = 0
    last_speech: int = 0
    verify_at: int | None = None  # launch stage-2 verify once end >= verify_at
    verified: bool = False
    rejected: bool = False
    speech_start_pending: bool = False  # SPEECH_START held back until WAKE is emitted
    capped: bool = False  # current segment continues one that hit max_utterance_s
    produced_text: bool = False  # set by the utterance job once this capture yielded text
    wall_hit: float = 0.0

    def restart(self, pos: int, *, speaking: bool, max_end: int) -> None:
        self.parts, self.start, self.end = [], pos, pos
        self.speech_started = self.speech_seen = speaking
        self.speech_start = self.last_speech = pos
        self.capped = speaking
        self.max_end = max_end


class ListenerImpl:
    """contracts.Listener implementation for one AudioInput."""

    def __init__(
        self,
        source: AudioInput,
        wake: WakeDetector | None,
        vad: VAD,
        stt: STT,
        cfg: Config,
        aec: bool | Callable[[], bool] = False,
        metrics: Metrics | None = None,
    ) -> None:
        self.source = source
        self.source_name = source.name
        self.speaking = False
        self.metrics = metrics or Metrics()
        self.frame_ms_max = 0.0  # loop time per frame (KWS executor wait excluded)
        self.kws_ms_max = 0.0
        self._wake = wake
        self._vad = vad
        self._stt = stt
        self._cfg = cfg
        self._aec = aec
        self._phrases = WakePhrases(cfg.wake.phrases)
        self._requires_wake = source.requires_wake_word and wake is not None
        self._ring = RingBuffer(_samples(PREROLL_S))
        self._pos = 0  # absolute index of the next sample
        self._capture: _Capture | None = None
        self._window_end: int | None = None
        self._window_reason: WindowReason | None = None
        self._speech_run = 0  # consecutive VAD speech frames (for window starts)
        self._muted = False
        self._current_text = ""
        self._guard_until = 0
        self._queue: asyncio.Queue[ListenerEvent | None] = asyncio.Queue()
        self._chain: asyncio.Task[None] | None = None
        self._jobs: set[asyncio.Task[None]] = set()
        self._kws_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kws_pool")
        self._kws_batch: list[np.ndarray] = []
        self._kws_inflight: deque[tuple[asyncio.Future[WakeHit | None], float]] = deque()
        self._vad_active = False  # VAD only runs while something can use it (idle CPU)
        self._stt_missing_logged = False
        lc = cfg.listen
        self._silence = _samples(lc.silence_ms / 1000)
        self._max_utt = _samples(lc.max_utterance_s)
        self._no_speech = _samples(lc.no_speech_timeout_s)
        self._vad_th = lc.vad_threshold
        self._tail_guard = _samples(lc.tail_guard_ms / 1000)

    # ------------------------------------------------------------------ public API

    @property
    def aec(self) -> bool:
        return bool(self._aec() if callable(self._aec) else self._aec)

    async def run(self) -> None:
        try:
            async for frame in self.source.frames():
                await self._on_frame(np.asarray(frame, dtype=np.float32))
            while self._kws_inflight:  # late hits at end of input still count (file sources)
                await self._collect_kws(wait=True)
        finally:
            self._kws_pool.shutdown(wait=False)
            self._end_of_input()
            while self._chain is not None and not self._chain.done():
                await asyncio.shield(self._chain)
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[ListenerEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    def open_window(self, seconds: float, reason: WindowReason) -> None:
        self._window_end = self._pos + _samples(seconds)
        self._window_reason = reason
        # PTT starts on the very first speech frame; others need a short run.
        self._speech_run = WINDOW_START_FRAMES if reason is WindowReason.PTT else 0

    def close_window(self) -> None:
        self._window_end = None
        self._window_reason = None
        c = self._capture
        if c is not None and not c.woke and c.window is not WindowReason.SPEAKING:
            self._capture = None

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self._capture = None
            self._window_end = None
            if self._wake is not None:
                self._kws_pool.submit(self._wake.reset)  # same thread as accept(): no race

    def set_wake_detector(self, det: WakeDetector | None) -> None:
        """Swap the KWS detector between frames (batches already in flight finish on the old one)."""
        self._wake = det
        self._requires_wake = self.source.requires_wake_word and det is not None
        self._kws_batch.clear()

    def set_wake_phrases(self, phrases: WakePhrases) -> None:
        """Swap the stage-2 verifier/stripper (single attribute store: safe from any thread)."""
        self._phrases = phrases

    def set_speaking(self, speaking: bool, current_text: str = "") -> None:
        if self.speaking and not speaking:
            if not self.aec:
                self._guard_until = self._pos + self._tail_guard
            c = self._capture
            if c is not None and c.window is WindowReason.SPEAKING:
                self._capture = None  # playback ended: no more stop-word captures
        self.speaking = speaking
        self._current_text = current_text if speaking else ""

    # ------------------------------------------------------------------ frame path

    async def _on_frame(self, frame: np.ndarray) -> None:
        t0 = time.perf_counter()
        idx = self._pos
        self._pos += frame.shape[0]
        self._ring.append(frame)
        if not self._muted and self._requires_wake and self._wake is not None:
            self._kws_batch.append(frame)
            if len(self._kws_batch) >= KWS_BATCH_FRAMES:
                chunk = np.concatenate(self._kws_batch)
                self._kws_batch.clear()
                fut = asyncio.get_running_loop().run_in_executor(self._kws_pool, self._wake.accept, chunk, self._pos - chunk.shape[0])
                self._kws_inflight.append((fut, time.perf_counter()))
        t_wait = time.perf_counter()
        await self._collect_kws(wait=len(self._kws_inflight) > MAX_KWS_INFLIGHT)
        waited = time.perf_counter() - t_wait
        vad_needed = self._capture is not None or self._window_end is not None or self.speaking
        if vad_needed and not self._vad_active:
            self._vad.reset()
        self._vad_active = vad_needed
        is_speech = vad_needed and self._vad.accept(frame) >= self._vad_th
        self._speech_run = self._speech_run + 1 if is_speech else 0
        if not self._muted:
            self._step_capture(frame, idx, is_speech)
            self._step_window(is_speech)
        dt = (time.perf_counter() - t0 - waited) * 1000
        self.frame_ms_max = max(self.frame_ms_max, dt)
        self.metrics.observe("listener_frame", dt)
        if dt > 15:
            log.warning("listener frame took %.1f ms (budget 20 ms)", dt)

    async def _collect_kws(self, *, wait: bool) -> None:
        """Harvest finished KWS batches in order (``wait``: block for the oldest one)."""
        while self._kws_inflight and (wait or self._kws_inflight[0][0].done()):
            fut, t0 = self._kws_inflight.popleft()
            wait = False
            try:
                hit = await fut
            except Exception:
                log.exception("KWS failed on a batch")
                continue
            ms = (time.perf_counter() - t0) * 1000  # submit -> result, incl. queueing
            self.kws_ms_max = max(self.kws_ms_max, ms)
            self.metrics.observe("listener_kws", ms)
            if hit is not None and not self._muted:
                self._on_hit(hit)

    def _on_hit(self, hit: WakeHit) -> None:
        self.metrics.incr("kws_hits")
        if self.speaking and not self.aec and self._phrases.mentions(self._current_text):
            self.metrics.incr("wakes_suppressed")
            log.info("wake hit suppressed (self-trigger: speaking %r)", self._current_text[:40])
            return
        cur = self._capture
        if cur is not None and cur.woke and not cur.rejected:
            return  # already capturing after a wake word
        start = max(self._ring.oldest, hit.sample_index - _samples(WAKE_LOOKBACK_S))
        # The KWS hit can land after a short command fused with the wake phrase ("Hey Mac, stop"),
        # so endpointing is armed right away: silence after the hit transcribes the look-back.
        cap = _Capture(
            woke=True,
            start=start,
            deadline=hit.sample_index + self._no_speech,
            max_end=hit.sample_index + self._max_utt,
            hit=hit,
            speech_started=True,
            speech_start=hit.sample_index,
            last_speech=hit.sample_index,
            wall_hit=time.monotonic(),
        )
        cap.parts.append(self._ring.slice(start, self._pos))
        cap.end = self._pos
        if self._cfg.wake.verify:
            cap.verify_at = hit.sample_index + _samples(VERIFY_TAIL_S)
        else:
            cap.verified = True
            self._enqueue(self._emit_wake(cap))
        self._capture = cap

    def _step_capture(self, frame: np.ndarray, idx: int, is_speech: bool) -> None:
        c = self._capture
        if c is None:
            return
        if c.rejected:
            self._capture = None
            return
        if idx >= c.end:  # frames already included via the pre-roll slice are skipped
            c.parts.append(frame.copy())
            c.end = idx + frame.shape[0]
        if c.verify_at is not None and c.end >= c.verify_at:
            c.verify_at = None
            self._enqueue(self._verify(c, self._ring.slice(c.end - _samples(VERIFY_WINDOW_S), c.end)))
        after_hit = c.hit is None or idx >= c.hit.sample_index
        if is_speech and after_hit:
            c.last_speech = c.end
            if not c.speech_seen:
                c.speech_seen = c.speech_started = True
                c.speech_start = idx - (self._speech_run - 1) * FRAME_SAMPLES
                if c.woke and not c.verified:
                    c.speech_start_pending = True  # emitted right after WAKE (keeps event order)
                else:
                    self._enqueue(self._emit(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name)))
        if c.speech_started:
            if c.end >= c.max_end:
                self._finalize(c, capped=is_speech)  # still talking: the rest continues as dictation
            elif c.end - c.last_speech >= self._silence:
                self._finalize(c, capped=False)
        elif c.end >= c.deadline:
            self._capture = None
            self._enqueue(self._no_speech_job(c))

    def _step_window(self, is_speech: bool) -> None:
        if self._capture is not None:
            return
        suppressed = (self.speaking or self._pos < self._guard_until) and not self.aec
        if self._window_end is not None:
            if is_speech and not suppressed and self._speech_run >= WINDOW_START_FRAMES:
                self._start_vad_capture(self._window_reason, self._pos + self._max_utt)
                self._window_end = None  # one utterance per window
            elif self._pos >= self._window_end:
                reason, self._window_end, self._window_reason = self._window_reason, None, None
                self._enqueue(self._emit(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name, window=reason)))
        elif self.speaking and is_speech and self._speech_run >= WINDOW_START_FRAMES:
            # Stop-word capture while Mac talks ("stop", "that's enough"): short, VAD-segmented.
            self._start_vad_capture(WindowReason.SPEAKING, self._pos + _samples(STOP_CAPTURE_S))

    def _start_vad_capture(self, window: WindowReason | None, max_end: int) -> None:
        start = max(self._ring.oldest, self._pos - _samples(WINDOW_PREROLL_S) - self._speech_run * FRAME_SAMPLES)
        cap = _Capture(
            woke=False,
            start=start,
            deadline=max_end,
            max_end=max_end,
            window=window,
            speech_started=True,
            speech_seen=True,
            speech_start=start,
            last_speech=self._pos,
        )
        cap.parts.append(self._ring.slice(start, self._pos))
        cap.end = self._pos
        self._capture = cap
        if window is not WindowReason.SPEAKING:
            self._enqueue(self._emit(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name)))

    def _finalize(self, c: _Capture, *, capped: bool) -> None:
        """Transcribe the current segment; ``capped`` = speech is still going at max_utterance_s."""
        keep_to = min(c.end, c.last_speech + _samples(TRAILING_KEEP_S))
        audio = np.concatenate(c.parts)[: max(0, keep_to - c.start)]
        same_breath = c.woke and c.hit is not None and not c.capped and c.speech_start - c.hit.sample_index <= _samples(SAME_BREATH_S)
        # (no post-hit speech at all = the command was fused with the wake phrase: same breath)
        self._enqueue(self._utterance_job(c, audio, time.monotonic(), continuation=c.capped, same_breath=same_breath))
        if capped and c.window is not WindowReason.SPEAKING:
            c.restart(self._pos, speaking=True, max_end=self._pos + self._max_utt)
        elif c.woke and self._pos < c.deadline:
            c.restart(self._pos, speaking=False, max_end=self._pos + self._max_utt)  # maybe only "hey mac" so far
        else:
            self._capture = None

    def _end_of_input(self) -> None:
        c, self._capture = self._capture, None
        if c is not None and c.speech_started:
            self._finalize(c, capped=False)
            self._capture = None

    # ------------------------------------------------------------------ jobs (chained)

    def _enqueue(self, coro: Coroutine[Any, Any, None]) -> None:
        prev = self._chain

        async def job() -> None:
            if prev is not None:
                with contextlib.suppress(Exception):  # predecessor already logged its error
                    await prev
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("listener job failed")

        task = asyncio.get_running_loop().create_task(job())
        self._chain = task
        self._jobs.add(task)
        task.add_done_callback(self._jobs.discard)

    async def _emit(self, ev: ListenerEvent) -> None:
        self._queue.put_nowait(ev)

    async def _emit_wake(self, c: _Capture) -> None:
        if c.wall_hit:
            self.metrics.observe_since("kws_hit_to_wake", c.wall_hit)
        self._window_end = None  # a verified wake supersedes any open window
        self._queue.put_nowait(ListenerEvent(ListenerEventKind.WAKE, self.source_name, wake=c.hit))
        if c.speech_start_pending:
            c.speech_start_pending = False
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.SPEECH_START, self.source_name))

    async def _transcribe(self, audio: np.ndarray, metric: str) -> str | None:
        """STT through the executor; None if the model is not installed (logged once)."""
        t0 = time.monotonic()
        try:
            text = await self._stt.transcribe(audio)
        except FileNotFoundError:
            if not self._stt_missing_logged:
                self._stt_missing_logged = True
                log.warning("STT model missing: wake hits are unverified and utterances are dropped")
            return None
        self.metrics.observe_since(metric, t0)
        return text.strip()

    async def _verify(self, c: _Capture, audio: np.ndarray) -> None:
        text = await self._transcribe(audio, "wake_verify_stt")
        if text is None or self._phrases.verify(text):
            c.verified = True  # no STT -> lighter path: trust the KWS hit
            await self._emit_wake(c)
        else:
            c.rejected = True
            self.metrics.incr("false_wakes")
            log.info("false wake rejected by stage-2 (heard %r)", text)
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.FALSE_WAKE, self.source_name, wake=c.hit))

    async def _utterance_job(self, c: _Capture, audio: np.ndarray, wall_end: float, *, continuation: bool, same_breath: bool) -> None:
        if c.rejected:
            return
        text = await self._transcribe(audio, "listener_stt")
        if text is None:
            return
        if c.woke and not c.produced_text:
            text = self._phrases.remove(text)
        if c.window is WindowReason.SPEAKING:
            if not is_stop(text) or (not self.aec and self._own_voice(text)):
                return  # anything but a bare stop word (or our own voice) is dropped silently
            self.metrics.incr("stop_words")
        if not text:
            if not c.woke:
                self._queue.put_nowait(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name, window=c.window))
            return  # woke: the restarted capture keeps listening until the deadline
        woke = c.woke and not c.produced_text
        c.produced_text = True
        dur = audio.shape[0] / SAMPLE_RATE
        utt = Utterance(
            text=text, source=self.source_name, woke=woke, started_at=wall_end - dur, ended_at=wall_end, audio_s=dur,
            window=c.window, same_breath=woke and same_breath, continuation=continuation,
        )
        self._queue.put_nowait(ListenerEvent(ListenerEventKind.UTTERANCE, self.source_name, utterance=utt, wake=c.hit if woke else None))
        if self._capture is c and not c.speech_seen:
            self._capture = None  # got the command; no need to keep waiting for one

    def _own_voice(self, heard: str) -> bool:
        """Self-trigger check without AEC: is Mac itself saying a word of the heard stop phrase?"""
        spoken = set(re.findall(r"[a-z']{4,}", self._current_text.lower()))
        return any(w in spoken for w in re.findall(r"[a-z']{4,}", heard.lower()))

    async def _no_speech_job(self, c: _Capture) -> None:
        if not c.rejected and not c.produced_text:
            self._queue.put_nowait(ListenerEvent(ListenerEventKind.NO_SPEECH, self.source_name))
