"""ListenerImpl with fake KWS/VAD/STT: deterministic, sample-time endpointing."""

from __future__ import annotations

import asyncio

import numpy as np

from macd.audio.file_backend import FileInput
from macd.config import Config
from macd.contracts import SAMPLE_RATE, ListenerEventKind, WakeHit, WindowReason
from macd.speech.listener import ListenerImpl

K = ListenerEventKind


class FakeWake:
    """Fires once when the stream passes ``hit_at`` samples."""

    def __init__(self, hit_at: int | None) -> None:
        self.hit_at = hit_at
        self.fired = False

    def accept(self, frame, sample_index):
        end = sample_index + len(frame)
        if self.hit_at is not None and not self.fired and end >= self.hit_at:
            self.fired = True
            return WakeHit("hey mac", end)
        return None

    def reset(self):
        self.fired = False


class FakeVAD:
    """Speech = frame energy above zero (test audio is 0.5 in speech regions)."""

    def accept(self, frame):
        return 1.0 if float(np.abs(frame).max()) > 0.1 else 0.0

    def reset(self):
        pass


class FakeSTT:
    def __init__(self, text: str = "hey mac what time is it", missing: bool = False) -> None:
        self.text, self.missing, self.calls = text, missing, []

    async def transcribe(self, pcm):
        await asyncio.sleep(0)
        if self.missing:
            raise FileNotFoundError("no model")
        self.calls.append(len(pcm) / SAMPLE_RATE)
        return "" if len(pcm) < SAMPLE_RATE // 5 else self.text


def audio(*regions: tuple[float, float], total_s: float = 6.0) -> np.ndarray:
    x = np.zeros(int(total_s * SAMPLE_RATE), np.float32)
    for a, b in regions:
        x[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] = 0.5
    return x


async def run(pcm, wake_at: float | None, stt=None, cfg=None, before=None, trailing=0.5):
    cfg = cfg or Config()
    src = FileInput(pcm, trailing_silence_s=trailing)
    stt = stt or FakeSTT()
    lis = ListenerImpl(src, FakeWake(int(wake_at * SAMPLE_RATE) if wake_at is not None else None), FakeVAD(), stt, cfg)
    if before:
        before(lis)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    return [e.kind for e in events], events, lis


async def test_wake_then_utterance_same_breath():
    # "hey mac" 1.0-1.6 s (hit at 1.6), command 1.7-2.6 s, then silence
    kinds, events, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6)
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE]
    utt = events[-1].utterance
    assert utt.woke and utt.text == "What time is it"
    assert utt.same_breath and not utt.continuation and utt.window is None
    assert events[0].wake.sample_index == int(1.6 * SAMPLE_RATE)
    assert 2.8 < utt.audio_s < 3.0  # 1.6 s look-back before the hit (clamped at 0) + command + 0.3 s tail
    assert lis.frame_ms_max < 20
    assert lis.metrics.latency("listener_frame")["n"] > 100


async def test_false_wake_rejected():
    kinds, _, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, stt=FakeSTT("I ate a big mac"))
    assert kinds == [K.FALSE_WAKE]
    assert lis.metrics.counters["false_wakes"] == 1


async def test_verify_disabled_trusts_kws():
    cfg = Config()
    cfg.wake.verify = False
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, cfg=cfg)
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE]


async def test_wake_without_speech_times_out():
    cfg = Config()
    cfg.listen.no_speech_timeout_s = 2.0
    kinds, _, _ = await run(audio((1.0, 1.6)), wake_at=1.6, cfg=cfg, stt=FakeSTT("hey mac"))
    assert kinds == [K.WAKE, K.NO_SPEECH]


async def test_command_fused_before_late_hit():
    """KWS hit lands after a short command ("Hey Mac, stop" -> hit at 1.9 s): silence after the
    hit transcribes the look-back, and the utterance counts as same breath."""
    kinds, events, _ = await run(audio((1.0, 1.8)), wake_at=1.9, stt=FakeSTT("Hey Macstop!"))
    assert kinds == [K.WAKE, K.UTTERANCE]
    assert events[-1].utterance.text == "Stop!" and events[-1].utterance.same_breath


async def test_wake_pause_then_command():
    """Wake alone (KWS fires mid-word, transcript strips to nothing) -> keep listening ->
    the command after the pause is still the woke utterance."""
    stt = FakeSTT("hey mac")
    orig = stt.transcribe

    async def transcribe(pcm):
        # second call is the continuation (short audio)
        return "open mail" if len(stt.calls) >= 2 else await orig(pcm)

    stt.transcribe = transcribe  # type: ignore[method-assign]
    kinds, events, _ = await run(audio((1.0, 1.7), (3.0, 3.8)), wake_at=1.5, stt=stt)
    assert kinds == [K.WAKE, K.SPEECH_START, K.SPEECH_START, K.UTTERANCE]
    assert events[-1].utterance.woke and events[-1].utterance.text == "Open mail"
    assert not events[-1].utterance.same_breath


async def test_window_mode_and_expiry():
    kinds, events, _ = await run(audio((1.0, 2.0)), wake_at=None, stt=FakeSTT("yes"), before=lambda l: l.open_window(3.0, WindowReason.CONFIRM))
    assert kinds == [K.SPEECH_START, K.UTTERANCE]
    assert not events[-1].utterance.woke and events[-1].utterance.text == "yes"
    assert events[-1].utterance.window is WindowReason.CONFIRM
    kinds, events, _ = await run(audio(total_s=3.0), wake_at=None, before=lambda l: l.open_window(1.0, WindowReason.FOLLOWUP))
    assert kinds == [K.NO_SPEECH] and events[0].window is WindowReason.FOLLOWUP


async def test_window_suppressed_while_speaking_without_aec():
    def before(l):
        l.open_window(5.0, WindowReason.FOLLOWUP)
        l.set_speaking(True, "Here is the answer")

    kinds, _, _ = await run(audio((1.0, 2.0)), wake_at=None, before=before)
    assert kinds == [K.NO_SPEECH]


async def test_self_trigger_suppressed_when_spoken_text_mentions_mac():
    kinds, _, lis = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: l.set_speaking(True, "I'm Mac, your assistant"))
    assert kinds == [] and lis.metrics.counters["wakes_suppressed"] == 1
    # with AEC the same hit is accepted
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: (setattr(l, "_aec", True), l.set_speaking(True, "I'm Mac")))
    assert K.WAKE in kinds


async def test_muted_ignores_everything():
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, before=lambda l: l.set_muted(True))
    assert kinds == []


async def test_stt_missing_still_wakes():
    kinds, _, _ = await run(audio((1.0, 1.6), (1.7, 2.6)), wake_at=1.6, stt=FakeSTT(missing=True))
    assert kinds == [K.WAKE, K.SPEECH_START]


async def test_slow_kws_does_not_block_loop():
    """A 30 ms KWS step runs on kws_pool: loop time per frame stays < 20 ms and other tasks keep running."""
    import time

    class SlowWake(FakeWake):
        def accept(self, frame, sample_index):
            time.sleep(0.03)
            return super().accept(frame, sample_index)

    gaps = []

    async def ticker():
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6), total_s=3.0), trailing_silence_s=0.0)
    lis = ListenerImpl(src, SlowWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT(), Config())
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev.kind)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 30)
    t.cancel()
    assert events[:2] == [K.WAKE, K.SPEECH_START]
    assert lis.frame_ms_max < 20
    assert lis.kws_ms_max >= 30
    # The property is "the loop is not blocked by the KWS step" (30 ms), not a wall-clock
    # bound: under load the ticker itself jitters, so no gap may reach the step length.
    assert max(gaps) < 0.03, max(gaps)


async def test_max_utterance_cap():
    cfg = Config()
    cfg.listen.max_utterance_s = 2.0
    kinds, events, _ = await run(audio((1.0, 1.6), (1.7, 5.0)), wake_at=1.6, cfg=cfg, trailing=1.0)
    # the cap finalizes the first part AND the rest keeps being captured as a continuation
    assert kinds == [K.WAKE, K.SPEECH_START, K.UTTERANCE, K.UTTERANCE]
    first, rest = events[2].utterance, events[3].utterance
    assert first.woke and not first.continuation and first.audio_s < 4.0  # 1.6 s look-back + 2 s cap
    assert rest.continuation and not rest.woke and rest.audio_s > 1.0


async def test_stop_word_while_speaking():
    """While Mac talks, short VAD segments are transcribed; only bare stop words come out."""
    speaking = lambda l: l.set_speaking(True, "Here is a long answer about the weather")
    kinds, events, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("okay stop"), before=speaking)
    assert kinds == [K.UTTERANCE]
    u = events[0].utterance
    assert u.window is WindowReason.SPEAKING and not u.woke and u.text == "okay stop"
    # not a stop word -> dropped silently
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("what about tomorrow"), before=speaking)
    assert kinds == []
    # AEC off and Mac itself just said "stop" -> self-trigger, dropped
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("stop"), before=lambda l: l.set_speaking(True, "I will stop here"))
    assert kinds == []
    # long speech while speaking is capped at 2 s per segment and still only yields stop words
    kinds, _, lis = await run(audio((1.0, 4.0)), wake_at=None, stt=FakeSTT("blah blah"), before=speaking)
    assert kinds == [] and lis.metrics.counters.get("stop_words", 0) == 0
    # playback ended -> no more stop-word captures (window closed by set_speaking(False))
    kinds, _, _ = await run(audio((1.0, 1.5)), wake_at=None, stt=FakeSTT("stop"), before=lambda l: (l.set_speaking(True, "x"), l.set_speaking(False)))
    assert kinds == []


class EventSink:
    def __init__(self):
        self.events = []

    def emit(self, type_, **fields):
        self.events.append((type_, fields))
        return True


async def test_wake_instrumentation_events_and_counters():
    sink = EventSink()
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6)), trailing_silence_s=0.5)
    lis = ListenerImpl(src, FakeWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT(), Config(), eventlog=sink)
    kinds = []

    async def collect():
        async for ev in lis.events():
            kinds.append(ev.kind)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    types = [t for t, _ in sink.events]
    assert types[:2] == ["kws_hit", "wake_verify"]
    assert sink.events[0][1]["phrase"] == "hey mac" and sink.events[1][1] == {
        "source": "file", "accepted": True, "transcript": "hey mac what time is it", "reason": "phrase_found"}
    c = lis.metrics.counters
    assert c["kws_hits"] == 1 and c["wakes"] == 1 and c.get("false_wakes", 0) == 0
    assert lis.metrics.latency("kws_hit_to_wake")["n"] == 1
    # rejected + suppressed hits are logged with reasons
    sink = EventSink()
    src = FileInput(audio((1.0, 1.6), (1.7, 2.6)), trailing_silence_s=0.5)
    lis = ListenerImpl(src, FakeWake(int(1.6 * SAMPLE_RATE)), FakeVAD(), FakeSTT("big mac"), Config(), eventlog=sink)
    lis.set_speaking(True, "I'm Mac")
    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 10)
    assert [t for t, _ in sink.events] == ["kws_hit", "wake_suppressed"] and sink.events[1][1]["reason"] == "self_trigger"
    assert lis.metrics.counters["wakes_suppressed"] == 1


async def test_wake_right_after_mac_stops_speaking_and_during_followup():
    """The tail guard only blocks VAD captures; a wake word 0.5 s after playback ends is caught,
    and a wake hit inside an open FOLLOWUP window still takes the woke path."""
    def before(l):
        l.set_speaking(True, "Here is the answer")
        l.set_speaking(False)  # playback just ended at t=0: guard active for tail_guard_ms
        l.open_window(6.0, WindowReason.FOLLOWUP)

    kinds, events, lis = await run(audio((0.5, 1.1), (1.2, 2.1)), wake_at=1.1, before=before)
    # the follow-up window's VAD capture starts on "hey mac" itself (SPEECH_START), then the KWS
    # hit takes over: exactly one utterance, woke, not attributed to the window
    assert [k for k in kinds if k is not K.SPEECH_START] == [K.WAKE, K.UTTERANCE]
    assert events[-1].utterance.woke and events[-1].utterance.window is None
    assert lis._window_end is None  # the wake closed the follow-up window
