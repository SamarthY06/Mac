"""Model-backed speech tests (real KWS / VAD / Parakeet / Kokoro). Skipped if models missing.

Audio is fed non-realtime through FileInput; the listener's endpointing runs in sample
time, so results don't depend on machine speed.
"""

from __future__ import annotations

import asyncio
import re
import time

import numpy as np
import pytest

from macd.audio.file_backend import FileInput
from macd.audio.resample import resample
from macd.config import Config
from macd.contracts import SAMPLE_RATE, ListenerEventKind, WindowReason
from macd.speech import models

BASE = ["kws", "vad", "kokoro", "voices"]


def needs(*keys: str):
    missing = models.missing_models(list(keys))
    return pytest.mark.skipif(bool(missing), reason=f"speech models missing: {missing}")


pytestmark = [pytest.mark.models, needs(*BASE)]
needs_stt = needs("parakeet")


@pytest.fixture(scope="module")
def stack():
    from macd.speech.factory import build_speech

    try:
        st = build_speech(Config())
    except Exception as e:  # noqa: BLE001 - a partially downloaded model file must skip, not error
        pytest.skip(f"speech models not loadable (download in progress?): {e}")
    yield st
    st.close()


def tts16(stack, text: str) -> np.ndarray:
    pcm = stack.tts.kokoro.synthesize_sync(text)
    return resample(pcm, stack.tts.sample_rate, SAMPLE_RATE)


def padded(speech: np.ndarray, lead_s: float = 1.0, tail_s: float = 1.0, noise: float = 0.003, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(int(lead_s * SAMPLE_RATE)), speech, np.zeros(int(tail_s * SAMPLE_RATE))])
    return (x + rng.normal(0, noise, x.shape)).astype(np.float32)


async def run_listener(stack, pcm: np.ndarray, cfg: Config | None = None, window: tuple[float, WindowReason] | None = None,
                       trailing_silence_s: float = 1.5, realtime: bool = False):
    from macd.speech.factory import make_listener

    cfg = cfg or Config()
    src = FileInput(pcm, realtime=realtime, trailing_silence_s=trailing_silence_s)
    lis = make_listener(src, stack, cfg)
    if window:
        lis.open_window(*window)
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    t0 = time.perf_counter()
    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 120)
    return events, lis, time.perf_counter() - t0


def kinds(events, *, all_kinds: bool = False):
    return [e.kind for e in events if all_kinds or e.kind is not ListenerEventKind.SPEECH_START]


@needs_stt
async def test_wake_then_utterance_one_breath(stack):
    speech = tts16(stack, "Hey Mac, what is two plus two?")
    events, lis, wall = await run_listener(stack, padded(speech))
    ks = kinds(events)
    assert ListenerEventKind.WAKE in ks, ks
    assert ListenerEventKind.UTTERANCE in ks, ks
    assert ks.index(ListenerEventKind.WAKE) < ks.index(ListenerEventKind.UTTERANCE)
    utt = next(e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE)
    assert utt.woke
    norm = utt.text.lower()
    assert re.search(r"(two|2) (plus|\+) (two|2)|2 ?\+ ?2", norm), utt.text
    assert "mac" not in norm, utt.text
    assert utt.same_breath and utt.window is None
    wake = next(e.wake for e in events if e.kind is ListenerEventKind.WAKE)
    utt_ev = next(e for e in events if e.kind is ListenerEventKind.UTTERANCE)
    speech_end = SAMPLE_RATE + len(speech)
    lat = lis.metrics.latency
    print(
        f"\n[latency] KWS hit at {wake.sample_index / SAMPLE_RATE:.2f}s (speech {1.0:.2f}-{speech_end / SAMPLE_RATE:.2f}s); "
        f"hit->WAKE (verify STT) {lat('kws_hit_to_wake')['last']:.0f} ms; verify STT {lat('wake_verify_stt')['last']:.0f} ms; "
        f"endpoint->transcript {(utt_ev.t - utt.ended_at) * 1000:.0f} ms; STT {lat('listener_stt')['last']:.0f} ms for {utt.audio_s:.2f}s; "
        f"loop time/frame p50 {lat('listener_frame')['p50']:.2f} ms max {lis.frame_ms_max:.1f} ms; KWS max {lis.kws_ms_max:.1f} ms; "
        f"wall {wall:.2f}s; text={utt.text!r}"
    )
    assert lis.frame_ms_max < 20


@pytest.mark.parametrize(
    "text",
    ["I ate a big mac yesterday.", "My macbook pro is slow.", "The weather is nice today.",
     "Hey Mark, can you order pizza?", "Hey Matt, are you coming?", "Hi Mike, how are you?"],
)
async def test_negatives_no_wake(stack, text):
    """Runs even without Parakeet: then KWS hits are accepted unverified, i.e. stricter.
    Person names are rejected with accept_similar_names off (the strict N6 policy)."""
    cfg = Config()
    cfg.wake.accept_similar_names = False
    events, lis, _ = await run_listener(stack, padded(tts16(stack, text)), cfg=cfg)
    ks = kinds(events)
    assert ListenerEventKind.WAKE not in ks, (text, ks, dict(lis.metrics.counters))
    assert ListenerEventKind.UTTERANCE not in ks


@needs_stt
async def test_wake_latency_realtime(stack):
    """Wall-clock wake latency with real-time-paced audio (what the daemon sees): budget 250 ms."""
    events, lis, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac, what is two plus two?")), realtime=True, trailing_silence_s=1.0)
    assert ListenerEventKind.UTTERANCE in kinds(events)
    lat = lis.metrics.latency
    hit_to_wake, verify = lat("kws_hit_to_wake")["last"], lat("wake_verify_stt")["last"]
    print(f"\n[realtime] kws_hit_to_wake {hit_to_wake:.0f} ms (verify STT {verify:.0f} ms); "
          f"loop/frame p95 {lat('listener_frame')['p95']} ms; kws p95 {lat('listener_kws')['p95']} ms")
    # Pipeline overhead beyond the STT itself: 100 ms audio tail + batch harvest + hops.
    # (The absolute budget of 250 ms holds on an unloaded machine; STT time varies with load.)
    assert hit_to_wake - verify < 160, (hit_to_wake, verify)
    if verify < 120:
        assert hit_to_wake < 250


@needs_stt
@pytest.mark.parametrize("voice", ["hf_alpha", "hm_psi", "hf_beta"])
async def test_accented_wake(stack, voice):
    """Hindi-accented English (Kokoro hf/hm voices, lang en-us) must wake with the defaults:
    Parakeet hears "Hey Mark" for these, so stage 2 relies on accept_similar_names."""
    k = stack.tts.kokoro
    saved = k.voice
    k.voice = voice
    try:
        speech = tts16(stack, "Hey Mac, what time is it?")
    finally:
        k.voice = saved
    events, lis, _ = await run_listener(stack, padded(speech))
    ks = kinds(events)
    assert ListenerEventKind.WAKE in ks, (voice, ks, dict(lis.metrics.counters))
    utt = next(e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE)
    assert utt.woke and "time" in utt.text.lower(), utt.text


@needs_stt
async def test_wake_then_stop_word(stack):
    """'Hey Mac, stop.' may be transcribed fused ('Hey Macstop!'); the text must still be 'stop'."""
    events, _, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac, stop.")))
    utts = [e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE]
    assert ListenerEventKind.WAKE in kinds(events)
    assert len(utts) == 1 and utts[0].woke and re.sub(r"[^a-z]", "", utts[0].text.lower()) == "stop", utts


@needs_stt
async def test_window_mode_without_wake_word(stack):
    events, _, _ = await run_listener(stack, padded(tts16(stack, "Yes, please send it.")), window=(6.0, WindowReason.CONFIRM))
    utts = [e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE]
    assert len(utts) == 1 and not utts[0].woke
    assert "send" in utts[0].text.lower()
    assert ListenerEventKind.WAKE not in kinds(events)


async def test_window_expires_no_speech(stack):
    events, _, _ = await run_listener(stack, np.zeros(SAMPLE_RATE * 2, np.float32), window=(1.0, WindowReason.FOLLOWUP))
    assert kinds(events) == [ListenerEventKind.NO_SPEECH]


@needs_stt
async def test_wake_alone_then_no_speech(stack):
    cfg = Config()
    cfg.listen.no_speech_timeout_s = 3.0
    events, _, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac.")), cfg=cfg, trailing_silence_s=4.0)
    assert kinds(events) == [ListenerEventKind.WAKE, ListenerEventKind.NO_SPEECH]


@needs_stt
async def test_wake_pause_then_command(stack):
    """'Hey Mac' ... 1.2 s pause ... command: still captured as the woke utterance."""
    a = tts16(stack, "Hey Mac.")
    b = tts16(stack, "Open my calendar for tomorrow.")
    pcm = padded(np.concatenate([a, np.zeros(int(1.2 * SAMPLE_RATE), np.float32), b]))
    events, _, _ = await run_listener(stack, pcm)
    utts = [e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE]
    assert ListenerEventKind.WAKE in kinds(events)
    assert len(utts) == 1 and utts[0].woke and "calendar" in utts[0].text.lower(), kinds(events)


def _wer(ref: str, hyp: str) -> float:
    r = re.sub(r"[^a-z0-9 ]", "", ref.lower()).split()
    h = re.sub(r"[^a-z0-9 ]", "", hyp.lower()).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


@needs_stt
async def test_stt_accuracy_and_tts_rtf(stack):
    ref = "The quick brown fox jumps over the lazy dog near the river bank."
    t0 = time.perf_counter()
    pcm24 = stack.tts.kokoro.synthesize_sync(ref)
    tts_s = time.perf_counter() - t0
    dur = len(pcm24) / stack.tts.sample_rate
    pcm = padded(resample(pcm24, stack.tts.sample_rate, SAMPLE_RATE), 0.3, 0.3)
    t0 = time.perf_counter()
    hyp = await stack.stt.transcribe(pcm)
    stt_s = time.perf_counter() - t0
    wer = _wer(ref, hyp)
    print(f"\n[stt] {hyp!r} WER={wer:.2f} STT {stt_s * 1000:.0f} ms for {len(pcm) / SAMPLE_RATE:.1f}s audio; "
          f"TTS {tts_s * 1000:.0f} ms for {dur:.1f}s audio (RTF {tts_s / dur:.2f})")
    assert wer <= 0.2, hyp


async def test_tts_fillers_and_empty(stack):
    await stack.tts.prewarm(["Okay.", "On it."])
    cached = await stack.tts.synthesize("Okay.")
    assert len(cached) > 1000 and (await stack.tts.synthesize("Okay.")) is cached  # served from the filler cache
    assert (await stack.tts.synthesize("")).size == 0
    assert (await stack.tts.synthesize("  ...  ")).size == 0


async def test_vad_probabilities(stack):
    vad = stack.make_vad()
    speech = tts16(stack, "This is a test of voice activity detection.")
    probs = [vad.accept(speech[i : i + 320]) for i in range(0, len(speech) - 320, 320)]
    vad.reset()
    sil = [vad.accept(np.zeros(320, np.float32)) for _ in range(50)]
    assert max(probs) > 0.8 and np.mean(np.array(probs) > 0.5) > 0.5
    assert max(sil) < 0.2


async def test_kws_hit_without_verification(stack):
    """KWS alone (verify=False) fires on the wake phrase; no Parakeet needed."""
    cfg = Config()
    cfg.wake.verify = False
    events, lis, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac, what is two plus two?")), cfg=cfg)
    assert ListenerEventKind.WAKE in kinds(events), dict(lis.metrics.counters)
    assert lis.metrics.counters["kws_hits"] >= 1


async def test_listener_loop_time_while_tts_runs(stack):
    """Daemon scenario: Kokoro synthesising on tts_pool while the listener handles frames.
    The GIL switch interval set by build_speech keeps loop time per frame < 20 ms."""
    import threading

    stop = threading.Event()

    def tts_hog() -> None:
        while not stop.is_set():
            stack.tts.kokoro.synthesize_sync("I'm checking that for you right now, one moment please.")

    th = threading.Thread(target=tts_hog)
    th.start()
    try:
        events, lis, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac, what is two plus two?")))
    finally:
        stop.set()
        th.join()
    print(f"\n[loop] while TTS runs: loop time/frame p50 {lis.metrics.latency('listener_frame')['p50']:.2f} ms max {lis.frame_ms_max:.1f} ms; KWS max {lis.kws_ms_max:.1f} ms")
    assert ListenerEventKind.UTTERANCE in kinds(events)
    assert lis.frame_ms_max < 20


@needs_stt
async def test_stop_word_over_playback(stack):
    """UX#1: user says "stop" while Mac is talking (AEC on): UTTERANCE with window=SPEAKING."""
    from macd.audio.file_backend import FileInput
    from macd.speech.factory import make_listener

    stop = tts16(stack, "Stop.")
    pcm = padded(stop, lead_s=1.0, tail_s=1.5)
    src = FileInput(pcm, realtime=False, trailing_silence_s=0.5)
    lis = make_listener(src, stack, Config(), aec=True)
    lis.set_speaking(True, "Here is what I found about the weather in Paris today.")
    events = []

    async def collect():
        async for ev in lis.events():
            events.append(ev)

    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 60)
    utts = [e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE]
    assert len(utts) == 1 and utts[0].window is WindowReason.SPEAKING and not utts[0].woke, [e.kind for e in events]
    assert "stop" in utts[0].text.lower()
    # a non-stop sentence over playback is dropped
    src = FileInput(padded(tts16(stack, "What about tomorrow?"), 1.0, 1.5), realtime=False, trailing_silence_s=0.5)
    lis = make_listener(src, stack, Config(), aec=True)
    lis.set_speaking(True, "Here is what I found.")
    events = []
    await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 60)
    assert not [e for e in events if e.kind is ListenerEventKind.UTTERANCE]


@needs_stt
async def test_settings_switch_wake_phrase_and_voice_live(stack, tmp_path, monkeypatch):
    """Settings window path: a live listener follows set_wake_phrases(["hey jarvis"]); Kokoro
    voice switches and previews without touching the active settings."""
    import asyncio

    from macd import config as cfgmod
    from macd.audio.file_backend import FileInput
    from macd.speech.factory import make_listener

    monkeypatch.setattr(cfgmod, "STATE_PATH", tmp_path / "state.json")  # never the real daemon state
    cfg = Config()
    clip = padded(tts16(stack, "Hey Jarvis, what time is it?"))
    src = FileInput(clip, realtime=False, trailing_silence_s=1.5)
    lis = make_listener(src, stack, cfg)  # created BEFORE the phrase change: must be swapped live
    try:
        await asyncio.to_thread(stack.set_wake_phrases, ["hey jarvis"])
        assert stack.voice_settings()["wake_phrases"] == ["hey jarvis"]
        events = []

        async def collect():
            async for ev in lis.events():
                events.append(ev)

        await asyncio.wait_for(asyncio.gather(lis.run(), collect()), 120)
        ks = kinds(events)
        assert ListenerEventKind.WAKE in ks, ks
        utt = next(e.utterance for e in events if e.kind is ListenerEventKind.UTTERANCE)
        assert utt.woke and "time" in utt.text.lower() and "jarvis" not in utt.text.lower(), utt.text
        # the old phrase no longer wakes
        events2, _, _ = await run_listener(stack, padded(tts16(stack, "Hey Mac, what time is it?")), cfg=cfg)
        assert ListenerEventKind.WAKE not in kinds(events2)
        # voice catalogue / preview / live switch
        ids = {v["id"] for v in stack.list_voices()}
        assert {"af_heart", "am_adam", "clone"} <= ids and len(ids) > 50
        before = stack.voice_settings()
        sample = await asyncio.to_thread(stack.preview, "am_adam")
        assert sample.dtype == np.float32 and 0.5 < len(sample) / stack.tts.sample_rate < 6
        assert stack.voice_settings() == before
        await asyncio.to_thread(stack.set_kokoro_voice, "am_adam")
        assert stack.tts.kokoro.voice == "am_adam" and (await stack.tts.synthesize("Okay.")).size > 0
    finally:
        await asyncio.to_thread(stack.set_wake_phrases, ["hey mac", "ok mac"])
        await asyncio.to_thread(stack.set_kokoro_voice, "af_heart")
