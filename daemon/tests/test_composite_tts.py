"""CompositeTTS routing/fallback and SpeechStack.set_voice persistence (fake engines)."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from macd import config
from macd.eventlog import Metrics
from macd.speech.factory import VOICE_RESET, VOICE_SET, SpeechStack
from macd.speech.tts import DEFAULT_FILLERS, CompositeTTS


class Unsuitable(ValueError):
    pass


class FakeKokoro:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prewarmed: list[str] = []
        self.closed = False

    async def synthesize(self, text):
        self.calls.append(text)
        return np.full(2400, 0.1, np.float32)

    async def prewarm(self, texts=DEFAULT_FILLERS):
        self.prewarmed += list(texts)

    def close(self):
        self.closed = True


class FakeClone:
    """Fake Pocket: 200 ms per word with 20 ms pauses between words; ``babble_on`` call indexes
    return 5 s; ``always_babble`` phrases never come out right; ``fail`` raises on every call."""

    def __init__(self, sample_rate=24_000, fail: Exception | None = None, babble_on=(), always_babble=()) -> None:
        self.sample_rate, self.fail = sample_rate, fail
        self.babble_on, self.always_babble = set(babble_on), set(always_babble)
        self.calls: list[str] = []
        self.reference: Path | None = None
        self.closed = False

    def synthesize_sync(self, text):
        i = len(self.calls)
        self.calls.append(text)
        if self.fail:
            raise self.fail
        if len(text.split()) < 2:
            raise Unsuitable(text)
        if i in self.babble_on or text in self.always_babble:
            return np.full(self.sample_rate * 5, 0.2, np.float32)
        word, gap = np.full(int(self.sample_rate * 0.2), 0.2, np.float32), np.zeros(int(self.sample_rate * 0.02), np.float32)
        parts = [word] * len(text.split())
        return np.concatenate([x for w in parts for x in (w, gap)][:-1])

    def set_reference(self, wav):
        self.reference = Path(wav)

    def close(self):
        self.closed = True


def fake_len(text: str, sr: int = 24_000) -> int:
    n = len(text.split())
    return int(sr * 0.2) * n + int(sr * 0.02) * (n - 1)


def make(clone=None, metrics=None):
    k = FakeKokoro()
    return CompositeTTS(k, clone, unsuitable=Unsuitable, metrics=metrics), k


async def test_clone_active_speaks_everything_no_kokoro():
    clone = FakeClone()
    tts, k = make(clone)
    assert tts.route("Okay.") == "clone" and tts.route("The capital of France is Paris.") == "clone"
    assert set(DEFAULT_FILLERS) <= set(tts._clone_cache)  # fillers pre-synthesised at activation
    for t in ("Okay.", "On it.", "The capital of France is Paris."):
        pcm = await tts.synthesize(t)
        assert len(pcm) > 0
    assert k.calls == []  # one voice: Kokoro never spoke
    assert "The capital of France is Paris." in clone.calls


async def test_no_clone_everything_kokoro():
    tts, k = make(None)
    assert tts.route("The capital of France is Paris.") == "kokoro"
    await tts.synthesize("The capital of France is Paris.")
    assert k.calls == ["The capital of France is Paris."]


def test_fillers_cached_from_good_attempt_after_babble():
    clone = FakeClone(babble_on={1})  # the 2nd generation ever babbles (5 s)
    tts, _ = make(clone)
    second = clone.calls[1]
    assert clone.calls.count(second) == 2  # retried once, then accepted
    assert all(0.15 <= len(v) / 24_000 <= 2.5 for v in tts._clone_cache.values())


def test_phrase_failing_three_times_uses_earcon(caplog):
    from macd.audio.earcons import earcon

    clone = FakeClone(always_babble={"One moment."})
    tts, _ = make(clone)
    assert clone.calls.count("One moment.") == 3
    assert np.array_equal(tts._clone_cache["One moment."], earcon("tick", 24_000)[0])
    assert any("filler_clone_failed 'One moment.'" in r.message for r in caplog.records)


async def test_unsuitable_runtime_input_uses_closest_filler_or_earcon():
    from macd.audio.earcons import earcon

    tts, k = make(FakeClone())
    assert np.array_equal(await tts.synthesize("Okay"), tts._clone_cache["Okay."])  # closest cached filler
    assert np.array_equal(await tts.synthesize("Zzz"), earcon("tick", 24_000)[0])
    assert k.calls == []


async def test_fallback_on_exception_counts_and_logs_loudly(caplog):
    m = Metrics()
    clone = FakeClone()
    tts, k = make(clone, m)
    clone.fail = RuntimeError("boom")  # engine breaks after activation
    for i in range(3):
        pcm = await tts.synthesize(f"This is sentence number {i} for the test.")
        assert len(pcm) == 2400
    assert m.counters["tts_fallbacks"] == 3 and len(k.calls) == 3
    assert sum("speaking it with Kokoro" in r.message for r in caplog.records) == 3


async def test_composite_rate_resamples_clone_to_kokoro_rate():
    tts, _ = make(FakeClone(sample_rate=16_000))
    assert tts.sample_rate == 24_000
    pcm = await tts.synthesize("The capital of France is Paris.")
    assert abs(len(pcm) - fake_len("The capital of France is Paris.")) <= 4
    for k, v in tts._clone_cache.items():  # multi-word fillers exact; one-word ones cut from the doubled form
        assert abs(len(v) - fake_len(k)) <= (4 if len(k.split()) > 1 else 24_000 * 0.03), (k, len(v))


def test_doubled_filler_hitting_the_cap_is_retried():
    clone = FakeClone(babble_on={0})  # first generation ("Okay. Okay.") babbles to the cap
    tts, _ = make(clone)
    assert clone.calls[:2] == ["Okay. Okay.", "Okay. Okay."]
    assert abs(len(tts._clone_cache["Okay."]) / 24_000 - 0.2) < 0.03


async def test_short_runtime_sentence_prefers_close_cached_filler():
    clone = FakeClone()
    tts, _ = make(clone)
    n = len(clone.calls)
    assert np.array_equal(await tts.synthesize("Okay!"), tts._clone_cache["Okay."])
    assert len(clone.calls) == n  # not sent to the engine
    await tts.synthesize("All set, then.")  # no close filler: the engine gets a try
    assert clone.calls[-1] == "All set, then."


def test_single_word_fillers_are_cut_from_doubled_form():
    clone = FakeClone()
    tts, _ = make(clone)
    assert "Okay. Okay." in clone.calls and "Okay." not in clone.calls
    assert abs(len(tts._clone_cache["Okay."]) / 24_000 - 0.2) < 0.03


async def test_prewarm_adds_phrases_to_clone_cache_and_reset_restores_kokoro():
    clone = FakeClone()
    tts, k = make(clone)
    await tts.prewarm(["Got it.", "Okay, stopped."])
    assert "Got it." in tts._clone_cache and "Okay, stopped." in tts._clone_cache
    assert k.prewarmed[:2] == ["Got it.", "Okay, stopped."]
    tts.deactivate_clone()
    assert clone.closed and tts.route("Okay.") == "kokoro" and not tts._clone_cache
    await tts.synthesize("Okay.")
    assert k.calls == ["Okay."]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_DIR", tmp_path / "voice")
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr("macd.speech.models.missing_models", lambda keys=None: [])
    return tmp_path


def make_stack(clone=None):
    tts, k = make(clone)
    made = []

    def make_clone(sample):
        made.append(sample)
        c = FakeClone()
        c.set_reference(sample)
        return c

    stack = SpeechStack(make_wake=lambda: None, make_vad=lambda: None, stt=None, tts=tts, make_clone=make_clone)  # type: ignore[arg-type]
    return stack, k, made


def test_set_voice_persists_and_switches(home):
    sample = home / "me.wav"
    sample.write_bytes(b"RIFF....WAVEfake")
    stack, _, made = make_stack()
    assert stack.voice_engine == "kokoro"
    assert stack.set_voice(str(sample)) == VOICE_SET
    assert stack.voice_engine == "clone" and made == [home / "voice" / "sample.wav"]
    assert (home / "voice" / "sample.wav").read_bytes() == sample.read_bytes()
    assert stack.clone_sample == str(home / "voice" / "sample.wav")
    state = json.loads((home / "state.json").read_text())
    assert state == {"voice_engine": "clone", "clone_sample": str(home / "voice" / "sample.wav")}
    assert set(DEFAULT_FILLERS) <= set(stack.tts._clone_cache)  # fillers cloned at activation
    # a second sample re-references the existing engine (and re-caches) instead of building a new one
    sample2 = home / "other.wav"
    sample2.write_bytes(b"RIFF....WAVEother")
    stack.set_voice(str(sample2))
    assert len(made) == 1 and stack.tts.clone.reference == home / "voice" / "sample.wav"
    assert set(DEFAULT_FILLERS) <= set(stack.tts._clone_cache)
    clone = stack.tts.clone
    assert stack.set_voice(None) == VOICE_RESET
    assert stack.voice_engine == "kokoro" and clone.closed and stack.clone_sample is None
    assert json.loads((home / "state.json").read_text())["voice_engine"] == "kokoro"


def test_set_voice_errors(home):
    stack, _, _ = make_stack()
    with pytest.raises(FileNotFoundError):
        stack.set_voice(str(home / "nope.wav"))
    from macd.speech import models

    models.missing_models = lambda keys=None: ["pocket"]  # restored by monkeypatch teardown
    (home / "me.wav").write_bytes(b"x")
    with pytest.raises(ValueError):
        stack.set_voice(str(home / "me.wav"))


async def test_sync_set_voice_is_thread_safe_enough_for_to_thread(home):
    (home / "me.wav").write_bytes(b"x")
    stack, _, _ = make_stack()
    assert await asyncio.to_thread(stack.set_voice, str(home / "me.wav")) == VOICE_SET
