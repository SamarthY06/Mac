from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from macd import config
from macd.settings import PREVIEW_TEXT, Settings, validate, wake_phrases_for


class FakeStack:
    voice_engine = "kokoro"
    clone_sample: str | None = None
    calls: ClassVar[list[tuple[str, Any]]] = []

    class tts:
        sample_rate = 24_000

    def set_wake_phrases(self, phrases: list[str]) -> None:
        if any("x" in p for p in phrases):
            raise ValueError("unusable phrase")
        self.calls.append(("wake", phrases))

    def set_kokoro_voice(self, voice: str, lang: str, speed: float) -> None:
        if voice == "nope":
            raise ValueError("unknown voice")
        self.calls.append(("kokoro", (voice, lang, speed)))

    def set_voice(self, path: str | None) -> str:
        self.voice_engine = "clone" if path else "kokoro"
        self.calls.append(("engine", path))
        return "ok"

    def list_voices(self) -> list[dict[str, Any]]:
        return [{"id": "af_heart", "name": "Heart", "lang": "en-us", "gender": "f"}]

    def preview(self, voice: str, text: str) -> np.ndarray:
        if voice == "nope":
            raise KeyError(voice)
        self.calls.append(("preview", (voice, text)))
        return np.zeros(240, dtype=np.float32)


class FakeBrain:
    names: ClassVar[list[str]] = []

    def set_name(self, name: str) -> None:
        self.names.append(name)


def make(cfg: config.Config | None = None) -> tuple[Settings, FakeStack, FakeBrain, list[str], list[tuple]]:
    FakeStack.calls = []
    FakeBrain.names = []
    stack, brain = FakeStack(), FakeBrain()
    said: list[str] = []
    played: list[tuple] = []
    async def prewarm() -> None:
        FakeStack.calls.append(("prewarm", None))

    s = Settings(cfg or config.Config(), stack=stack, brain=brain, speak=said.append,
                 play=lambda pcm, sr: played.append((len(pcm), sr)), prewarm=prewarm)
    return s, stack, brain, said, played


def test_get_shape_matches_set_keys() -> None:
    s, *_ = make()
    data = s.get()
    assert set(config.SETTABLE_KEYS) <= set(data)
    assert data["wake.phrases"] == ["hey mac", "ok mac"] and data["name"] == "Mac" and data["tts.engine"] == "kokoro"
    assert data["stt_languages"] == ["en"] and data["clone_sample"] is None and data["aliases"]["language"] == "tts.lang"


async def test_set_round_trip_saves_config_and_applies_live() -> None:
    s, stack, brain, said, _ = make()
    res = await s.set({"name": "Jarvis", "tts.speed": "1.2", "listen.silence_ms": 900, "language": "en-gb",
                       "brain.idle_shutdown_min": 30})
    assert res["errors"] == {}
    assert res["applied"] == {"name": "Jarvis", "tts.speed": 1.2, "listen.silence_ms": 900, "tts.lang": "en-gb",
                              "brain.idle_shutdown_min": 30, "wake.phrases": ["hey jarvis", "ok jarvis"]}
    assert brain.names == ["Jarvis"]
    assert ("wake", ["hey jarvis", "ok jarvis"]) in stack.calls
    assert ("kokoro", ("af_heart", "en-us", 1.2)) in stack.calls and ("kokoro", ("af_heart", "en-gb", 1.2)) in stack.calls
    assert ("prewarm", None) in stack.calls
    assert said == ["From now on, say Hey Jarvis."]
    assert s.cfg.name == "Jarvis" and s.cfg.listen.silence_ms == 900 and s.cfg.brain.idle_shutdown_min == 30
    reloaded = config.load()  # from the per-test CONFIG_PATH
    assert reloaded.name == "Jarvis" and reloaded.wake.phrases == ["hey jarvis", "ok jarvis"]
    assert reloaded.tts.speed == 1.2 and reloaded.tts.lang == "en-gb" and reloaded.listen.silence_ms == 900


async def test_explicit_wake_phrases_win_over_derived() -> None:
    s, _stack, _, said, _ = make()
    res = await s.set({"name": "Friday", "wake_phrases": "hey friday,yo friday"})
    assert res["applied"]["wake.phrases"] == ["hey friday", "yo friday"] and res["errors"] == {}
    assert said == ["From now on, say Hey Friday."]


@pytest.mark.parametrize(
    ("changes", "bad_key", "msg"),
    [({"name": ""}, "name", "1 to 32"), ({"name": "x" * 40}, "name", "1 to 32"), ({"tts.speed": 5}, "tts.speed", "between"),
     ({"listen.silence_ms": 50}, "listen.silence_ms", "between"), ({"tts.engine": "robot"}, "tts.engine", "kokoro"),
     ({"wake.phrases": []}, "wake.phrases", "at least one"), ({"wake.phrases": "hey this is way too long"}, "wake.phrases", "two to four"),
     ({"wake.phrases": ["jarvis"]}, "wake.phrases", "two to four"),
     ({"stt.threads": 2}, "stt.threads", "not allowed"), ({"brain.idle_shutdown_min": "soon"}, "brain.idle_shutdown_min", "invalid literal"),
     ({"tts.voice": "nope"}, "tts.voice", "unknown voice"), ({"tts.engine": "clone"}, "tts.engine", "no cloned voice")],
)
async def test_validation_errors(changes: dict, bad_key: str, msg: str) -> None:
    s, _stack, _, said, _ = make()
    before = s.cfg.name
    res = await s.set(changes)
    assert bad_key in res["errors"] and msg in res["errors"][bad_key] and res["applied"] == {}
    assert s.cfg.name == before and said == [] and not config.CONFIG_PATH.exists()


async def test_partial_failure_applies_the_good_keys() -> None:
    s, _stack, _, _, _ = make()
    res = await s.set({"tts.speed": 1.5, "wake.phrases": ["hey xeno"]})
    assert res["applied"] == {"tts.speed": 1.5} and "wake.phrases" in res["errors"]
    assert config.load().tts.speed == 1.5 and s.cfg.wake.phrases == ["hey mac", "ok mac"]


async def test_engine_switch_uses_set_voice() -> None:
    s, stack, _, _, _ = make()
    stack.clone_sample = "/tmp/me.wav"
    assert (await s.set({"engine": "clone"}))["applied"] == {"tts.engine": "clone"}
    assert ("engine", "/tmp/me.wav") in stack.calls and s.get()["tts.engine"] == "clone"
    assert (await s.set({"tts.engine": "kokoro"}))["applied"] == {"tts.engine": "kokoro"}
    assert ("engine", None) in stack.calls


async def test_bad_changes_argument() -> None:
    s, *_ = make()
    with pytest.raises(ValueError):
        await s.set({})
    with pytest.raises(ValueError):
        await s.set("name=Jarvis")  # type: ignore[arg-type]


async def test_voices_and_preview() -> None:
    s, stack, _, _, played = make()
    assert await s.voices() == [{"id": "af_heart", "name": "Heart", "lang": "en-us", "gender": "f"}]
    res = await s.preview("af_heart")
    assert res == {"queued": True, "voice": "af_heart", "text": PREVIEW_TEXT} and played == [(240, 24_000)]
    assert ("preview", ("af_heart", PREVIEW_TEXT)) in stack.calls
    assert (await s.preview("af_heart", "Custom line."))["text"] == "Custom line."
    with pytest.raises(ValueError, match="cannot preview"):
        await s.preview("nope")
    with pytest.raises(ValueError):
        await s.preview("")
    s.stack = None
    with pytest.raises(ValueError, match="unavailable"):
        await s.voices()


def test_validate_and_helpers() -> None:
    validate("name", "Mac Mini")
    with pytest.raises(ValueError):
        validate("name", "Mac!")
    assert wake_phrases_for(" Jarvis ") == ["hey jarvis", "ok jarvis"]
