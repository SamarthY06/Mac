"""PocketTTS (Kyutai Pocket TTS voice cloning via sherpa-onnx).

Pure tests (text gating, runaway cap, reference conditioning) always run. Model tests clone a
Kokoro-synthesised reference voice (no real person) and are skipped unless the Pocket, Kokoro
and Parakeet models are installed under config.MODELS_DIR.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np
import pytest

from macd import config
from macd.audio.resample import float_to_int16, resample
from macd.contracts import SAMPLE_RATE
from macd.speech import models
from macd.speech.pocket_tts import (
    MODEL_FILES,
    PocketTTS,
    UnsuitableText,
    cap_audio,
    load_reference,
    split_chunks,
)

POCKET_DIR = config.MODELS_DIR / "kyutai-pocket-tts-int8"
SENTENCE_10 = "I opened your calendar and you have two meetings tomorrow morning."
REF_TEXT = (
    "The quick brown fox jumps over the lazy dog. I like to walk by the river in the evening, "
    "when the light is soft and the city is quiet. Tomorrow we could try the new bakery on the corner."
)


def _pocket_present() -> bool:
    return all((POCKET_DIR / f).is_file() for f in MODEL_FILES)


needs_models = pytest.mark.skipif(
    not _pocket_present() or bool(models.missing_models(["kokoro", "voices", "parakeet"])),
    reason=f"pocket model missing in {POCKET_DIR} or kokoro/voices/parakeet not installed",
)


def write_wav(path: Path, pcm: np.ndarray, sr: int) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(float_to_int16(pcm).tobytes())
    return path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_recall(expected: str, got: str) -> float:
    exp, hyp = words(expected), words(got)
    hit = 0
    for w in exp:
        if w in hyp:
            hyp.remove(w)
            hit += 1
    return hit / max(len(exp), 1)


def median_f0_hz(pcm: np.ndarray, sr: int, lo: float = 60.0, hi: float = 400.0) -> float:
    """Median fundamental frequency over voiced 40 ms frames (normalised autocorrelation peak)."""
    n = int(0.04 * sr)
    lag_lo, lag_hi = int(sr / hi), int(sr / lo)
    f0s = []
    for start in range(0, pcm.shape[0] - n, n // 2):
        x = pcm[start : start + n] - pcm[start : start + n].mean()
        e = float((x * x).sum())
        if e < 1e-3:
            continue
        ac = np.correlate(x, x, mode="full")[n - 1 :] / e
        lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
        if ac[lag] > 0.6:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


# ---------------------------------------------------------------- pure tests (no models)


@pytest.mark.parametrize("text", ["", "   ", "...", "Okay.", "Okay", "Sure!", " — "])
def test_unsuitable_text_rejected(text: str) -> None:
    with pytest.raises(UnsuitableText):
        PocketTTS.check_text(text)


@pytest.mark.parametrize("text", ["On it.", "Okay, sure.", "I opened your calendar."])
def test_suitable_text_accepted(text: str) -> None:
    assert PocketTTS.check_text(f"  {text}  ") == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("On it, give me a second.", ["On it, give me a second."]),
        (SENTENCE_10, ["I opened your calendar", "and you have two meetings tomorrow morning."]),
        ("Okay. I opened your calendar.", ["Okay. I opened your calendar."]),
        (
            "The train to the city leaves at half past seven every morning, so we should leave the house by six.",
            ["The train to the city leaves at half past seven every morning,", "so we should leave the house by six."],
        ),
        (
            "The quick brown fox jumps over the lazy dog, and then it went home to sleep for the rest of the afternoon.",
            ["The quick brown fox jumps over the lazy dog,", "and then it went home to sleep for the rest of the afternoon."],
        ),
        (" ".join(f"w{i}" for i in range(30)), [" ".join(f"w{i}" for i in range(12)), " ".join(f"w{i}" for i in range(12, 28)), "w28 w29"]),
        ("One two three four five six seven.", ["One two three four five six seven."]),
        (
            "Sure, I can do that. Opening Mail now, and then I will draft the reply.",
            ["Sure, I can do that.", "Opening Mail now,", "and then I will draft the reply."],
        ),
    ],
)
def test_split_chunks(text: str, expected: list[str]) -> None:
    chunks = split_chunks(text)
    assert chunks == expected
    assert " ".join(chunks) == text
    assert all(len(c.split()) >= 2 for c in chunks[1:]) or len(chunks) == 1


def test_cap_audio_truncates_at_zero_crossing_with_fade() -> None:
    sr = 24_000
    t = np.arange(30 * sr) / sr
    x = (0.8 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out, truncated = cap_audio(x, sr, 2.0)
    assert truncated
    assert 1.95 * sr <= out.shape[0] <= 2.0 * sr
    assert abs(out[-1]) < 1e-3  # faded to (near) silence
    same, truncated = cap_audio(x[: sr // 2], sr, 2.0)
    assert not truncated and same.shape[0] == sr // 2


def test_load_reference_trims_silence_and_normalises(tmp_path: Path) -> None:
    sr = 16_000
    tone = 0.2 * np.sin(2 * np.pi * 180 * np.arange(3 * sr) / sr)
    pcm = np.concatenate([np.zeros(sr), tone, np.zeros(2 * sr)]).astype(np.float32)
    ref = load_reference(write_wav(tmp_path / "ref.wav", pcm, sr))
    assert ref.dtype == np.float32
    assert abs(ref.shape[0] / 24_000 - 3.0) < 0.1  # silence stripped, resampled to 24 kHz
    assert abs(float(np.abs(ref).max()) - 0.9) < 1e-3


def test_load_reference_rejects_silence_and_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference(tmp_path / "nope.wav")
    with pytest.raises(ValueError):
        load_reference(write_wav(tmp_path / "short.wav", np.zeros(2400, np.float32), 24_000))


# ---------------------------------------------------------------- model tests


@pytest.fixture(scope="module")
def kokoro():
    from macd.speech.tts import KokoroTTS

    k = KokoroTTS(models.path("kokoro"), models.path("voices"))
    yield k
    k.close()


@pytest.fixture(scope="module")
def ref_wavs(kokoro, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("refs")
    out = {}
    for name, voice, lang in (("george", "bm_george", "en-gb"), ("heart", "af_heart", "en-us")):
        kokoro.voice, kokoro.lang = voice, lang
        pcm = kokoro.synthesize_sync(REF_TEXT)
        assert pcm.shape[0] / kokoro.sample_rate > 8
        out[name] = write_wav(d / f"{name}.wav", pcm, kokoro.sample_rate)
    return out


@pytest.fixture(scope="module")
def pocket(ref_wavs):
    p = PocketTTS(POCKET_DIR, ref_wavs["george"], threads=2)
    p.warmup_sync()
    yield p
    p.close()


@pytest.fixture(scope="module")
def stt():
    from macd.speech.stt import ParakeetSTT

    s = ParakeetSTT(models.path("parakeet"))
    yield s
    s.close()


@pytest.mark.models
@needs_models
def test_clone_is_intelligible_and_fast(pocket: PocketTTS, stt) -> None:
    # Timing: best of two runs (the suite shares the CPU with other work). Intelligibility: the
    # model samples, so a mispronounced word happens; one re-synthesis is allowed.
    texts, ttfas, rtfs = [], [], []
    for _ in range(2):
        pcm = pocket.synthesize_sync(SENTENCE_10)
        assert pcm.dtype == np.float32 and pcm.ndim == 1
        assert 1.5 < pcm.shape[0] / pocket.sample_rate < 8
        assert not pocket.last_runaway
        ttfas.append(pocket.last_ttfa_ms)
        rtfs.append(pocket.last_rtf)
        texts.append(stt.transcribe_sync(resample(pcm, pocket.sample_rate, SAMPLE_RATE)))
        if word_recall(SENTENCE_10, texts[-1]) >= 0.8 and ttfas[-1] < 400:
            break
    assert any(word_recall(SENTENCE_10, t) >= 0.8 for t in texts), texts
    assert min(rtfs) < 0.5, rtfs
    assert min(ttfas) < 400, ttfas


@pytest.mark.models
@needs_models
async def test_async_synthesize_rejects_one_word(pocket: PocketTTS) -> None:
    with pytest.raises(UnsuitableText):
        await pocket.synthesize("Okay.")
    pcm = await pocket.synthesize("On it, give me a second.")
    assert pcm.shape[0] / pocket.sample_rate <= pocket.cap_seconds("On it, give me a second.")


@pytest.mark.models
@needs_models
async def test_synthesize_stream_first_chunk_is_fast(pocket: PocketTTS, stt) -> None:
    import time

    firsts = []
    for _ in range(2):  # best of two: timing on a shared CPU
        t0 = time.perf_counter()
        chunks, arrival = [], []
        async for pcm in pocket.synthesize_stream(SENTENCE_10):
            chunks.append(pcm)
            arrival.append((time.perf_counter() - t0) * 1000)
        assert len(chunks) == 2
        assert pocket.last_ttfa_ms is not None and pocket.last_ttfa_ms <= arrival[0]
        firsts.append(arrival[0])
        if arrival[0] < 400:
            break
    assert min(firsts) < 400, firsts
    text = stt.transcribe_sync(resample(np.concatenate(chunks), pocket.sample_rate, SAMPLE_RATE))
    assert word_recall(SENTENCE_10, text) >= 0.6, text  # looser than the non-stream test: no retry here


@pytest.mark.models
@needs_models
def test_runaway_is_capped(pocket: PocketTTS) -> None:
    """Two-word inputs sometimes babble for tens of seconds; the cap must always hold."""
    text = "Okay okay."
    cap = pocket.cap_seconds(text)
    assert cap == pytest.approx(2.0)
    for _ in range(3):
        pcm = pocket.synthesize_sync(text)
        assert pcm.shape[0] <= cap * pocket.sample_rate
    # Explicit max_seconds also applies.
    assert pocket.cap_seconds(" ".join(["word"] * 200)) == pocket.max_seconds


@pytest.mark.models
@needs_models
def test_set_reference_switches_voice(pocket: PocketTTS, ref_wavs) -> None:
    sentence = "The train to the city leaves at half past seven every morning."
    low = pocket.synthesize_sync(sentence)
    id_george = pocket.reference_id
    pocket.set_reference(ref_wavs["heart"])
    try:
        assert pocket.reference_id != id_george
        assert 8 < pocket.reference_seconds <= 15
        high = pocket.synthesize_sync(sentence)
    finally:
        pocket.set_reference(ref_wavs["george"])
    f_low, f_high = median_f0_hz(low, pocket.sample_rate), median_f0_hz(high, pocket.sample_rate)
    # The female clone is pitched above the male one. (Pitch transfer is only partial: measured
    # refs ~136 / ~197 Hz gave clones ~130 / ~155-170 Hz, so the margin is deliberately modest.)
    assert f_high > f_low * 1.1, (f_low, f_high)


@pytest.mark.models
@needs_models
def test_requires_reference() -> None:
    p = PocketTTS(POCKET_DIR)
    try:
        assert not p.has_reference
        with pytest.raises(RuntimeError):
            p.synthesize_sync("Hello there, friend.")
    finally:
        p.close()
