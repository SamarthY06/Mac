"""KokoroTTS: kokoro-onnx v1.0 in its own single-worker executor ("tts_pool")."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from macd.eventlog import Metrics

log = logging.getLogger(__name__)

KOKORO_SR = 24_000
DEFAULT_FILLERS = ("Okay.", "On it.", "Let me check.", "Sure.", "One moment.")

# Kokoro voice ids are "<language><gender>_<name>"; all nine espeak codes verified with kokoro-onnx 0.6.
LANGUAGES: dict[str, tuple[str, str]] = {  # prefix -> (display, kokoro lang code)
    "a": ("English (US)", "en-us"),
    "b": ("English (UK)", "en-gb"),
    "e": ("Spanish", "es"),
    "f": ("French", "fr-fr"),
    "h": ("Hindi", "hi"),
    "i": ("Italian", "it"),
    "j": ("Japanese", "ja"),
    "p": ("Portuguese (Brazil)", "pt-br"),
    "z": ("Chinese (Mandarin)", "cmn"),
}
PREVIEW_TEXT: dict[str, str] = {
    "en-us": "Hi, I'm {name}. This is how I sound.",
    "en-gb": "Hi, I'm {name}. This is how I sound.",
    "es": "Hola, soy {name}. Así es como sueno.",
    "fr-fr": "Bonjour, je suis {name}. Voici ma voix.",
    "hi": "नमस्ते, मैं {name} हूँ। मेरी आवाज़ ऐसी है।",
    "it": "Ciao, sono {name}. Questa è la mia voce.",
    "ja": "こんにちは、{name}です。これが私の声です。",
    "pt-br": "Olá, eu sou {name}. É assim que eu soo.",
    "cmn": "你好，我是{name}。这是我的声音。",
}


def describe_voice(voice_id: str) -> dict[str, Any]:
    """{id, name, lang, language, gender} for a Kokoro voice id such as "af_heart"."""
    prefix, _, name = voice_id.partition("_")
    language, lang = LANGUAGES.get(prefix[:1], ("Unknown", "en-us"))
    gender = {"f": "f", "m": "m"}.get(prefix[1:2], "")
    return {"id": voice_id, "name": name.capitalize() or voice_id, "lang": lang, "language": language, "gender": gender}


class KokoroTTS:
    """contracts.TTS. Synthesis calls are serialized on the tts_pool thread; ``prewarm``ed
    phrases are served from an in-memory cache (filled on the loop thread only)."""

    sample_rate: int = KOKORO_SR

    def __init__(self, model_path: Path, voices_path: Path, voice: str = "af_heart", speed: float = 1.0, lang: str = "en-us", threads: int = 4) -> None:
        self.model_path, self.voices_path = Path(model_path), Path(voices_path)
        self.voice, self.speed, self.lang, self.threads = voice, speed, lang, threads
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_pool")
        self._cache: dict[str, np.ndarray] = {}
        self._kokoro = self._load()

    def _load(self) -> Any:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        logging.getLogger("phonemizer").setLevel(logging.ERROR)  # "words count mismatch" noise on short texts
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        kokoro = Kokoro.from_session(sess, str(self.voices_path))
        voices = kokoro.get_voices()
        if self.voice not in voices:
            log.warning("voice %r not found; using %r", self.voice, voices[0])
            self.voice = voices[0]
        log.info("Kokoro loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        return kokoro

    def synthesize_sync(self, text: str) -> np.ndarray:
        """Blocking synthesis (tts_pool thread, or tests). Empty/unspeakable text -> empty array."""
        text = text.strip()
        if not text or not any(ch.isalnum() for ch in text):
            return np.zeros(0, dtype=np.float32)
        try:
            audio, sr = self._kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        except (IndexError, ValueError, RuntimeError, KeyError):
            # Phonemizer/tokenizer edge cases (e.g. text that normalises to nothing).
            log.warning("kokoro failed on %r", text[:80], exc_info=True)
            return np.zeros(0, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != self.sample_rate:
            from macd.audio.resample import resample

            audio = resample(audio, sr, self.sample_rate)
        return audio

    async def synthesize(self, text: str) -> np.ndarray:
        hit = self._cache.get(text.strip())
        if hit is not None:
            return hit
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesize_sync, text)

    async def prewarm(self, texts: list[str] | tuple[str, ...] = DEFAULT_FILLERS) -> None:
        """Pre-synthesize filler phrases so the first acknowledgement is instant."""
        for t in texts:
            key = t.strip()
            if key and key not in self._cache:
                pcm = await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesize_sync, key)
                pcm.setflags(write=False)
                self._cache[key] = pcm

    # ---- blocking helpers for settings threads (never call on the event loop): they are
    # queued on tts_pool behind whatever is being synthesised, so playback is not disturbed.

    def voices(self) -> list[str]:
        return list(self._kokoro.get_voices())

    def render(self, text: str, voice: str | None = None, lang: str | None = None, speed: float | None = None) -> np.ndarray:
        """Synthesize ``text`` with an alternative voice/lang/speed without changing settings."""
        if voice is not None and voice not in self.voices():
            raise ValueError(f"unknown voice {voice!r}")

        def job() -> np.ndarray:
            saved = (self.voice, self.lang, self.speed)
            self.voice, self.lang, self.speed = voice or saved[0], lang or saved[1], speed or saved[2]
            try:
                return self.synthesize_sync(text)
            finally:
                self.voice, self.lang, self.speed = saved

        return self._executor.submit(job).result()

    def set_voice(self, voice: str, lang: str | None = None, speed: float | None = None, prewarm: tuple[str, ...] | list[str] = ()) -> None:
        """Switch voice/lang/speed; the filler cache is rebuilt (blocking, queued behind playback)."""
        if voice not in self.voices():
            raise ValueError(f"unknown voice {voice!r}")
        if speed is not None and not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        phrases = [t.strip() for t in (prewarm or self._cache)]

        def job() -> None:
            self.voice = voice
            self.lang = lang or self.lang
            self.speed = speed or self.speed
            self._cache = {}
            for t in phrases:
                if t:
                    pcm = self.synthesize_sync(t)
                    pcm.setflags(write=False)
                    self._cache[t] = pcm

        self._executor.submit(job).result()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class _Engine(Protocol):
    sample_rate: int

    def synthesize_sync(self, text: str) -> np.ndarray: ...
    def set_reference(self, wav: Path) -> None: ...
    def close(self) -> None: ...


FILLER_MIN_S, FILLER_MAX_S = 0.15, 2.5  # accepted duration for a cloned filler (else: babble/empty)
DOUBLED_MAX_S = 1.9  # a doubled one-word filler longer than this hit the engine's 2 s runaway cap
FILLER_ATTEMPTS = 3
SHORT_WORDS = 3  # runtime sentences shorter than this prefer a close cached filler (babble-prone)


def _first_half(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Cut a doubled phrase at the quietest 20 ms in its middle third (the pause between the two)."""
    win = sr // 50
    n = pcm.shape[0]
    if n < 4 * win:
        return pcm
    lo, hi = n // 3, 2 * n // 3
    energy = np.convolve(pcm[lo:hi] ** 2, np.ones(win, dtype=np.float32), mode="valid")
    cut = lo + int(np.argmin(energy)) + win // 2
    return pcm[:cut]


class CompositeTTS:
    """One voice at a time: Kokoro by default, a cloned voice (Pocket) when activated.

    With the clone active *everything* is spoken by it — fillers/short phrases are
    pre-synthesised when the voice is set (retrying babble, else a soft earcon stands in),
    unsuitable runtime inputs (one word) reuse the closest cached filler or the earcon, and
    Kokoro is only the loud fallback when the clone engine itself raises. All clone work runs
    on one thread ("clone_pool"), so caching and runtime synthesis never overlap.
    """

    def __init__(self, kokoro: KokoroTTS, clone: _Engine | None = None, unsuitable: type[BaseException] = ValueError, metrics: Metrics | None = None) -> None:
        self.kokoro = kokoro
        self.clone: _Engine | None = None
        self.unsuitable = unsuitable
        self.metrics = metrics or Metrics()
        self.sample_rate = kokoro.sample_rate
        self._fillers: list[str] = list(DEFAULT_FILLERS)
        self._clone_cache: dict[str, np.ndarray] = {}
        self._clone_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clone_pool")
        if clone is not None:
            self.activate_clone(clone)

    def route(self, text: str) -> str:
        return "clone" if self.clone is not None else "kokoro"

    # ------------------------------------------------------------------ voice switching (blocking)

    def activate_clone(self, engine: _Engine) -> None:
        """Make ``engine`` the voice and pre-synthesise the fillers with it (blocking, any thread)."""
        self.clone = engine
        self._clone_pool.submit(self._cache_fillers).result()

    def set_reference(self, wav: Path) -> None:
        if self.clone is None:
            raise RuntimeError("no clone engine active")
        self._clone_pool.submit(self._set_reference, wav).result()

    def _set_reference(self, wav: Path) -> None:
        assert self.clone is not None
        self.clone.set_reference(wav)
        self._clone_cache.clear()
        self._cache_fillers()

    def clone_render(self, text: str) -> np.ndarray:
        """Blocking preview with the clone engine (queued on clone_pool)."""
        if self.clone is None:
            raise RuntimeError("no clone engine active")
        engine = self.clone
        return self._clone_pool.submit(lambda: self._to_rate(engine.synthesize_sync(text), engine.sample_rate)).result()

    def deactivate_clone(self) -> None:
        old, self.clone = self.clone, None
        self._clone_cache.clear()
        if old is not None:
            self._clone_pool.submit(old.close).result()

    def _cache_fillers(self) -> None:
        """clone_pool thread: cache every filler; babble/empty -> retry, then a soft earcon."""
        from macd.audio.earcons import earcon

        engine = self.clone
        if engine is None:
            return
        ok = 0
        for phrase in self._fillers:
            if phrase in self._clone_cache:
                ok += 1
                continue
            pcm = None
            # One-word fillers ("Okay.") are unsuitable for the clone engine (it babbles), so they
            # are generated doubled ("Okay. Okay.") and cut at the pause in the middle.
            single = len(phrase.split()) == 1
            for _ in range(FILLER_ATTEMPTS):
                try:
                    cand = self._to_rate(engine.synthesize_sync(f"{phrase} {phrase}" if single else phrase), engine.sample_rate)
                except Exception:
                    log.warning("filler_clone_failed %r (engine error)", phrase, exc_info=True)
                    break
                if single:
                    if cand.shape[0] / self.sample_rate > DOUBLED_MAX_S:
                        continue  # truncated by the runaway cap: babble, try again
                    cand = _first_half(cand, self.sample_rate)
                dur = cand.shape[0] / self.sample_rate
                if FILLER_MIN_S <= dur <= FILLER_MAX_S:
                    pcm = cand
                    break
            if pcm is None:
                log.warning("filler_clone_failed %r: using a short earcon instead", phrase)
                pcm = earcon("tick", self.sample_rate)[0]
            else:
                ok += 1
            pcm.setflags(write=False)
            self._clone_cache[phrase] = pcm
        log.info("cloned voice ready: %d/%d fillers synthesised", ok, len(self._fillers))

    def _to_rate(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sr != self.sample_rate:
            from macd.audio.resample import resample

            pcm = resample(pcm, sr, self.sample_rate)
        return pcm

    def _closest_cached(self, text: str) -> np.ndarray | None:
        """The cached filler closest to ``text`` (difflib >= 0.6), if any."""
        import difflib

        keys = list(self._clone_cache)
        lower = [k.lower() for k in keys]
        match = difflib.get_close_matches(text.strip().lower(), lower, n=1, cutoff=0.6)
        return self._clone_cache[keys[lower.index(match[0])]] if match else None

    # ------------------------------------------------------------------ synthesis

    async def synthesize(self, text: str) -> np.ndarray:
        engine = self.clone
        if engine is None:
            return await self.kokoro.synthesize(text)
        t = text.strip()
        hit = self._clone_cache.get(t)
        if hit is None and len(t.split()) < SHORT_WORDS:
            hit = self._closest_cached(t)  # very short sentences are babble-prone on the engine
        if hit is not None:
            return hit
        loop = asyncio.get_running_loop()
        try:
            pcm = await loop.run_in_executor(self._clone_pool, engine.synthesize_sync, t)
        except self.unsuitable:
            from macd.audio.earcons import earcon

            return self._closest_cached(t) if self._closest_cached(t) is not None else earcon("tick", self.sample_rate)[0]
        except Exception:
            self.metrics.incr("tts_fallbacks")
            log.exception("cloned voice failed on %r; speaking it with Kokoro", t[:60])
            return await self.kokoro.synthesize(t)
        return self._to_rate(pcm, engine.sample_rate)

    async def prewarm(self, texts: list[str] | tuple[str, ...] = DEFAULT_FILLERS) -> None:
        """Kokoro filler cache; with a clone active, cache the same phrases in the clone too."""
        for t in texts:
            if t.strip() and t.strip() not in self._fillers:
                self._fillers.append(t.strip())
        await self.kokoro.prewarm(texts)
        if self.clone is not None:
            await asyncio.get_running_loop().run_in_executor(self._clone_pool, self._cache_fillers)

    def close(self) -> None:
        self.kokoro.close()
        if self.clone is not None:
            self.clone.close()
        self._clone_pool.shutdown(wait=False, cancel_futures=True)
