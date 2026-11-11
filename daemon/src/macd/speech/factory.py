"""Assemble the speech stack from config (incl. the optional cloned voice)."""

from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from macd import config
from macd.config import Config
from macd.contracts import VAD, AudioInput, WakeDetector
from macd.eventlog import Metrics
from macd.speech import models
from macd.speech.listener import ListenerImpl
from macd.speech.stt import ParakeetSTT
from macd.speech.tts import (
    LANGUAGES,
    PREVIEW_TEXT,
    CompositeTTS,
    KokoroTTS,
    describe_voice,
)
from macd.speech.wake import SherpaWakeDetector, WakePhrases, validate_wake_phrases

log = logging.getLogger(__name__)

VOICE_SET = "Done. This is my new voice."
VOICE_RESET = "Back to my default voice."


def make_pocket(cfg: Config, sample: Path) -> Any:
    """The cloned-voice engine (imported lazily: optional dependency/model)."""
    from macd.speech.pocket_tts import PocketTTS

    return PocketTTS(models.path("pocket"), reference_wav=sample, speed=cfg.tts.speed, threads=cfg.tts.clone_threads)


@dataclass
class SpeechStack:
    """Shared heavy models + per-source factories (KWS/VAD streams are per source)."""

    make_wake: Callable[[], WakeDetector]
    make_vad: Callable[[], VAD]
    stt: ParakeetSTT
    tts: CompositeTTS
    cfg: Config = field(default_factory=Config)
    make_clone: Callable[[Path], Any] = field(default=lambda sample: make_pocket(Config(), sample))  # injectable for tests
    clone_sample: str | None = None  # path of the active reference sample (None = Kokoro)
    voice_error: str | None = None  # why the persisted "clone" choice could not be loaded at startup
    make_kws_model: Callable[[list[str]], Any] = field(default=lambda phrases: None)  # injectable for tests
    make_detector: Callable[[Any], WakeDetector] = SherpaWakeDetector  # model -> per-listener detector
    turn: Any = None  # SmartTurn shared by all listeners (None = end of turn on silence alone)
    listeners: weakref.WeakSet[ListenerImpl] = field(default_factory=weakref.WeakSet)

    @property
    def voice_engine(self) -> str:
        return "clone" if self.tts.clone is not None else "kokoro"

    # ---- settings (all sync/blocking: call via to_thread; never on the event loop)

    def set_wake_phrases(self, phrases: list[str]) -> None:
        """Rebuild the KWS for ``phrases`` and swap it into every live listener; persists."""
        clean = validate_wake_phrases(phrases)
        model = self.make_kws_model(clean)  # ValueError if a phrase has no KWS tokens
        wp = WakePhrases(clean, accept_similar_names=self.cfg.wake.accept_similar_names)
        self.cfg.wake.phrases = clean
        if model is not None:
            self.make_wake = lambda: self.make_detector(model)
        for lis in list(self.listeners):
            if lis.source.requires_wake_word:
                lis.set_wake_detector(self.make_wake())
            lis.set_wake_phrases(wp)
        config.update_state(wake_phrases=clean)

    def set_kokoro_voice(self, voice: str, lang: str | None = None, speed: float | None = None) -> None:
        """Switch Kokoro's voice/lang/speed live (queued behind the item being spoken); persists."""
        lang = lang or describe_voice(voice)["lang"]
        self.tts.kokoro.set_voice(voice, lang, speed)
        self.cfg.tts.voice, self.cfg.tts.lang = voice, lang
        if speed is not None:
            self.cfg.tts.speed = speed
        config.update_state(tts_voice=voice, tts_lang=lang, tts_speed=self.cfg.tts.speed)

    def list_voices(self) -> list[dict[str, Any]]:
        voices = [describe_voice(v) for v in self.tts.kokoro.voices()]
        clone_ok = not models.missing_models(["pocket"])
        voices.append({"id": "clone", "name": "Your cloned voice", "lang": "en-us", "language": "English", "gender": "",
                       "available": clone_ok, "active": self.tts.clone is not None})
        return voices

    def preview(self, voice: str, text: str | None = None) -> Any:
        """A short sample in ``voice`` (Kokoro id or "clone") without changing settings; float32 @ tts.sample_rate."""
        if voice == "clone":
            if self.tts.clone is None:
                raise ValueError("no cloned voice is set up yet")
            return self.tts.clone_render(text or PREVIEW_TEXT["en-us"].format(name="Mac"))
        info = describe_voice(voice)
        text = text or PREVIEW_TEXT.get(info["lang"], PREVIEW_TEXT["en-us"]).format(name=info["name"])
        return self.tts.kokoro.render(text, voice=voice, lang=info["lang"])

    def voice_settings(self) -> dict[str, Any]:
        k = self.tts.kokoro
        return {
            "engine": self.voice_engine, "kokoro_voice": k.voice, "lang": k.lang, "speed": k.speed,
            "clone_sample": self.clone_sample, "wake_phrases": list(self.cfg.wake.phrases),
            "stt_languages": self.stt_languages(), "languages": {code: name for name, code in LANGUAGES.values()},
        }

    @staticmethod
    def stt_languages() -> list[str]:
        return ["en"]  # Parakeet-TDT v2 is English-only

    def set_voice(self, sample_path: str | None) -> str:
        """Blocking (call via to_thread). ``sample_path`` -> cloned voice from that sample (copied to
        VOICE_DIR); None -> Kokoro. Persists the choice in state.json; returns the confirmation
        to speak. Raises FileNotFoundError / ValueError for a bad sample or missing model."""
        if sample_path is None:
            self.tts.deactivate_clone()
            self.clone_sample = None
            config.update_state(voice_engine="kokoro", clone_sample="")
            return VOICE_RESET
        src = Path(sample_path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"voice sample not found: {src}")
        if models.missing_models(["pocket"]):
            raise ValueError("the voice-cloning model is not installed (python -m macd.speech.models pocket)")
        config.VOICE_DIR.mkdir(parents=True, exist_ok=True)
        dest = config.VOICE_DIR / ("sample" + (src.suffix.lower() or ".wav"))
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        if self.tts.clone is None:
            self.tts.activate_clone(self.make_clone(dest))  # also pre-synthesises the fillers
        else:
            self.tts.set_reference(dest)
        self.clone_sample = str(dest)
        self.voice_error = None
        config.update_state(voice_engine="clone", clone_sample=str(dest))
        return VOICE_SET

    async def warmup(self) -> None:
        """Run one STT and TTS inference in their executors so the first real turn is fast."""
        import numpy as np

        from macd.contracts import SAMPLE_RATE

        if self.stt.available:
            await self.stt.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        await self.tts.prewarm()  # with a clone active this also caches the fillers in the clone

    def close(self) -> None:
        self.stt.close()
        self.tts.close()


def _clone_engine(cfg: Config) -> tuple[Any | None, str | None, str | None]:
    """(engine, sample, error) for the persisted/configured voice choice (state.json wins)."""
    st = config.read_state()
    engine = st.get("voice_engine", cfg.tts.engine)
    sample = st.get("clone_sample", cfg.tts.clone_sample)
    if engine != "clone":
        return None, None, None
    if not sample or not Path(sample).is_file():
        return None, None, "cloned voice selected but the reference sample is missing"
    if models.missing_models(["pocket"]):
        return None, None, "cloned voice selected but the Pocket TTS model is not installed"
    try:
        return make_pocket(cfg, Path(sample)), sample, None
    except Exception as e:  # any load failure means: fall back to Kokoro, report once
        log.exception("could not load the cloned voice")
        return None, None, f"cloned voice failed to load: {e}"


def build_speech(cfg: Config, metrics: Metrics | None = None) -> SpeechStack:
    """Load models (KWS and VAD are warmed up here; call ``stack.warmup()`` from the loop
    for STT/TTS). KWS/VAD/Kokoro are required; a missing Parakeet is loaded lazily (the
    listener then accepts wake hits unverified until the model arrives)."""
    missing = models.missing_models(["kws", "vad", "kokoro", "voices"])
    if missing:
        raise FileNotFoundError(f"speech models missing: {', '.join(missing)} (run `python -m macd.speech.models`)")
    # kokoro-onnx does Python-level work (tokeniser/phonemiser) on the tts_pool thread and
    # holds the GIL for up to a switch interval each time the loop thread needs it back;
    # with the default 5 ms that adds up to ~90 ms per frame on the listener (measured).
    # 1 ms keeps the loop's per-frame time around 1 ms while TTS runs.
    if sys.getswitchinterval() > 0.001:
        sys.setswitchinterval(0.001)
    st = config.read_state()  # settings changed at runtime win over config.toml
    if st.get("wake_phrases"):
        cfg.wake.phrases = list(st["wake_phrases"])
    cfg.tts.voice = st.get("tts_voice", cfg.tts.voice)
    cfg.tts.lang = st.get("tts_lang", cfg.tts.lang)
    cfg.tts.speed = float(st.get("tts_speed", cfg.tts.speed))
    stt_missing = bool(models.missing_models(["parakeet"]))
    if stt_missing:
        log.warning("Parakeet STT model missing; wake verification and transcription are disabled until it is downloaded")
    from macd.speech.wake import KeywordSpotterModel

    def make_kws_model(phrases: list[str]) -> KeywordSpotterModel:
        return KeywordSpotterModel(models.path("kws"), phrases, cfg.wake.kws_score, cfg.wake.kws_threshold)

    kws = make_kws_model(cfg.wake.phrases)
    vad_path = models.path("vad")
    from macd.speech.vad import SileroVAD
    stt = ParakeetSTT(models.path("parakeet"), threads=cfg.stt.threads, lazy=cfg.stt.lazy or stt_missing)
    kokoro = KokoroTTS(models.path("kokoro"), models.path("voices"), voice=cfg.tts.voice, speed=cfg.tts.speed, lang=cfg.tts.lang)
    unsuitable: type[BaseException] = ValueError
    with contextlib.suppress(ImportError):
        from macd.speech.pocket_tts import UnsuitableText

        unsuitable = UnsuitableText
    clone, sample, error = _clone_engine(cfg)
    if error:
        log.warning("%s; using Kokoro", error)
    return SpeechStack(
        turn=_smart_turn(cfg),
        make_wake=lambda: SherpaWakeDetector(kws),
        make_vad=lambda: SileroVAD(vad_path),
        stt=stt,
        tts=CompositeTTS(kokoro, clone, unsuitable=unsuitable, metrics=metrics),
        cfg=cfg,
        make_clone=lambda s: make_pocket(cfg, s),
        clone_sample=sample,
        voice_error=error,
        make_kws_model=make_kws_model,
    )


def _smart_turn(cfg: Config) -> Any:
    if not cfg.listen.smart_turn:
        return None
    if models.missing_models(["turn"]):
        log.warning("Smart Turn model missing; ending turns on %d ms of silence (run `python -m macd.speech.models`)", cfg.listen.silence_ms)
        return None
    from macd.speech.turn import SmartTurn

    try:
        return SmartTurn(models.path("turn"))
    except Exception:
        log.exception("Smart Turn failed to load; ending turns on silence")
        return None


def make_listener(
    source: AudioInput, stack: SpeechStack, cfg: Config, aec: bool | Callable[[], bool] = False,
    metrics: Metrics | None = None, eventlog: Any = None,
) -> ListenerImpl:
    """One Listener per AudioInput; sources that don't need a wake word get no KWS.
    Pass the daemon's ``Metrics`` (kws_hits/wakes/false_wakes/wakes_suppressed counters and
    kws_hit_to_wake/listener_* latencies in `status`) and ``EventLog`` (kws_hit / wake_verify /
    wake_suppressed events)."""
    wake = stack.make_wake() if source.requires_wake_word else None
    lis = ListenerImpl(source, wake, stack.make_vad(), stack.stt, cfg, aec=aec, metrics=metrics, eventlog=eventlog, turn=stack.turn)
    stack.listeners.add(lis)  # so set_wake_phrases can swap detectors live
    return lis
