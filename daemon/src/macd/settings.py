"""Runtime settings for the app's Settings window and `macctl settings` (control commands
settings_get / settings_set / voices_list / voice_preview).

Keys are the dotted names in `config.SETTABLE_KEYS`; the friendly names returned by
`settings_get` (ALIASES) are accepted too. A change is validated (type + range), applied LIVE
where it matters (brain name, wake phrases, Kokoro voice, …), then the whole config is saved
atomically. Partial failure is reported per key; successful keys are still applied + saved.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from macd import config

log = logging.getLogger(__name__)

ALIASES: dict[str, str] = {
    "name": "name", "user_name": "user_name", "results_dir": "results_dir",
    "wake_phrases": "wake.phrases", "language": "tts.lang", "voice": "tts.voice", "speed": "tts.speed",
    "engine": "tts.engine", "idle_shutdown_min": "brain.idle_shutdown_min", "followup_s": "listen.followup_s",
    "silence_ms": "listen.silence_ms", "chrome": "brain.chrome", "setting_sources": "brain.setting_sources",
    "remote_enabled": "remote.enabled",
}
STT_LANGUAGES = ["en"]  # Parakeet TDT 0.6B v2 is English-only
PREVIEW_TEXT = "Hi, I'm Mac. This is how I sound."
MAX_NAME_LEN = 32
RANGES: dict[str, tuple[float, float]] = {
    "tts.speed": (0.5, 2.0), "listen.silence_ms": (200, 3000), "listen.followup_s": (0, 60),
    "listen.max_utterance_s": (5, 600), "brain.idle_shutdown_min": (1, 720), "wake.kws_threshold": (0.01, 1.0),
}


def canonical(key: str) -> str:
    return ALIASES.get(key, key)


def validate(key: str, value: Any) -> None:
    """Range / shape checks beyond `config.set_key`'s type coercion. Raises ValueError."""
    if key == "name":
        if not value.strip() or len(value) > MAX_NAME_LEN or not value.replace(" ", "").replace("-", "").isalnum():
            raise ValueError("name must be 1 to 32 letters, digits, spaces or dashes")
    elif key == "wake.phrases":
        if not value:
            raise ValueError("at least one wake phrase is required")
        for p in value:
            if not 2 <= len(p.split()) <= 4 or not p.replace(" ", "").replace("'", "").isalpha():
                raise ValueError(f"wake phrase {p!r} must be two to four words (letters only)")
    elif key == "tts.engine":
        if value not in ("kokoro", "clone"):
            raise ValueError("engine must be 'kokoro' or 'clone'")
    elif key == "tts.lang" and not value.strip():
        raise ValueError("language must not be empty")
    elif key == "tts.voice" and not value.strip():
        raise ValueError("voice must not be empty")
    elif key in RANGES:
        lo, hi = RANGES[key]
        if not lo <= value <= hi:
            raise ValueError(f"{key} must be between {lo:g} and {hi:g}")


def wake_phrases_for(name: str) -> list[str]:
    n = name.strip().lower()
    return [f"hey {n}", f"ok {n}"]


@dataclass
class Settings:
    cfg: config.Config
    stack: Any | None = None  # SpeechStack: set_wake_phrases / set_kokoro_voice / list_voices / preview / set_voice
    brain: Any | None = None  # Brain: set_name
    speak: Callable[[str], None] | None = None  # orchestrator.say
    play: Callable[[np.ndarray, int], None] | None = None  # speech.play_pcm as a system item
    save: Callable[[config.Config], None] = config.save
    prewarm: Callable[[], Awaitable[None]] | None = None  # re-warm the filler cache after a voice change

    # ------------------------------------------------------------------ get

    def get(self) -> dict[str, Any]:
        """Flat, keyed exactly like `settings_set` (dotted config keys) + read-only extras."""
        c = self.cfg
        data: dict[str, Any] = {key: config.get_key(c, key) for key in config.SETTABLE_KEYS}
        data["wake.phrases"] = list(c.wake.phrases)
        data["brain.setting_sources"] = list(c.brain.setting_sources)
        data["tts.engine"] = self.stack.voice_engine if self.stack is not None else c.tts.engine
        data["clone_sample"] = (self.stack.clone_sample if self.stack is not None else c.tts.clone_sample) or None
        data["stt_languages"] = list(STT_LANGUAGES)
        data["aliases"] = dict(ALIASES)
        return data

    # ------------------------------------------------------------------ set

    async def set(self, changes: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(changes, dict) or not changes:
            raise ValueError("settings_set needs a non-empty 'changes' object")
        applied: dict[str, Any] = {}
        errors: dict[str, str] = {}
        coerced: dict[str, Any] = {}
        trial = copy.deepcopy(self.cfg)
        for raw_key, value in changes.items():
            key = canonical(str(raw_key))
            try:
                v = config.set_key(trial, key, value)
                validate(key, v)
                coerced[key] = v
            except (ValueError, TypeError) as e:
                errors[str(raw_key)] = str(e)
        if "name" in coerced and "wake.phrases" not in coerced and "wake.phrases" not in errors:
            # Renaming implies the wake phrases unless they were given explicitly.
            coerced["wake.phrases"] = wake_phrases_for(coerced["name"])
        for key, v in coerced.items():
            try:
                await self._apply_live(key, v)
                config.set_key(self.cfg, key, v)
                applied[key] = v
            except Exception as e:
                log.exception("applying setting %s failed", key)
                errors[key] = str(e)
        if applied:
            self.save(self.cfg)
            self._announce(applied)
        return {"applied": applied, "errors": errors}

    async def _apply_live(self, key: str, value: Any) -> None:
        if key == "name" and self.brain is not None:
            res = self.brain.set_name(value)
            if asyncio.iscoroutine(res):
                await res
        elif key == "wake.phrases" and self.stack is not None:
            await asyncio.to_thread(self.stack.set_wake_phrases, list(value))
        elif key in ("tts.voice", "tts.lang", "tts.speed") and self.stack is not None:
            t = self.cfg.tts
            voice = value if key == "tts.voice" else t.voice
            lang = value if key == "tts.lang" else t.lang
            speed = value if key == "tts.speed" else t.speed
            await asyncio.to_thread(self.stack.set_kokoro_voice, voice, lang, float(speed))
            if self.prewarm is not None:
                await self.prewarm()
        elif key == "tts.engine" and self.stack is not None:
            if value == "kokoro":
                await asyncio.to_thread(self.stack.set_voice, None)
            elif not self.stack.clone_sample:
                raise ValueError("no cloned voice yet: record one with `macctl voice set <wav>` first")
            else:
                await asyncio.to_thread(self.stack.set_voice, self.stack.clone_sample)
        # listen.* / brain.idle_shutdown_min / remote.enabled / brain.* are read from cfg live.

    def _announce(self, applied: dict[str, Any]) -> None:
        if self.speak is None:
            return
        if applied.get("wake.phrases"):
            self.speak(f"From now on, say {applied['wake.phrases'][0].title()}.")
        elif "name" in applied:
            self.speak(f"From now on, call me {applied['name']}.")

    # ------------------------------------------------------------------ voices

    async def voices(self) -> list[dict[str, Any]]:
        if self.stack is None:
            raise ValueError("voice is unavailable (speech models not loaded)")
        return list(await asyncio.to_thread(self.stack.list_voices))

    async def preview(self, voice: Any, text: Any = None) -> dict[str, Any]:
        if self.stack is None or self.play is None:
            raise ValueError("voice is unavailable (speech models not loaded)")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice_preview needs a 'voice' id")
        sample = text if isinstance(text, str) and text.strip() else PREVIEW_TEXT
        try:
            pcm = await asyncio.to_thread(self.stack.preview, voice.strip(), sample[:300])
        except (ValueError, KeyError) as e:
            raise ValueError(f"cannot preview voice {voice!r}: {e}") from None
        self.play(np.asarray(pcm, dtype=np.float32), int(self.stack.tts.sample_rate))
        return {"queued": True, "voice": voice.strip(), "text": sample[:300]}
