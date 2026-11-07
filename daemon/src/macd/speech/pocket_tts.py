"""PocketTTS: Kyutai Pocket TTS (100M, voice cloning) via sherpa-onnx, on its own executor.

Same shape as :class:`macd.speech.tts.KokoroTTS` (contracts.TTS). Differences that the caller
must handle:

- A reference voice is required (``reference_wav`` or :meth:`set_reference`); any audio file
  ffmpeg can read is accepted, decoded to 24 kHz mono, edge-silence trimmed, cut to 15 s and
  peak-normalised. The clip's quality is reproduced in the output, so use a clean recording.
- Very short inputs make the model babble for tens of seconds. ``synthesize`` therefore raises
  :class:`UnsuitableText` for empty / one-word text (route those, and all fillers and earcons,
  to Kokoro), and every output is capped at ``min(max_seconds, 0.25 s * words / speed + 1.5 s)``
  (truncated at a zero crossing, logged as ``pocket_runaway``).
- The sherpa offline path only returns audio once a whole input has been generated (about
  0.2 s per second of speech), so text is synthesised in short chunks: a first chunk of at most
  ``FIRST_CHUNK_WORDS`` words (cut at clause punctuation or before a conjunction), then clause-
  sized chunks. :meth:`synthesize_stream` yields chunks as they are ready; :meth:`synthesize`
  concatenates them. ``last_ttfa_ms`` is the time until the first chunk existed.

Model: ``kyutai-pocket-tts-int8`` (code MIT, weights CC-BY-4.0: attribute Kyutai; the licence
prohibits cloning a voice without the speaker's consent).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

POCKET_SR = 24_000
MODEL_FILES = (
    "lm_main.int8.onnx",
    "lm_flow.int8.onnx",
    "encoder.onnx",
    "decoder.int8.onnx",
    "text_conditioner.onnx",
    "vocab.json",
    "token_scores.json",
)
MIN_WORDS = 2
MIN_REFERENCE_S = 1.0
MAX_REFERENCE_S = 15.0
CAP_PER_WORD_S = 0.25
CAP_BASE_S = 1.5
WARMUP_TEXT = "Hello, this is a short warm-up sentence."
FIRST_CHUNK_WORDS = 12  # hard limit for the first chunk (~3.5 s of speech -> first audio in ~500 ms worst case)
CHUNK_WORDS = 16  # hard limit for later chunks, synthesised while the previous one plays
MIN_CHUNK_WORDS = 3  # never cut at punctuation before this many words
FIRST_CONJ_WORDS = 4  # first chunk: cut before a conjunction from this many words (-> ~200-300 ms TTFA)
LATER_CONJ_WORDS = 8  # later chunks: cut before a conjunction only once this long
CHUNK_GAP_S = 0.06  # silence inserted between chunks when concatenating

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CLAUSE_END = re.compile(r"[,;:—–]$|[.!?]['\")]*$")
_CONJUNCTIONS = frozenset({"and", "but", "so", "or", "because", "then", "that", "which", "when", "while", "after", "before"})


class UnsuitableText(ValueError):
    """Text the clone engine must not be given (empty, no letters, or a single word)."""


def _count_words(text: str) -> int:
    return len(_WORD.findall(text))


def split_chunks(text: str, first_max: int = FIRST_CHUNK_WORDS, max_words: int = CHUNK_WORDS) -> list[str]:
    """Split text into synthesis chunks at natural pauses: after clause punctuation (once a chunk
    has ``MIN_CHUNK_WORDS`` words) or before a conjunction (once it has ``FIRST_CONJ_WORDS`` /
    ``LATER_CONJ_WORDS`` words); a hard word limit is the last resort because cutting mid-phrase
    makes the model stumble. Every chunk has at least 2 words (a trailing short piece is merged
    into the previous chunk)."""
    tokens = text.split()
    chunks: list[list[str]] = []
    cur: list[str] = []
    limit, conj_min = first_max, FIRST_CONJ_WORDS
    for i, tok in enumerate(tokens):
        if len(cur) >= conj_min and tok.lower().strip("'\"") in _CONJUNCTIONS:
            chunks.append(cur)
            cur, limit, conj_min = [], max_words, LATER_CONJ_WORDS
        cur.append(tok)
        last = i == len(tokens) - 1
        if not last and ((len(cur) >= MIN_CHUNK_WORDS and _CLAUSE_END.search(tok)) or len(cur) >= limit):
            chunks.append(cur)
            cur, limit, conj_min = [], max_words, LATER_CONJ_WORDS
    if cur:
        if len(cur) < 2 and chunks:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return [" ".join(c) for c in chunks]


def cap_audio(pcm: np.ndarray, sample_rate: int, cap_s: float) -> tuple[np.ndarray, bool]:
    """Truncate ``pcm`` to at most ``cap_s`` seconds at the last zero crossing before the cap.
    Returns (audio, truncated)."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    limit = int(cap_s * sample_rate)
    if x.shape[0] <= limit:
        return x, False
    head = x[:limit]
    sign = np.signbit(head)
    crossings = np.flatnonzero(sign[1:] != sign[:-1])
    # Look for a zero crossing in the last 50 ms before the cap, so we don't cut a waveform mid-cycle.
    window_start = max(0, limit - sample_rate // 20)
    recent = crossings[crossings >= window_start]
    cut = int(recent[-1]) + 1 if recent.size else limit
    out = head[:cut].copy()
    fade = min(out.shape[0], sample_rate // 100)  # 10 ms fade-out
    if fade:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out, True


def _trim_silence(x: np.ndarray, sr: int, frame_ms: int = 20, rel_threshold: float = 0.05) -> np.ndarray:
    """Strip leading/trailing frames whose RMS is below ``rel_threshold`` of the loudest frame."""
    n = sr * frame_ms // 1000
    if x.shape[0] < 2 * n:
        return x
    frames = x[: x.shape[0] // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return x
    loud = np.flatnonzero(rms >= rel_threshold * peak)
    if loud.size == 0:
        return x
    return x[loud[0] * n : (loud[-1] + 1) * n]


def load_reference(path: Path, sample_rate: int = POCKET_SR, max_seconds: float = MAX_REFERENCE_S) -> np.ndarray:
    """Decode any audio file to float32 mono at ``sample_rate`` (ffmpeg; plain WAV works without
    it), trim edge silence, keep the first ``max_seconds`` and peak-normalise to 0.9."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference audio not found: {path}")
    x = _decode(path, sample_rate)
    x = _trim_silence(x, sample_rate)
    if x.shape[0] < MIN_REFERENCE_S * sample_rate:
        raise ValueError(f"reference audio too short after trimming ({x.shape[0] / sample_rate:.2f} s < {MIN_REFERENCE_S} s)")
    x = x[: int(max_seconds * sample_rate)]
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x * (0.9 / peak)
    return np.ascontiguousarray(x, dtype=np.float32)


def _decode(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg could not decode {path.name}: {proc.stderr.decode(errors='replace').strip()[:200]}")
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    # No ffmpeg: accept 16-bit PCM WAV only.
    import wave

    from macd.audio.resample import resample

    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("without ffmpeg only 16-bit PCM WAV references are supported")
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            ch, sr = w.getnchannels(), w.getframerate()
    except wave.Error as e:
        raise ValueError(f"cannot read {path.name} (install ffmpeg for non-WAV references): {e}") from e
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    return resample(raw, sr, sample_rate)


class PocketTTS:
    """contracts.TTS backed by Kyutai Pocket TTS (sherpa-onnx). Synthesis is serialized on a
    single-worker executor; the reference voice can be swapped at any time."""

    sample_rate: int = POCKET_SR

    def __init__(
        self,
        model_dir: Path,
        reference_wav: Path | None = None,
        speed: float = 1.0,
        threads: int = 2,
        max_seconds: float = 20.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.speed, self.threads, self.max_seconds = float(speed), int(threads), float(max_seconds)
        self.num_steps = 2  # flow-matching steps; 2 vs 5 made no audible/measurable difference
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts_pool")
        self._gen: Any = None  # sherpa_onnx.GenerationConfig holding the reference audio
        self.reference_id: str | None = None
        self.reference_seconds: float = 0.0
        self.last_rtf: float | None = None
        self.last_ttfa_ms: float | None = None
        self.last_runaway: bool = False
        self._tts = self._load()
        if reference_wav is not None:
            self.set_reference(reference_wav)

    @property
    def available(self) -> bool:
        return all((self.model_dir / f).is_file() for f in MODEL_FILES)

    def _load(self) -> Any:
        if not self.available:
            missing = [f for f in MODEL_FILES if not (self.model_dir / f).is_file()]
            raise FileNotFoundError(f"Pocket TTS model incomplete in {self.model_dir}: missing {missing}")
        import sherpa_onnx

        d = self.model_dir
        t0 = time.perf_counter()
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                    lm_flow=str(d / "lm_flow.int8.onnx"),
                    lm_main=str(d / "lm_main.int8.onnx"),
                    encoder=str(d / "encoder.onnx"),
                    decoder=str(d / "decoder.int8.onnx"),
                    text_conditioner=str(d / "text_conditioner.onnx"),
                    vocab_json=str(d / "vocab.json"),
                    token_scores_json=str(d / "token_scores.json"),
                ),
                num_threads=self.threads,
                provider="cpu",
                debug=False,
            )
        )
        if not cfg.validate():
            raise ValueError(f"invalid Pocket TTS config for {d}")
        tts = sherpa_onnx.OfflineTts(cfg)
        log.info("Pocket TTS loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        return tts

    # -- reference voice --------------------------------------------------------------------

    def set_reference(self, wav: Path) -> None:
        """Decode + condition a reference clip and make it the voice for subsequent synthesis.
        Blocking (ffmpeg); may be called from any thread — the swap is atomic."""
        import sherpa_onnx

        ref = load_reference(wav, self.sample_rate)
        gen = sherpa_onnx.GenerationConfig()
        gen.reference_audio = ref
        gen.reference_sample_rate = self.sample_rate
        gen.num_steps = self.num_steps
        gen.speed = self.speed
        self._gen = gen
        self.reference_id = hashlib.sha256(ref.tobytes()).hexdigest()[:16]
        self.reference_seconds = ref.shape[0] / self.sample_rate
        log.info("Pocket TTS reference set from %s: %.1f s (id %s)", Path(wav).name, self.reference_seconds, self.reference_id)

    @property
    def has_reference(self) -> bool:
        return self._gen is not None

    # -- synthesis --------------------------------------------------------------------------

    @staticmethod
    def check_text(text: str) -> str:
        """Return the stripped text or raise UnsuitableText (empty, no letters, or < 2 words)."""
        text = text.strip()
        if not text or not any(ch.isalnum() for ch in text):
            raise UnsuitableText("empty text")
        if _count_words(text) < MIN_WORDS:
            raise UnsuitableText(f"too short for the clone engine: {text!r}")
        return text

    def cap_seconds(self, text: str) -> float:
        return min(self.max_seconds, CAP_PER_WORD_S * _count_words(text) / max(self.speed, 0.25) + CAP_BASE_S)

    def _generate(self, chunk: str, gen: Any) -> np.ndarray:
        """One model call; runaway-capped. Blocking."""
        try:
            audio = self._tts.generate(chunk, gen)
        except RuntimeError:
            log.warning("pocket failed on %r", chunk[:80], exc_info=True)
            raise
        pcm = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        if audio.sample_rate != self.sample_rate:
            from macd.audio.resample import resample

            pcm = resample(pcm, audio.sample_rate, self.sample_rate)
        if pcm.shape[0] == 0:
            raise RuntimeError(f"Pocket TTS produced no audio for {chunk[:80]!r}")
        cap = self.cap_seconds(chunk)
        pcm, truncated = cap_audio(pcm, self.sample_rate, cap)
        if truncated:
            self.last_runaway = True
            log.warning("pocket_runaway: %r produced > %.1f s of audio; truncated", chunk[:80], cap)
        return pcm

    def _begin(self, text: str) -> tuple[list[str], Any, float]:
        text = self.check_text(text)
        gen = self._gen
        if gen is None:
            raise RuntimeError("Pocket TTS: no reference voice set")
        self.last_runaway = False
        self.last_rtf = self.last_ttfa_ms = None
        return split_chunks(text), gen, time.perf_counter()

    def _record(self, t0: float, samples: int, first: bool) -> None:
        elapsed = time.perf_counter() - t0
        if first:
            self.last_ttfa_ms = elapsed * 1000.0
        self.last_rtf = elapsed / (samples / self.sample_rate) if samples else None

    def synthesize_sync(self, text: str) -> np.ndarray:
        """Blocking synthesis of the whole text (tts_pool thread, or tests).
        Raises UnsuitableText / RuntimeError."""
        chunks, gen, t0 = self._begin(text)
        gap = np.zeros(int(CHUNK_GAP_S * self.sample_rate), np.float32)
        parts: list[np.ndarray] = []
        total = 0
        for i, chunk in enumerate(chunks):
            pcm = self._generate(chunk, gen)
            if i:
                parts.append(gap)
                total += gap.shape[0]
            parts.append(pcm)
            total += pcm.shape[0]
            self._record(t0, total, first=i == 0)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    async def synthesize(self, text: str) -> np.ndarray:
        self.check_text(text)  # fail fast on the loop thread, before queueing
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.synthesize_sync, text)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield float32 chunks (first one ~6 words) as each is synthesised on the executor.
        Playing chunk N while N+1 is generated keeps time-to-first-audio at ~300 ms."""
        chunks, gen, t0 = self._begin(text)
        loop = asyncio.get_running_loop()
        total = 0
        for i, chunk in enumerate(chunks):
            pcm = await loop.run_in_executor(self._executor, self._generate, chunk, gen)
            total += pcm.shape[0]
            self._record(t0, total, first=i == 0)
            yield pcm

    def warmup_sync(self) -> None:
        """One short synthesis so the first real sentence doesn't pay for ORT lazy init."""
        if self._gen is None:
            return
        t0 = time.perf_counter()
        self.synthesize_sync(WARMUP_TEXT)
        log.info("Pocket TTS warm-up in %.0f ms", (time.perf_counter() - t0) * 1000)

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self.warmup_sync)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
