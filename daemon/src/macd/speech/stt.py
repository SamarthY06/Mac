"""ParakeetSTT: NeMo Parakeet-TDT 0.6B v2 (int8) via sherpa-onnx OfflineRecognizer.

Owns a single-worker executor ("stt_pool") so STT never queues behind TTS and calls into
the recognizer are serialized. The model is loaded on that thread (lazily if asked).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from macd.contracts import SAMPLE_RATE

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_MIN_SAMPLES = SAMPLE_RATE // 10  # < 100 ms of audio: nothing meaningful to transcribe
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def clean_transcript(text: str) -> str:
    t = _WS.sub(" ", text.replace("▁", " ")).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", t)


class ParakeetSTT:
    """contracts.STT. ``lazy=True`` defers the model load to the first transcription."""

    def __init__(self, model_dir: Path, threads: int = 4, lazy: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.threads = threads
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt_pool")
        self._rec: Any = None
        if not lazy:
            self._load()

    @property
    def available(self) -> bool:
        """True if the model files are on disk (lazy instances may be created without them)."""
        return all((self.model_dir / f).is_file() for f in MODEL_FILES)

    def _load(self) -> Any:
        if self._rec is None:
            if not self.available:
                raise FileNotFoundError(f"Parakeet model not found in {self.model_dir} (run `python -m macd.speech.models`)")
            import sherpa_onnx

            d = self.model_dir
            t0 = time.perf_counter()
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(d / "encoder.int8.onnx"),
                decoder=str(d / "decoder.int8.onnx"),
                joiner=str(d / "joiner.int8.onnx"),
                tokens=str(d / "tokens.txt"),
                num_threads=self.threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                model_type="nemo_transducer",
            )
            log.info("Parakeet loaded in %.0f ms", (time.perf_counter() - t0) * 1000)
        return self._rec

    def transcribe_sync(self, pcm: np.ndarray) -> str:
        """Blocking transcription (stt_pool thread, or tests)."""
        x = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        if x.shape[0] < _MIN_SAMPLES:
            return ""
        rec = self._load()
        stream = rec.create_stream()
        # Pad 0.3 s of silence: transducers tend to drop the final word without it.
        stream.accept_waveform(SAMPLE_RATE, np.concatenate([x, np.zeros(SAMPLE_RATE * 3 // 10, np.float32)]))
        rec.decode_stream(stream)
        return clean_transcript(stream.result.text)

    async def transcribe(self, pcm: np.ndarray) -> str:
        return await asyncio.get_running_loop().run_in_executor(self._executor, self.transcribe_sync, pcm)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
