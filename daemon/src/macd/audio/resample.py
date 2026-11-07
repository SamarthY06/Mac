"""Resampling and sample-format helpers (soxr)."""

from __future__ import annotations

import numpy as np
import soxr


def resample(pcm: np.ndarray, sr_in: int, sr_out: int, quality: str = "HQ") -> np.ndarray:
    """One-shot resample of a mono float32 signal."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if sr_in == sr_out or x.size == 0:
        return x
    return soxr.resample(x, sr_in, sr_out, quality=quality).astype(np.float32, copy=False)


class StreamResampler:
    """Chunked resampler for live streams (keeps filter state between chunks)."""

    def __init__(self, sr_in: int, sr_out: int, quality: str = "HQ") -> None:
        self.sr_in, self.sr_out = sr_in, sr_out
        self._rs = None if sr_in == sr_out else soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality=quality)

    def process(self, chunk: np.ndarray, last: bool = False) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._rs is None:
            return x
        return self._rs.resample_chunk(x, last=last)


def float_to_int16(pcm: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).round().astype(np.int16)


def int16_to_float(pcm: np.ndarray) -> np.ndarray:
    return np.asarray(pcm, dtype=np.int16).astype(np.float32) / 32768.0
