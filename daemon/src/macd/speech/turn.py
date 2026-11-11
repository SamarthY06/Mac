"""SmartTurn: "has the speaker finished?" from the audio itself (Pipecat Smart Turn v3).

Silence alone cannot tell "open the file about..." (thinking, more to come) from "open the
file." (done). Smart Turn is a Whisper-tiny encoder with a classifier head trained on the
prosody and phrasing of turn ends; it reads the last 8 s of a segment and returns the
probability that the turn is complete. ~10 ms on CPU, 8.7 MB int8 ONNX.

The log-mel front end below is adapted from Pipecat's numpy port of
``transformers.WhisperFeatureExtractor`` (pipecat/audio/turn/smart_turn/_whisper_features.py,
Copyright (c) 2024-2026 Daily, BSD-2-Clause; portions Copyright 2022-2023 The HuggingFace
Inc. team and the librosa & torchaudio authors, Apache-2.0). Model weights: BSD-2-Clause.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

log = logging.getLogger(__name__)

SR = 16_000
WINDOW_S = 8
_N_FFT, _HOP, _N_MELS = 400, 160, 80


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    m = 3.0 * f / 200.0
    hi = f >= 1000.0
    m[hi] = 15.0 + np.log(f[hi] / 1000.0) * (27.0 / np.log(6.4))
    return m


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    f = 200.0 * m / 3.0
    hi = m >= 15.0
    f[hi] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (m[hi] - 15.0))
    return f


def _mel_filters() -> np.ndarray:
    """Slaney-normalised triangular filterbank, shape (n_fft/2+1, n_mels)."""
    mels = np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([SR / 2]))[0], _N_MELS + 2)
    ff = _mel_to_hz(mels)
    bins = np.linspace(0, SR // 2, _N_FFT // 2 + 1)
    diff = np.diff(ff)
    slopes = ff[None, :] - bins[:, None]
    fb = np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))
    return fb * (2.0 / (ff[2:_N_MELS + 2] - ff[:_N_MELS]))[None, :]


_FILTERS = _mel_filters()
_HANN = np.hanning(_N_FFT + 1)[:-1]


def features(audio: np.ndarray) -> np.ndarray:
    """Whisper log-mel features (80 x 800) of the last 8 s of ``audio`` (16 kHz), left-padded."""
    n = SR * WINDOW_S
    x = np.asarray(audio, dtype=np.float32)[-n:]
    if x.size < n:
        x = np.pad(x, (n - x.size, 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    pad = _N_FFT // 2
    frames = sliding_window_view(np.pad(x.astype(np.float64), (pad, pad), mode="reflect"), _N_FFT)[::_HOP]
    power = (np.abs(np.fft.rfft(frames * _HANN, axis=-1)) ** 2).T
    logm = np.log10(np.maximum(1e-10, _FILTERS.T @ power))[:, :-1]
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32)


class SmartTurn:
    """``complete_prob(audio)`` -> P(turn finished). Thread-safe (one inference at a time)."""

    def __init__(self, path: Path, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    def complete_prob(self, audio: np.ndarray) -> float:
        feats = features(audio)[None, ...]
        with self._lock:
            out = self._sess.run(None, {"input_features": feats})
        return float(np.asarray(out[0]).reshape(-1)[0])
