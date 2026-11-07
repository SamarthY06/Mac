"""SileroVAD: per-frame speech probability from silero_vad.onnx.

Silero runs on 512-sample windows (32 ms @ 16 kHz) with recurrent state, while the
pipeline feeds 320-sample frames. We buffer internally, run every complete window and
return the most recent window's probability for each frame.

The model is driven directly with onnxruntime (sherpa-onnx's VoiceActivityDetector only
exposes speech segments, not probabilities). Not thread-safe: one instance per source,
used from the event loop only (~0.1 ms per window, single intra-op thread).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from macd.contracts import SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW = 512


class SileroVAD:
    """contracts.VAD over silero v4/v5 ONNX exports."""

    def __init__(self, model_path: Path | str, threads: int = 1) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        names = {i.name: i for i in self._sess.get_inputs()}
        self._v5 = "state" in names
        if self._v5:
            self._context_size = 64
        elif not {"h", "c"} <= names.keys():
            raise ValueError(f"unrecognised silero VAD inputs: {sorted(names)}")
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._buf = np.zeros(0, dtype=np.float32)
        self.prob = 0.0
        self.reset()
        self._run(np.zeros(WINDOW, dtype=np.float32))  # warm-up (first ORT run is slow)
        self.reset()

    def reset(self) -> None:
        if self._v5:
            self._state: Any = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros(self._context_size, dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self.prob = 0.0

    def _run(self, window: np.ndarray) -> float:
        if self._v5:
            x = np.concatenate([self._context, window])[None, :]
            out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
            self._context = window[-self._context_size :]
        else:
            out, self._h, self._c = self._sess.run(None, {"input": window[None, :], "h": self._h, "c": self._c, "sr": self._sr})
        return float(np.asarray(out).reshape(-1)[0])

    def accept(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        while self._buf.shape[0] >= WINDOW:
            self.prob = self._run(self._buf[:WINDOW])
            self._buf = self._buf[WINDOW:]
        return self.prob
