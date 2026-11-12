"""InputGain: the slow gain on the KWS feed."""

from __future__ import annotations

import numpy as np

from macd.contracts import SAMPLE_RATE
from macd.speech.wake import InputGain

CHUNK = 1280  # 80 ms, the KWS batch


def tone(peak: float, n: int = CHUNK) -> np.ndarray:
    return (peak * np.sin(np.arange(n) * 0.3)).astype(np.float32)


def test_quiet_speech_is_lifted_to_the_target_without_clipping():
    g = InputGain(target=0.2, max_gain=30.0)
    out = g(tone(0.01))
    assert out.dtype == np.float32
    assert 0.19 < float(np.abs(out).max()) <= 0.2001


def test_gain_is_capped_for_near_silence():
    g = InputGain(target=0.2, max_gain=30.0)
    out = g(tone(0.001))
    assert float(np.abs(out).max()) <= 0.001 * 30.0 + 1e-6


def test_loud_input_passes_unchanged():
    g = InputGain(target=0.2)
    x = tone(0.6)
    assert g(x) is x


def test_a_loud_burst_lowers_the_gain_then_it_recovers_slowly():
    g = InputGain(target=0.2, max_gain=30.0, release_s=3.0)
    g(tone(0.4))  # a clap
    after_clap = float(np.abs(g(tone(0.01))).max())
    assert after_clap < 0.011  # the envelope still remembers the clap: little gain
    for _ in range(int(15 * SAMPLE_RATE / CHUNK)):  # 15 s of quiet room
        g(tone(0.005))
    assert float(np.abs(g(tone(0.01))).max()) > 0.19  # back to full lift


def test_empty_chunk():
    g = InputGain()
    x = np.zeros(0, np.float32)
    assert g(x) is x
