"""Synthesized short chimes (no asset files). ``earcon(name) -> (float32 pcm, sr)``."""

from __future__ import annotations

from functools import cache

import numpy as np

EARCON_SR = 24_000
_PEAK = 0.35  # well below full scale; earcons should be unobtrusive

# name -> list of (frequency_hz, start_s, duration_s)
_SPECS: dict[str, list[tuple[float, float, float]]] = {
    "wake": [(880.0, 0.0, 0.09), (1318.5, 0.07, 0.12)],  # rising fifth
    "listening": [(1046.5, 0.0, 0.08)],  # single soft blip
    "done": [(1318.5, 0.0, 0.09), (987.8, 0.07, 0.14)],  # falling
    "error": [(330.0, 0.0, 0.14), (247.0, 0.12, 0.2)],  # low descending
    "tick": [(660.0, 0.0, 0.06)],  # "still working": very short, lower and quieter than wake
}
_GAIN = {"tick": 0.5}  # relative to _PEAK

NAMES = tuple(_SPECS)


def _tone(freq: float, dur: float, sr: int) -> np.ndarray:
    t = np.arange(int(dur * sr), dtype=np.float32) / sr
    # fundamental + soft 2nd harmonic, exponential decay, short raised-cosine attack
    wave = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
    env = np.exp(-t / (dur * 0.35))
    attack = min(len(t), int(0.006 * sr))
    if attack:
        env[:attack] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, attack))
    release = min(len(t), int(0.01 * sr))
    if release:
        env[-release:] *= np.linspace(1.0, 0.0, release)
    return (wave * env).astype(np.float32)


@cache
def _render(name: str, sr: int) -> np.ndarray:
    spec = _SPECS[name]
    total = max(s + d for _, s, d in spec)
    out = np.zeros(int(total * sr) + 1, dtype=np.float32)
    for freq, start, dur in spec:
        tone = _tone(freq, dur, sr)
        i = int(start * sr)
        out[i : i + len(tone)] += tone
    peak = float(np.max(np.abs(out))) or 1.0
    out *= _PEAK * _GAIN.get(name, 1.0) / peak
    out.setflags(write=False)
    return out


def earcon(name: str, sr: int = EARCON_SR) -> tuple[np.ndarray, int]:
    """Return a (read-only, cached) chime. Raises KeyError for unknown names."""
    if name not in _SPECS:
        raise KeyError(f"unknown earcon {name!r}; known: {', '.join(NAMES)}")
    return _render(name, sr), sr
