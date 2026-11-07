import numpy as np
import pytest

from macd.audio.earcons import NAMES, earcon


@pytest.mark.parametrize("name", ["wake", "done", "error", "listening", "tick"])
def test_earcon_shape(name):
    pcm, sr = earcon(name)
    assert name in NAMES
    assert pcm.dtype == np.float32 and pcm.ndim == 1
    assert 0.05 < len(pcm) / sr < 0.5
    assert 0.1 < np.max(np.abs(pcm)) <= 0.5
    assert abs(pcm[-1]) < 1e-3  # no click at the end


def test_tick_is_short_and_soft():
    tick, sr = earcon("tick")
    wake, _ = earcon("wake")
    assert len(tick) / sr < 0.08
    assert np.max(np.abs(tick)) < np.max(np.abs(wake)) * 0.6


def test_every_name_the_daemon_uses_exists():
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "macd"
    used = set()
    for f in src.rglob("*.py"):
        if f.name != "earcons.py":
            used |= set(re.findall(r"""_play_earcon\(\s*["']([a-z_]+)["']""", f.read_text()))
            used |= set(re.findall(r"""\bearcon\(\s*["']([a-z_]+)["']""", f.read_text()))
    assert used, "no earcon call sites found"
    assert used <= set(NAMES), used - set(NAMES)


def test_earcons_distinct_and_cached():
    a, _ = earcon("wake")
    b, _ = earcon("done")
    assert a is earcon("wake")[0]
    assert len(a) != len(b) or not np.allclose(a, b)
    assert earcon("wake", 16000)[1] == 16000


def test_unknown():
    with pytest.raises(KeyError):
        earcon("nope")
