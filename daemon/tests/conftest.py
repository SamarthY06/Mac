"""Test-wide safety net: no test may touch the real ~/.mac-agent or ~/Library/Logs/Mac.

Every path constant in macd.config (except MODELS_DIR, which the `models`-marked tests
read) is redirected to a per-test temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from macd import config

_REDIRECTED = ("RUN_DIR", "WORKSPACE_DIR", "LOG_DIR", "CONFIG_PATH", "STATE_PATH", "MEMORY_PATH",
               "CONTROL_SOCK", "AUDIO_SOCK", "EVENT_LOG")


@pytest.fixture(autouse=True)
def _isolated_mac_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "mac-agent"
    run = home / "run"
    logs = tmp_path / "logs"
    paths = {
        "HOME": home,
        "RUN_DIR": run,
        "WORKSPACE_DIR": home / "workspace",
        "LOG_DIR": logs,
        "CONFIG_PATH": home / "config.toml",
        "STATE_PATH": home / "state.json",
        "MEMORY_PATH": home / "workspace" / "memory.md",
        "CONTROL_SOCK": run / "control.sock",
        "AUDIO_SOCK": run / "audio.sock",
        "EVENT_LOG": logs / "events.jsonl",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv("MAC_HOME", str(home))
    monkeypatch.setenv("MAC_LOG_DIR", str(logs))
    return home


@pytest.fixture(autouse=True)
def _no_keep_awake(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test holds the developer's Mac awake: a scheduled run starts no caffeinate."""
    from agentd import tasks

    monkeypatch.setattr(tasks, "KEEP_AWAKE_ARGV", ())
