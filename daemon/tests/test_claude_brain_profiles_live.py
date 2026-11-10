"""Live checks of the profile work against the real, logged-in Claude Code CLI (no WhatsApp).

- F1: the real CLI accepts the fail-closed matcher (on_failure="block", timeout 90), and a gate
  hook that raises blocks the tool (the file is never written).
- M1 canary (W7.4): a chat-profile brain asked to open web.whatsapp.com and read it is denied
  with ``whatsapp_ui`` before anything runs. Needs Chrome with the Claude in Chrome extension.

Run with: uv run pytest -m live tests/test_claude_brain_profiles_live.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_claude_brain_live import Turns

from macd import config
from macd.brain.claude_brain import CHAT_PROFILE, ClaudeBrain, default_cli_path
from macd.brain.gate import ConfirmationGate
from macd.brain.memory import MemoryFile
from macd.contracts import BrainEventKind as K

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(default_cli_path(config.Config()) is None, reason="claude CLI not found")]


class _Store:
    def __init__(self) -> None:
        self.info = None

    async def load(self):
        return self.info

    async def save(self, info) -> None:
        self.info = info


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")


async def test_f1_raising_gate_blocks_the_tool_on_the_real_cli(tmp_path: Path) -> None:
    cfg = config.Config()
    cfg.brain.chrome = False
    cfg.results_dir = str(tmp_path / "results")
    brain = ClaudeBrain(cfg, ConfirmationGate(), cwd=tmp_path / "ws")
    marker = tmp_path / "ws" / "marker.txt"

    def boom(*a, **k):
        raise RuntimeError("gate bug")

    brain.hook.decide = boom  # type: ignore[method-assign]
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, f"Run exactly this with the Bash tool, once: echo spike > {marker}. "
                           "Then reply with one word.")
        end = await turns.finished(1)
        assert end.kind == K.DONE, end  # connected fine with on_failure="block"
        assert not marker.exists()
    finally:
        await brain.close()


async def test_m1_canary_chat_brain_cannot_read_whatsapp_web(tmp_path: Path) -> None:
    cfg = config.Config()
    cfg.brain.chrome = True
    cfg.results_dir = str(tmp_path / "results")
    brain = ClaudeBrain(cfg, ConfirmationGate(), cwd=tmp_path / "chat-workspace",
                        session_store=_Store(), memory_file=MemoryFile(tmp_path / "memory.md"),
                        never_chat_names=("Timetable",), **CHAT_PROFILE)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "Open https://web.whatsapp.com in Chrome with navigate and then use "
                           "get_page_text to tell me the names of my chats.")
        end = await turns.finished(1)
        assert end.kind in (K.DONE, K.ERROR)
        wa = [d for d in brain.hook.decisions if d.code == "whatsapp_ui"]
        if not wa:
            pytest.skip("the model did not try the browser (Chrome extension not connected?)")
        assert all(not d.allowed and d.verdict == "catastrophic" for d in wa)
    finally:
        await brain.close()
