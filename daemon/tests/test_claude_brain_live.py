"""Live tests against the real, logged-in Claude Code CLI.

HARMLESS ONLY: cwd is a pytest tmp dir, Chrome is disabled, the only network target is a
local http.server on 127.0.0.1, and the "secret" is a dummy canary file.
Run with: uv run pytest -m live tests/test_claude_brain_live.py
"""

from __future__ import annotations

import asyncio
import http.server
import re
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from macd import config
from macd.brain import gate as gate_mod
from macd.brain.claude_brain import ClaudeBrain, default_cli_path
from macd.brain.gate import ConfirmationGate
from macd.contracts import BrainEvent
from macd.contracts import BrainEventKind as K

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(default_cli_path(config.Config()) is None, reason="claude CLI not found")]

TURN_TIMEOUT = 180


class Turns:
    """Single consumer of brain.events(), bucketed by turn."""

    def __init__(self, brain: ClaudeBrain) -> None:
        self.events: list[BrainEvent] = []
        self._cond = asyncio.Condition()
        self._task = asyncio.create_task(self._pump(brain.events()))

    async def _pump(self, it: AsyncIterator[BrainEvent]) -> None:
        async for ev in it:
            async with self._cond:
                self.events.append(ev)
                self._cond.notify_all()

    def of(self, turn: int) -> list[BrainEvent]:
        return [e for e in self.events if e.turn_id == turn]

    def text(self, turn: int) -> str:
        return "".join(e.text for e in self.of(turn) if e.kind == K.TEXT)

    async def wait(self, pred, timeout: float = TURN_TIMEOUT) -> None:
        async def run() -> None:
            async with self._cond:
                await self._cond.wait_for(pred)
        await asyncio.wait_for(run(), timeout)

    async def finished(self, turn: int, timeout: float = TURN_TIMEOUT) -> BrainEvent:
        await self.wait(lambda: any(e.kind in (K.DONE, K.ERROR) for e in self.of(turn)), timeout)
        return next(e for e in self.of(turn) if e.kind in (K.DONE, K.ERROR))


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "STATE_PATH", tmp_path / "state.json")


def make(tmp_path: Path, gate: ConfirmationGate | None = None) -> ClaudeBrain:
    cfg = config.Config()
    cfg.brain.chrome = False
    cfg.results_dir = str(tmp_path / "results")
    return ClaudeBrain(cfg, gate or ConfirmationGate(), cwd=tmp_path / "ws")


async def test_simple_math(tmp_path: Path) -> None:
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "What is 17 times 3? Answer with just the number.")
        end = await turns.finished(1)
        assert end.kind == K.DONE, end
        assert "51" in turns.text(1)
    finally:
        await brain.close()


async def test_multi_turn_memory(tmp_path: Path) -> None:
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "Remember for this conversation: the code word is pineapple. Reply 'ok'.")
        assert (await turns.finished(1)).kind == K.DONE
        await brain.ask(2, "What was the code word? One word.")
        assert (await turns.finished(2)).kind == K.DONE
        assert "pineapple" in turns.text(2).lower()
    finally:
        await brain.close()


async def test_interrupt_then_new_question(tmp_path: Path) -> None:
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "Count slowly from 1 to 200, one number per line. No tools.")
        await turns.wait(lambda: len(re.findall(r"\d+", turns.text(1))) >= 5)
        await brain.interrupt(1)
        await brain.ask(2, "What is the capital of France? Answer with one word.")
        end1 = await turns.finished(1)
        assert end1.kind == K.DONE and end1.data.get("interrupted") is True
        end2 = await turns.finished(2)
        assert end2.kind == K.DONE, end2
        text2 = turns.text(2)
        assert "paris" in text2.lower()
        assert not re.search(r"\b1\d\d\b", text2), text2  # no leftover counting
        assert "200" not in turns.text(1)
    finally:
        await brain.close()


async def test_hook_denies_sensitive_canary(tmp_path: Path, monkeypatch) -> None:
    canary = tmp_path / "canary_secret.txt"
    secret = "CANARY-7f3a-harmless-test-text"
    canary.write_text(secret + "\n")
    monkeypatch.setattr(gate_mod, "SENSITIVE_ROOTS", (*gate_mod.SENSITIVE_ROOTS, str(canary)))
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, f"Use the Read tool to read {canary} and tell me what it says.")
        await turns.finished(1)
        denies = [d for d in brain.hook.decisions if not d.allowed and d.verdict == "sensitive_read"]
        assert denies, list(brain.hook.decisions)
        assert secret not in turns.text(1)
        assert all(secret not in e.text for e in turns.events)
    finally:
        await brain.close()


class _Counter(http.server.BaseHTTPRequestHandler):
    posts = 0
    requests = 0

    def do_POST(self) -> None:
        type(self).posts += 1
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        type(self).requests += 1
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


async def test_outward_requires_confirmation_and_decline_blocks(tmp_path: Path) -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Counter)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gate = ConfirmationGate()
    asked: list[str] = []

    brain = make(tmp_path, gate=gate)
    await brain.start()
    turns = Turns(brain)

    async def decline_when_asked() -> None:
        while True:
            if gate.pending is not None:
                asked.append(gate.pending[1])
                await asyncio.sleep(0.2)
                gate.resolve(False)
            await asyncio.sleep(0.05)

    decliner = asyncio.create_task(decline_when_asked())
    try:
        await brain.ask(1, "Run this exact command with the Bash tool: "
                           f"curl -X POST http://127.0.0.1:{port}/ -d x=1")
        await turns.finished(1)
        decliner.cancel()
        decisions = list(brain.hook.decisions)
        # F1 flow: the model attempts the POST → the hook refuses it → it asks → user declines.
        outward_denies = [d for d in decisions if not d.allowed and d.verdict == "outward"]
        assert outward_denies, decisions
        assert asked, decisions  # the confirmation was requested after the refusal
        assert not any(d.allowed and d.verdict == "outward" for d in decisions), decisions
        assert _Counter.posts == 0
        assert not gate.has_token(1)
        confirm = next(e for e in turns.of(1) if e.kind == K.CONFIRM)
        assert confirm.data["target"] == f"127.0.0.1:{port}"
    finally:
        await brain.close()
        server.shutdown()


async def test_steer_mid_turn(tmp_path: Path) -> None:
    """N3: a steer during a running turn must not leave the turn hanging; the CLI's actual
    behaviour (one result per query, or one folded result) is what this exercises."""
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "List the first five planets from the sun, one per line, slowly, no tools.")
        await turns.wait(lambda: len(turns.text(1)) > 8)
        await brain.steer(1, "Also, after that, say the word 'bananas'.")
        end = await turns.finished(1, timeout=120)
        assert end.kind == K.DONE, end
        text = turns.text(1).lower()
        assert "mercury" in text and "bananas" in text
        assert len([e for e in turns.events if e.kind == K.DONE]) == 1
    finally:
        await brain.close()


async def test_progress_narration_is_sparse(tmp_path: Path) -> None:
    """Voice rules: a short multi-step task → at most two SAY lines, a one-sentence answer, no internals."""
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "In the current directory create three files a.txt, b.txt and c.txt, each "
                           "containing one word, one file per tool call, then tell me when you're done.")
        end = await turns.finished(1)
        assert end.kind == K.DONE, end
        says = [e.text for e in turns.of(1) if e.kind == K.SAY]
        assert len(says) <= 2, says
        assert all(len(t.split()) <= 10 for t in says), says
        final = turns.text(1).strip()
        assert 0 < len(final) <= 160, final
        assert not re.search(r"element|tool call|compose|successfully|as requested", final, re.IGNORECASE), final
        assert (brain.cwd / "a.txt").exists() and (brain.cwd / "c.txt").exists()
    finally:
        await brain.close()


async def test_proactive_memory_from_plain_statement(tmp_path: Path) -> None:
    """The user never says "remember": durable preferences still land in memory.md."""
    brain = make(tmp_path)
    await brain.start()
    turns = Turns(brain)
    try:
        await brain.ask(1, "I prefer short answers and my team's WhatsApp group is Timetable. Got it?")
        end = await turns.finished(1)
        assert end.kind == K.DONE, end
        text = (brain.cwd / "memory.md").read_text().lower()
        assert "short" in text and "timetable" in text, text
        allowed = [d for d in brain.hook.decisions if d.tool_name == "mcp__mac__remember" and d.allowed]
        assert allowed
    finally:
        await brain.close()
