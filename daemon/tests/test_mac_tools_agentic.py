"""Agentic-manager round 1, builder B: AG10 confirmation question, the model-owned voice settings
tools (AG15, AG16, AG20), switch_workspace (AG1), the persona split (AG11), language (AG12) and
the model-written narration (AG13)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import test_claude_brain_profiles as _profiles
from test_claude_brain_unit import MockClient, collect, result, simple

from macd import config
from macd.brain import claude_brain as cb
from macd.brain import mac_tools as mt
from macd.brain import prompts
from macd.brain.gate import ConfirmationGate, GateContext, analyze
from macd.brain.memory import grounded
from macd.brain.narration import describe_tool
from macd.contracts import BrainEventKind as K

globals()["make"] = _profiles.make  # share the `make` fixture (mocked SDK client, chat=True profile)


def pre(tool: str, inp: dict[str, Any] | None = None, tuid: str = "t") -> dict[str, Any]:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": inp or {}, "tool_use_id": tuid}


def text_of(res: dict[str, Any]) -> str:
    return res["content"][0]["text"]


# ----------------------------------------------------------------------------- AG10


async def test_confirm_event_carries_the_models_question(make) -> None:
    gate = ConfirmationGate()
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        await brain.hook(pre("Bash", {"command": "curl -X DELETE https://api.example.com/drafts/1"}), "t")
        asyncio.get_running_loop().call_later(0.01, gate.resolve, True)
        seen["res"] = await brain.tools.request_confirmation(
            {"summary": "delete the draft email to Bob", "question": "Delete the draft?"})
        c.push(result())

    brain = make(r, gate=gate)
    await brain.start()
    await brain.ask(1, "delete it")
    evs = await collect(brain)
    await brain.close()
    (conf,) = [e for e in evs if e.kind == K.CONFIRM]
    assert conf.text == "delete the draft email to Bob"
    assert conf.data == {"target": "api.example.com", "question": "Delete the draft?"}
    assert text_of(seen["res"]) == mt.APPROVED


async def test_confirm_without_question_sends_empty_question(make) -> None:
    gate = ConfirmationGate()

    async def r(c: MockClient, p: str) -> None:
        await brain.hook(pre("Bash", {"command": "curl -d m=hi https://chat.example/send"}), "t")
        asyncio.get_running_loop().call_later(0.01, gate.resolve, False)
        await brain.tools.request_confirmation({"summary": "send Bob hi"})
        c.push(result())

    brain = make(r, gate=gate)
    await brain.start()
    await brain.ask(1, "x")
    (conf,) = [e for e in await collect(brain) if e.kind == K.CONFIRM]
    await brain.close()
    assert conf.data["question"] == ""


@pytest.mark.parametrize("summary,question,ok", [
    ("send 'late' to Priya on WhatsApp", "Send it?", True),
    ("send 'late' to Priya on WhatsApp", "", True),
    (" ".join(["word"] * 21), "Send it?", False),  # too long: rewrite, never truncated
    ("x" * 201, "Send it?", False),
    ("send it", "Shall I send this message to Priya right now?", False),  # 9 words
    ("send it", "Send it", False),  # not a question
    ("send it", "Send it? Really?", False),
    ("send it", "Send it — now?", False),
    ("", "Send it?", False),
])
def test_confirmation_text_is_checked_not_cut(summary: str, question: str, ok: bool) -> None:
    assert (mt.check_confirmation(summary, question) is None) is ok


async def test_over_long_summary_is_a_tool_error_and_nothing_is_asked() -> None:
    gate = ConfirmationGate()
    calls: list[Any] = []
    tools = mt.MacTools(gate=gate, current_turn=lambda: 1, memory_check=lambda t, a: (False, ""),
                        on_say=lambda t: None, on_confirm=lambda *a: calls.append(a))
    res = await tools.request_confirmation({"summary": " ".join(["long"] * 30), "question": "Send it?"})
    assert res.get("is_error") and "20 words" in text_of(res)
    res = await tools.request_confirmation({"summary": "send it", "question": "Please could you let me send it?"})
    assert res.get("is_error") and "6 words" in text_of(res)
    assert calls == [] and gate.pending is None


def test_never_list_also_checks_the_question() -> None:
    ctx = GateContext(home="/Users/a", workspace="/Users/a/.mac-agent/w", user="a", never_chat_names=("Timetable",))
    d = analyze("mcp__mac__request_confirmation", {"summary": "post the notes", "question": "Post to Timetable?"}, ctx)
    assert d.code == "never_chat"


# ----------------------------------------------------------------------------- voice settings tools


def _tool_names(b: cb.ClaudeBrain) -> set[str]:
    return {t.name for t in b.tools.tools()}


async def test_rename_and_reset_voice_go_through_the_callbacks(tmp_path: Path) -> None:
    calls: list[Any] = []

    async def rename(name: str) -> str | None:
        calls.append(("rename", name))
        if name == "Bob":
            return "that name is taken by another device"  # macd's style: a returned reason
        if name == "Hal":
            raise ValueError("wake phrase 'hey hal' is too short")  # also accepted
        return None

    async def reset_voice() -> None:
        calls.append(("reset",))

    brain = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "ws",
                           on_rename=rename, on_reset_voice=reset_voice)
    assert {"rename_assistant", "reset_voice"} <= _tool_names(brain)
    assert "go_to_sleep" not in _tool_names(brain) and "switch_workspace" not in _tool_names(brain)
    ok = await brain.tools.rename_assistant({"name": "  Jarvis "})
    assert not ok.get("is_error") and "Renamed to Jarvis" in text_of(ok)
    bad = await brain.tools.rename_assistant({"name": "Jarvis!!"})  # settings.validate rules
    assert bad.get("is_error") and "32 letters" in text_of(bad)
    long = await brain.tools.rename_assistant({"name": "x" * 33})
    assert long.get("is_error")
    taken = await brain.tools.rename_assistant({"name": "Bob"})
    assert taken.get("is_error") and "taken" in text_of(taken)
    short = await brain.tools.rename_assistant({"name": "Hal"})
    assert short.get("is_error") and "too short" in text_of(short)
    reset = await brain.tools.reset_voice({})
    assert not reset.get("is_error") and "standard voice" in text_of(reset)
    assert calls == [("rename", "Jarvis"), ("rename", "Bob"), ("rename", "Hal"), ("reset",)]


def test_settings_tools_are_absent_without_callbacks_and_never_outward(tmp_path: Path) -> None:
    voice = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "v")
    assert not {"rename_assistant", "reset_voice", "go_to_sleep", "switch_workspace"} & _tool_names(voice)
    ctx = GateContext(home="/Users/a", workspace="/Users/a/.mac-agent/w", user="a")
    for tool, inp in (("mcp__mac__rename_assistant", {"name": "Jarvis"}), ("mcp__mac__reset_voice", {}),
                      ("mcp__mac__go_to_sleep", {}),
                      ("mcp__mac__switch_workspace", {"folder": "Resume", "mode": "new"})):
        assert analyze(tool, inp, ctx).verdict == "allow"


async def test_go_to_sleep_tool(tmp_path: Path) -> None:
    slept: list[bool] = []
    brain = cb.ClaudeBrain(config.Config(), ConfirmationGate(), cwd=tmp_path / "ws",
                           on_go_to_sleep=lambda: slept.append(True))
    res = await brain.tools.go_to_sleep({})
    assert not res.get("is_error") and slept == [True]


# ----------------------------------------------------------------------------- switch_workspace (AG1)


async def test_switch_workspace_emits_status_and_stops_further_tools(make) -> None:
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        seen["bad_mode"] = await brain.tools.switch_workspace({"folder": "Resume", "mode": "maybe"})
        seen["ok"] = await brain.tools.switch_workspace({"folder": "Resume", "mode": "continue"})
        seen["again"] = await brain.tools.switch_workspace({"folder": "Blog", "mode": "new"})
        seen["after"] = await brain.hook(pre("Bash", {"command": "ls"}), "x")
        c.push(result())

    brain = make(r, chat=True)
    assert "switch_workspace" in _tool_names(brain)
    await brain.start()
    await brain.ask(6, "find more jobs like last time")
    evs = await collect(brain)
    await brain.close()
    assert seen["bad_mode"].get("is_error") and not seen["ok"].get("is_error")
    assert seen["again"].get("is_error")
    assert seen["after"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    switches = [e for e in evs if e.kind == K.STATUS and "switch_workspace" in e.data]
    assert len(switches) == 1 and switches[0].turn_id == 6
    assert switches[0].data["switch_workspace"] == {"folder": "Resume", "mode": "continue", "tools_run": 0}


async def test_switch_workspace_rejects_bad_folders(make) -> None:
    brain = make(simple("x"), chat=True)
    for folder in ("", "a\nb", "x" * 201):
        res = await brain.tools.switch_workspace({"folder": folder, "mode": "new"})
        assert res.get("is_error")
    # outside a turn: refused, nothing emitted
    res = await brain.tools.switch_workspace({"folder": "Resume", "mode": "new"})
    assert res.get("is_error")


# ----------------------------------------------------------------------------- persona (AG11, AG12)


def test_chat_persona_is_written_not_spoken(tmp_path: Path) -> None:
    cfg = config.Config()
    chat = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md", channel="chat",
                                  never_chat_names=("Timetable",), tools={"send_file", "switch_workspace"})
    for spoken in ("spoken aloud", "Mac folder", "out loud", "Shall I go ahead", "web.whatsapp.com in Chrome"):
        assert spoken not in chat, spoken
    assert "WhatsApp style" in chat and "mcp__mac__send_file" in chat and "switch_workspace" in chat
    assert "Off-limits chats" in chat and "Timetable" not in chat  # enforced by the gate, never volunteered
    assert "Reply in the language the user writes in." in chat
    assert "—" not in chat and "–" not in chat
    voice = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert "spoken aloud" in voice and "Off-limits" not in voice and "WhatsApp style" not in voice
    assert "web.whatsapp.com" in voice  # voice keeps WhatsApp (V6, Option A)


def test_persona_mentions_only_the_settings_tools_it_has(tmp_path: Path) -> None:
    cfg = config.Config()
    none = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert "rename_assistant" not in none
    some = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md", tools={"rename_assistant", "reset_voice"})
    assert "mcp__mac__rename_assistant" in some and "mcp__mac__reset_voice" in some
    assert "go_to_sleep" not in some


async def test_brain_persona_uses_its_channel(make) -> None:
    brain = make(simple("ok"), chat=True, never_chat_names=("Timetable",))
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    await brain.close()
    persona = MockClient.instances[0].options.system_prompt["append"]
    assert "WhatsApp style" in persona and "Off-limits chats" in persona and "Timetable" not in persona and "spoken aloud" not in persona


def test_voice_language_note_makes_no_claim_about_the_user(tmp_path: Path) -> None:
    cfg = config.Config()
    cfg.tts.lang = "fr-fr"
    text = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md")
    assert "Speak French" in text and "English" not in text.split("## Language")[1].split("##")[0]
    chat = prompts.persona_prompt(cfg, memory_path=tmp_path / "m.md", channel="chat")
    assert "## Language" not in chat  # chat answers in the user's own language


def test_memory_grounding_beyond_english() -> None:
    assert grounded("मेरा नाम निखिल है", "हाय मैक, मेरा नाम निखिल है")
    assert not grounded("मेरी पत्नी का नाम प्रिया है", "मौसम कैसा है")
    assert grounded("Je préfère les réponses courtes", "Mac, je préfère les réponses courtes.")
    assert not grounded("Je préfère les réponses courtes", "quelle heure est-il ?")
    assert grounded("Prefers short answers", "I prefer short answers")  # English path unchanged


# ----------------------------------------------------------------------------- AG13 narration


@pytest.mark.parametrize("name,inp,expected", [
    ("Bash", {"command": "find ~/Downloads -name '*.pdf'", "description": "Search Downloads for the resume PDF"},
     "Search Downloads for the resume PDF"),
    ("Task", {"description": "research flights — London to Lisbon.", "prompt": "..."},
     "Research flights, London to Lisbon"),
    ("Bash", {"command": "ls", "description": "x" * 61}, "Running a command"),
    ("Bash", {"command": "mdfind resume", "description": "  "}, "Searching your files"),
    ("mcp__claude-in-chrome__navigate", {"url": "https://example.com", "description": "Go"}, "Opening example.com"),
    ("mcp__mac__say", {"text": "hi", "description": "x"}, None),
])
def test_narration_prefers_the_models_description(name: str, inp: dict[str, Any], expected: str | None) -> None:
    assert describe_tool(name, inp) == expected


# ----------------------------------------------------------------------------- round 2


async def test_clear_switch_lets_tools_run_again(make) -> None:
    seen: dict[str, Any] = {}

    async def r(c: MockClient, p: str) -> None:
        await brain.tools.switch_workspace({"folder": "Resume", "mode": "new"})
        seen["blocked"] = await brain.hook(pre("Bash", {"command": "ls"}), "a")
        seen["wrong_turn"] = brain.clear_switch(99)
        seen["cleared"] = brain.clear_switch()  # agentd refused the move, then steers why
        seen["again"] = brain.clear_switch()
        seen["allowed"] = await brain.hook(pre("Bash", {"command": "ls"}), "b")
        c.push(result())

    brain = make(r, chat=True)
    await brain.start()
    await brain.ask(3, "x")
    await collect(brain)
    await brain.close()
    assert seen["blocked"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (seen["wrong_turn"], seen["cleared"], seen["again"]) == (False, True, False)
    assert seen["allowed"] == {}
    assert brain.clear_switch() is False  # nothing running


async def test_on_alive_reports_connect_and_every_disconnect(make) -> None:
    from test_claude_brain_unit import EOF

    alive: list[bool] = []

    async def r(c: MockClient, p: str) -> None:
        if p == "crash":
            c.push(EOF)  # the CLI dies mid-turn
        else:
            c.push(result())

    brain = make(r, on_alive=alive.append)
    await brain.start()
    await brain.ask(1, "x")
    await collect(brain)
    assert alive == [True]
    await brain.sleep()  # idle shutdown path (park)
    for _ in range(100):
        if alive == [True, False]:
            break
        await asyncio.sleep(0.01)
    assert alive == [True, False]
    await brain.ask(2, "crash")
    await collect(brain)
    assert alive == [True, False, True, False]
    await brain.ask(3, "y")
    await collect(brain)
    await brain.close()
    assert alive == [True, False, True, False, True, False]


class EventLock:
    """agentd's DesktopHandle shape: polled try_acquire plus an awaited, event-driven acquire."""

    def __init__(self) -> None:
        self.busy = True
        self.tries = 0
        self.released = asyncio.Event()
        self.held = False

    def try_acquire(self) -> bool:
        self.tries += 1
        if self.busy:
            return False
        self.held = True
        return True

    def release(self) -> None:
        self.held = False

    async def acquire(self, timeout: float, cancelled: Any = lambda: False) -> bool:
        try:
            await asyncio.wait_for(self.released.wait(), timeout)
        except TimeoutError:
            return False
        if cancelled():
            return False
        self.busy, self.held = False, True
        return True


async def test_desktop_guard_awaits_the_event_driven_lock_without_polling() -> None:
    import time

    from macd.brain.hooks import DesktopGuard

    lock = EventLock()
    guard = DesktopGuard([lock], wait_s=5, poll_s=10)  # a poll would take 10 s: the test proves none
    asyncio.get_running_loop().call_later(0.05, lock.released.set)
    t0 = time.monotonic()
    assert await guard.acquire(1, "t", lambda: False) == "ok"
    assert time.monotonic() - t0 < 1 and lock.tries == 1 and guard.held

    busy = EventLock()
    g2 = DesktopGuard([busy], wait_s=0.1, poll_s=10)
    t0 = time.monotonic()
    assert await g2.acquire(1, "t", lambda: False) == "busy"
    assert time.monotonic() - t0 < 1

    stopped = EventLock()
    flag = {"stop": False}
    g3 = DesktopGuard([stopped], wait_s=5, poll_s=10)

    def stop_and_free() -> None:
        flag["stop"] = True
        stopped.released.set()
    asyncio.get_running_loop().call_later(0.05, stop_and_free)
    assert await g3.acquire(1, "t", lambda: flag["stop"]) == "cancelled"
    assert not g3.held


async def test_desktop_guard_chain_event_lock_then_flock(tmp_path: Path) -> None:
    from macd.brain.hooks import DesktopGuard, FlockLock

    ev = EventLock()
    ev.busy = False
    other = FlockLock(tmp_path / "desktop.lock")
    assert other.try_acquire()  # local voice holds the transitional flock
    guard = DesktopGuard([ev, FlockLock(tmp_path / "desktop.lock")], wait_s=0.3, poll_s=0.02)
    assert await guard.acquire(1, "t", lambda: False) == "busy"
    assert not ev.held  # the partial hold of agentd's lock is given back
    other.release()
    assert await guard.acquire(1, "t", lambda: False) == "ok"
    guard.release()
